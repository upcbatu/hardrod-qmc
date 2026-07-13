from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.common.guide_api import (
    evaluate_guide,
    guide_grad_energy_valid,
    guide_log_values,
)
from hrdmc.monte_carlo.dmc.common.population import require_live_weight
from hrdmc.monte_carlo.dmc.local.mobility import local_step_mobility
from hrdmc.monte_carlo.dmc.local.telemetry import (
    DMCAdvanceResult,
    DMCStepTelemetry,
)
from hrdmc.wavefunctions.api import DMCGuide

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DMCStepResult:
    positions: FloatArray
    local_energies: FloatArray
    killed: NDArray[np.bool_]
    accepted: NDArray[np.bool_] | None = None
    invalid_proposal: NDArray[np.bool_] | None = None
    metropolis_rejected: NDArray[np.bool_] | None = None
    drift_norm_max: float = float("nan")


class DMCStep(Protocol):
    def __call__(
        self,
        rng: np.random.Generator,
        positions: FloatArray,
        guide: DMCGuide,
        dt: float,
        local_energies: FloatArray,
    ) -> DMCStepResult: ...


def euler_drift_diffusion_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: DMCGuide,
    dt: float,
    local_energies: FloatArray,
) -> DMCStepResult:
    """Euler importance-sampled drift-diffusion step in oscillator units."""

    grad, _current_energies, _current_valid = guide_grad_energy_valid(guide, positions)
    trial = positions + dt * grad + np.sqrt(dt) * rng.normal(size=positions.shape)
    trial_energies, valid = evaluate_guide(guide, trial)
    killed = ~valid
    return DMCStepResult(
        positions=np.where(killed[:, np.newaxis], positions, trial),
        local_energies=np.where(killed, local_energies, trial_energies),
        killed=killed,
        accepted=~killed,
        invalid_proposal=killed,
        metropolis_rejected=np.zeros(positions.shape[0], dtype=bool),
        drift_norm_max=float(np.max(np.linalg.norm(grad, axis=1))),
    )


def metropolis_drift_diffusion_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: DMCGuide,
    dt: float,
    local_energies: FloatArray,
) -> DMCStepResult:
    """MALA importance-sampled drift-diffusion step.

    Invalid hard-core proposals and Metropolis rejections leave the previous
    valid walker live; branching is applied afterwards by ``advance_local_step``.
    """

    grad_old, _old_energies, old_valid = guide_grad_energy_valid(guide, positions)
    if not np.all(old_valid):
        raise ValueError("metropolis local step requires valid input walkers")
    if not np.all(np.isfinite(grad_old)):
        raise ValueError("metropolis local step requires finite guide drift")

    trial = positions + dt * grad_old + np.sqrt(dt) * rng.normal(size=positions.shape)
    trial_energies, trial_valid = evaluate_guide(guide, trial)
    accepted = np.zeros(positions.shape[0], dtype=bool)
    invalid_proposal = ~trial_valid
    candidate_indices = np.flatnonzero(trial_valid)

    if candidate_indices.size:
        candidate_positions = trial[candidate_indices]
        grad_new, _new_energies, grad_new_valid = guide_grad_energy_valid(
            guide,
            candidate_positions,
        )
        finite_drift = grad_new_valid & np.all(np.isfinite(grad_new), axis=1)
        invalid_proposal[candidate_indices[~finite_drift]] = True
        candidate_indices = candidate_indices[finite_drift]
        grad_new = grad_new[finite_drift]
        if candidate_indices.size:
            log_old = guide_log_values(guide, positions[candidate_indices])
            log_new = guide_log_values(guide, trial[candidate_indices])
            forward_residual = (
                trial[candidate_indices]
                - positions[candidate_indices]
                - dt * grad_old[candidate_indices]
            )
            reverse_residual = (
                positions[candidate_indices] - trial[candidate_indices] - dt * grad_new
            )
            log_acceptance = (
                2.0 * (log_new - log_old)
                - 0.5
                * (
                    np.sum(reverse_residual * reverse_residual, axis=1)
                    - np.sum(forward_residual * forward_residual, axis=1)
                )
                / dt
            )
            log_uniform = np.log(rng.random(candidate_indices.size))
            accepted[candidate_indices] = log_uniform <= np.minimum(log_acceptance, 0.0)

    return DMCStepResult(
        positions=np.where(accepted[:, np.newaxis], trial, positions),
        local_energies=np.where(accepted, trial_energies, local_energies),
        killed=np.zeros(positions.shape[0], dtype=bool),
        accepted=accepted,
        invalid_proposal=invalid_proposal,
        metropolis_rejected=~accepted & ~invalid_proposal,
        drift_norm_max=float(np.max(np.linalg.norm(grad_old, axis=1))),
    )


def advance_local_step(
    local_step: DMCStep,
    guide: DMCGuide,
    rng: np.random.Generator,
    positions: FloatArray,
    local_energies: FloatArray,
    log_weights: FloatArray,
    dt: float,
    *,
    center: float = 0.0,
    rod_length: float = 0.0,
) -> DMCAdvanceResult:
    result = local_step(rng, positions, guide, dt, local_energies)
    next_positions = np.asarray(result.positions, dtype=float)
    next_energies = np.asarray(result.local_energies, dtype=float)
    killed = np.asarray(result.killed, dtype=bool) | (~np.isfinite(next_energies))
    if next_positions.shape != positions.shape:
        raise ValueError("local step returned positions with the wrong shape")
    if next_energies.shape != local_energies.shape or killed.shape != local_energies.shape:
        raise ValueError("local step returned energy/killed arrays with the wrong shape")
    reference_energy = float(np.mean(local_energies[np.isfinite(local_energies)]))
    increment = -dt * (0.5 * (local_energies + next_energies) - reference_energy)
    next_log_weights = np.where(
        killed | (~np.isfinite(increment)),
        -np.inf,
        log_weights + increment,
    )
    require_live_weight(next_log_weights)
    accepted = result.accepted
    invalid_proposal = result.invalid_proposal
    metropolis_rejected = result.metropolis_rejected
    mobility = local_step_mobility(
        positions,
        next_positions,
        center=center,
        rod_length=rod_length,
    )
    return DMCAdvanceResult(
        positions=np.where(killed[:, np.newaxis], positions, next_positions),
        local_energies=np.where(killed, local_energies, next_energies),
        log_weights=next_log_weights,
        killed=killed,
        telemetry=DMCStepTelemetry(
            local_acceptance_fraction=(
                float(np.mean(accepted)) if accepted is not None else float("nan")
            ),
            invalid_proposal_fraction=(
                float(np.mean(invalid_proposal)) if invalid_proposal is not None else float("nan")
            ),
            metropolis_rejection_fraction=(
                float(np.mean(metropolis_rejected))
                if metropolis_rejected is not None
                else float("nan")
            ),
            drift_norm_max=result.drift_norm_max,
            configuration_esjd=mobility.configuration_esjd,
            r2_esjd=mobility.r2_esjd,
            weighted_free_gap_esjd=mobility.weighted_free_gap_esjd,
        ),
    )
