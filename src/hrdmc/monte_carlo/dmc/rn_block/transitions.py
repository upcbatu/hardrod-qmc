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
from hrdmc.monte_carlo.dmc.common.numeric import finite_mean, finite_variance
from hrdmc.monte_carlo.dmc.common.population import require_live_weight
from hrdmc.monte_carlo.dmc.rn_block._collective import sample_collective_mixture
from hrdmc.monte_carlo.dmc.rn_block._telemetry import (
    AdvanceResult,
    StepTelemetry,
)
from hrdmc.monte_carlo.dmc.rn_block.config import RNBlockDMCConfig
from hrdmc.monte_carlo.dmc.rn_block.weights import (
    importance_sampled_rn_log_increment,
    rn_log_increment,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.systems.propagators import ProposalTransitionKernel, TargetTransitionKernel
from hrdmc.wavefunctions import DMCGuide

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RNBlockLocalStepResult:
    positions: FloatArray
    local_energies: FloatArray
    killed: NDArray[np.bool_]


class RNBlockLocalStep(Protocol):
    def __call__(
        self,
        rng: np.random.Generator,
        positions: FloatArray,
        guide: DMCGuide,
        dt: float,
        local_energies: FloatArray,
    ) -> RNBlockLocalStepResult: ...


def euler_drift_diffusion_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: DMCGuide,
    dt: float,
    local_energies: FloatArray,
) -> RNBlockLocalStepResult:
    """Default local DMC proposal using guide drift and hard fail on invalid trials."""

    grad, _current_energies, _current_valid = guide_grad_energy_valid(guide, positions)
    drift = 2.0 * grad
    trial = positions + dt * drift + np.sqrt(2.0 * dt) * rng.normal(size=positions.shape)
    trial_energies, valid = evaluate_guide(guide, trial)
    killed = ~valid
    return RNBlockLocalStepResult(
        positions=np.where(killed[:, np.newaxis], positions, trial),
        local_energies=np.where(killed, local_energies, trial_energies),
        killed=killed,
    )


def advance_rn_block(
    config: RNBlockDMCConfig,
    system: OpenLineHardRodSystem,
    guide: DMCGuide,
    target_kernel: TargetTransitionKernel,
    proposal_kernel: ProposalTransitionKernel,
    rng: np.random.Generator,
    positions: FloatArray,
    local_energies: FloatArray,
    log_weights: FloatArray,
    *,
    include_guide_ratio: bool,
) -> AdvanceResult:
    old_positions = positions
    proposal = sample_collective_mixture(config, system, proposal_kernel, rng, old_positions)
    target_log_density = target_kernel.log_density(
        old_positions,
        proposal.x_new,
        config.tau_block,
    )
    if include_guide_ratio:
        increment = importance_sampled_rn_log_increment(
            target_log_density,
            proposal.log_q_forward,
            guide_log_values(guide, old_positions),
            guide_log_values(guide, proposal.x_new),
        )
    else:
        increment = rn_log_increment(target_log_density, proposal.log_q_forward)
    trial_energies, valid = evaluate_guide(guide, proposal.x_new)
    killed = (~valid) | (~np.isfinite(increment))
    next_log_weights = np.where(killed, -np.inf, log_weights + increment)
    require_live_weight(next_log_weights)
    return AdvanceResult(
        positions=np.where(killed[:, np.newaxis], old_positions, proposal.x_new),
        local_energies=np.where(killed, local_energies, trial_energies),
        log_weights=next_log_weights,
        killed=killed,
        telemetry=StepTelemetry(
            rn_logk_mean=finite_mean(target_log_density),
            rn_logq_mean=finite_mean(proposal.log_q_forward),
            rn_logw_increment_mean=finite_mean(increment),
            rn_logw_increment_variance=finite_variance(increment),
        ),
    )


def advance_local_step(
    local_step: RNBlockLocalStep,
    guide: DMCGuide,
    rng: np.random.Generator,
    positions: FloatArray,
    local_energies: FloatArray,
    log_weights: FloatArray,
    dt: float,
) -> AdvanceResult:
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
    return AdvanceResult(
        positions=np.where(killed[:, np.newaxis], positions, next_positions),
        local_energies=np.where(killed, local_energies, next_energies),
        log_weights=next_log_weights,
        killed=killed,
        telemetry=StepTelemetry(),
    )
