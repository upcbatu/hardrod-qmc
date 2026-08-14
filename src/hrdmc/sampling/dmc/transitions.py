from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.sampling.dmc.guide_api import (
    evaluate_guide,
    guide_grad_energy_valid,
    guide_log_values,
)
from hrdmc.sampling.dmc.population import require_live_weight
from hrdmc.sampling.dmc.telemetry import (
    DMCAdvanceResult,
    DMCStepTelemetry,
)
from hrdmc.sampling.mala import (
    limit_drift as _shared_limit_drift,
)
from hrdmc.sampling.mala import (
    mala_step_with_evaluator,
    metropolis_log_acceptance,
)
from hrdmc.sampling.mobility import local_step_mobility
from hrdmc.trial.guide import DMCGuide


@dataclass(frozen=True)
class DMCConfig:
    """Algorithm controls for local importance-sampled DMC."""
    ess_resample_fraction: float = 0.35
    drift_limiter: str = "none"
    def validate(self) -> None:
        if not np.isfinite(self.ess_resample_fraction):
            raise ValueError("ess_resample_fraction must be finite")
        if not 0.0 <= self.ess_resample_fraction <= 1.0:
            raise ValueError("ess_resample_fraction must satisfy 0 <= fraction <= 1")
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")

FloatArray = NDArray[np.float64]
limit_drift = _shared_limit_drift
_metropolis_log_acceptance = metropolis_log_acceptance
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
@dataclass(frozen=True)
class _DMCGuideMALAEvaluator:
    """Preserve the established DMC guide adapter while sharing MALA mechanics."""
    guide: DMCGuide
    def gradient_local_valid(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        return guide_grad_energy_valid(self.guide, positions)
    def local_valid(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        return evaluate_guide(self.guide, positions)
    def log_values(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        values = guide_log_values(self.guide, positions)
        return values, np.isfinite(values)
def metropolis_drift_diffusion_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: DMCGuide,
    dt: float,
    local_energies: FloatArray,
    *,
    drift_limiter: str = "none",
) -> DMCStepResult:
    """MALA importance-sampled drift-diffusion step."""
    result = mala_step_with_evaluator(
        rng,
        positions,
        _DMCGuideMALAEvaluator(guide),
        dt,
        local_energies,
        drift_limiter=drift_limiter,
    )
    return DMCStepResult(
        positions=result.positions,
        local_energies=result.local_energies,
        killed=np.zeros(positions.shape[0], dtype=bool),
        accepted=result.accepted,
        invalid_proposal=result.invalid_proposal,
        metropolis_rejected=result.metropolis_rejected,
        drift_norm_max=result.drift_norm_max,
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
