from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from hrdmc.trial.guide import BatchedDMCGuide


@dataclass(frozen=True)
class VMCConfig:
    """Evolution controls for a batched guide-squared VMC run."""
    walkers: int
    burn_in_steps: int
    production_steps: int
    method: str = "mala"
    dt: float = 0.01
    step_size: float = 0.5
    drift_limiter: str = "none"
    def validate(self) -> None:
        if self.walkers <= 0:
            raise ValueError("walkers must be positive")
        if self.burn_in_steps < 0:
            raise ValueError("burn_in_steps must be non-negative")
        if self.production_steps <= 0:
            raise ValueError("production_steps must be positive")
        if self.method not in {"rwm", "mala"}:
            raise ValueError("method must be 'rwm' or 'mala'")
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")
        if self.method == "rwm":
            if not np.isfinite(self.step_size) or self.step_size <= 0.0:
                raise ValueError("step_size must be finite and positive for RWM")
            if self.drift_limiter != "none":
                raise ValueError("drift_limiter is only supported for MALA")
        elif not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive for MALA")

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
@dataclass(frozen=True)
class RWMStepResult:
    """One batched random-scan single-particle random-walk transition."""
    positions: FloatArray
    log_values: FloatArray
    particle_indices: IntArray
    accepted: BoolArray
    invalid_proposal: BoolArray
    nonfinite_proposal: BoolArray
    metropolis_rejected: BoolArray
def require_batched_guide(guide: object) -> BatchedDMCGuide:
    """Reject guides that would require a Python per-walker fallback."""
    required_methods = ("valid_batch", "batch_log_value", "batch_grad_lap_local")
    missing = [name for name in required_methods if not callable(getattr(guide, name, None))]
    if missing:
        names = ", ".join(missing)
        raise TypeError(f"production VMC requires batched guide methods: {names}")
    return cast(BatchedDMCGuide, guide)
def random_scan_rwm_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: BatchedDMCGuide,
    step_size: float,
    log_values: FloatArray,
) -> RWMStepResult:
    """Apply one symmetric one-particle RWM proposal to every walker."""
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    current = np.asarray(positions, dtype=float)
    current_log = np.asarray(log_values, dtype=float)
    if current.ndim != 2 or current.shape[0] <= 0 or current.shape[1] <= 0:
        raise ValueError("positions must have shape (walkers, particles)")
    walkers, particles = current.shape
    if current_log.shape != (walkers,) or not np.all(np.isfinite(current_log)):
        raise ValueError("log_values must be finite with one value per walker")
    if not np.all(np.isfinite(current)):
        raise ValueError("RWM requires finite input walkers")
    particle_indices = np.asarray(
        rng.integers(0, particles, size=walkers),
        dtype=np.int64,
    )
    displacements = np.asarray(
        rng.uniform(-step_size, step_size, size=walkers),
        dtype=float,
    )
    trial = current.copy()
    trial[np.arange(walkers), particle_indices] += displacements
    position_finite = np.all(np.isfinite(trial), axis=1)
    domain_valid = np.asarray(guide.valid_batch(trial), dtype=bool)
    trial_log, log_finite = guide.batch_log_value(trial)
    trial_log = np.asarray(trial_log, dtype=float)
    log_finite = np.asarray(log_finite, dtype=bool)
    _require_batch_scalar("valid_batch", walkers, domain_valid)
    _require_batch_scalar("batch_log_value values", walkers, trial_log)
    _require_batch_scalar("batch_log_value validity", walkers, log_finite)
    nonfinite_proposal = (~position_finite) | (
        domain_valid & ((~log_finite) | (~np.isfinite(trial_log)))
    )
    invalid_proposal = position_finite & (~domain_valid) & (~nonfinite_proposal)
    candidate = (
        position_finite
        & domain_valid
        & log_finite
        & np.isfinite(trial_log)
        & (~invalid_proposal)
        & (~nonfinite_proposal)
    )
    candidate_indices = np.flatnonzero(candidate)
    accepted = np.zeros(walkers, dtype=bool)
    if candidate_indices.size:
        with np.errstate(over="ignore", invalid="ignore"):
            log_acceptance = 2.0 * (trial_log[candidate_indices] - current_log[candidate_indices])
        log_uniform = np.log(rng.random(candidate_indices.size))
        finite_acceptance = np.isfinite(log_acceptance)
        accepted[candidate_indices] = finite_acceptance & (
            log_uniform <= np.minimum(log_acceptance, 0.0)
        )
        nonfinite_proposal[candidate_indices[~finite_acceptance]] = True
    metropolis_rejected = candidate & (~accepted) & (~nonfinite_proposal)
    partition = (
        accepted.astype(np.int8)
        + invalid_proposal.astype(np.int8)
        + nonfinite_proposal.astype(np.int8)
        + metropolis_rejected.astype(np.int8)
    )
    if not np.all(partition == 1):
        raise RuntimeError("RWM outcomes do not partition walker attempts")
    return RWMStepResult(
        positions=np.where(accepted[:, np.newaxis], trial, current),
        log_values=np.where(accepted, trial_log, current_log),
        particle_indices=particle_indices,
        accepted=accepted,
        invalid_proposal=invalid_proposal,
        nonfinite_proposal=nonfinite_proposal,
        metropolis_rejected=metropolis_rejected,
    )
def _require_batch_scalar(label: str, walkers: int, values: NDArray[np.generic]) -> None:
    if values.shape != (walkers,):
        raise ValueError(f"{label} must return one value per walker")
