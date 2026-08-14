from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.trial.guide import BatchedDMCGuide

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class _MALAEvaluator(Protocol):
    """Batched target evaluations required by the shared MALA transition."""

    def gradient_local_valid(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, BoolArray]: ...
    def local_valid(self, positions: FloatArray) -> tuple[FloatArray, BoolArray]: ...
    def log_values(self, positions: FloatArray) -> tuple[FloatArray, BoolArray]: ...


@dataclass(frozen=True)
class _BatchedGuideMALAEvaluator:
    """Adapt a production batched guide to the minimal MALA evaluator seam."""

    guide: BatchedDMCGuide

    def gradient_local_valid(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        gradient, _laplacian, local_energy, finite = self.guide.batch_grad_lap_local(positions)
        return (
            np.asarray(gradient, dtype=float),
            np.asarray(local_energy, dtype=float),
            np.asarray(finite, dtype=bool),
        )

    def local_valid(self, positions: FloatArray) -> tuple[FloatArray, BoolArray]:
        _gradient, _laplacian, local_energy, finite = self.guide.batch_grad_lap_local(positions)
        return np.asarray(local_energy, dtype=float), np.asarray(finite, dtype=bool)

    def log_values(self, positions: FloatArray) -> tuple[FloatArray, BoolArray]:
        log_values, finite = self.guide.batch_log_value(positions)
        return np.asarray(log_values, dtype=float), np.asarray(finite, dtype=bool)


@dataclass(frozen=True)
class MALAStepResult:
    """One guide-squared MALA transition without branching or walker weights."""

    positions: FloatArray
    local_energies: FloatArray
    accepted: BoolArray
    domain_invalid_proposal: BoolArray
    invalid_proposal: BoolArray
    nonfinite_proposal: BoolArray
    metropolis_rejected: BoolArray
    drift_norm_max: float


def limit_drift(
    drift: FloatArray,
    dt: float,
    *,
    method: str = "none",
) -> FloatArray:
    """Limit proposal drift by Umrigar--Nightingale--Runge Eq. 33 [Umrigar1993]."""
    values = np.asarray(drift, dtype=float)
    if method == "none":
        return values
    if method != "umrigar":
        raise ValueError("drift limiter method must be 'none' or 'umrigar'")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    scale = 2.0 / (1.0 + np.hypot(1.0, np.sqrt(2.0 * dt) * values))
    return values * scale


def metropolis_log_acceptance(
    positions_old: FloatArray,
    positions_new: FloatArray,
    log_guide_old: FloatArray,
    log_guide_new: FloatArray,
    proposal_drift_old: FloatArray,
    proposal_drift_new: FloatArray,
    dt: float,
) -> FloatArray:
    """Return the Hastings-corrected log acceptance for guide-squared MALA."""
    forward_residual = positions_new - positions_old - dt * proposal_drift_old
    reverse_residual = positions_old - positions_new - dt * proposal_drift_new
    return (
        2.0 * (log_guide_new - log_guide_old)
        - 0.5
        * (
            np.sum(reverse_residual * reverse_residual, axis=1)
            - np.sum(forward_residual * forward_residual, axis=1)
        )
        / dt
    )


def mala_step(
    rng: np.random.Generator,
    positions: FloatArray,
    guide: BatchedDMCGuide,
    dt: float,
    local_energies: FloatArray,
    *,
    drift_limiter: str = "none",
) -> MALAStepResult:
    """Advance a batched guide-squared target by one whole-configuration MALA step."""
    return mala_step_with_evaluator(
        rng,
        positions,
        _BatchedGuideMALAEvaluator(guide),
        dt,
        local_energies,
        drift_limiter=drift_limiter,
    )


def mala_step_with_evaluator(
    rng: np.random.Generator,
    positions: FloatArray,
    evaluator: _MALAEvaluator,
    dt: float,
    local_energies: FloatArray,
    *,
    drift_limiter: str = "none",
) -> MALAStepResult:
    """Advance a guide-squared target through an explicit batched evaluator."""
    current, energies, grad_old = _validated_mala_state(positions, local_energies, evaluator, dt)
    proposal_drift_old = limit_drift(grad_old, dt, method=drift_limiter)
    trial = current + dt * proposal_drift_old + np.sqrt(dt) * rng.normal(size=current.shape)
    trial_energies, trial_valid = evaluator.local_valid(trial)
    _require_scalar_evaluation("trial", current.shape[0], trial_energies, trial_valid)
    accepted = np.zeros(current.shape[0], dtype=bool)
    position_finite = np.all(np.isfinite(trial), axis=1)
    domain_invalid_proposal = position_finite & (~trial_valid)
    nonfinite_proposal = (~position_finite) | (trial_valid & (~np.isfinite(trial_energies)))
    log_nonfinite_proposal = np.zeros(current.shape[0], dtype=bool)
    candidate_indices = np.flatnonzero(position_finite & trial_valid & np.isfinite(trial_energies))
    if candidate_indices.size:
        candidate_positions = trial[candidate_indices]
        grad_new, _new_energies, grad_new_valid = evaluator.gradient_local_valid(
            candidate_positions
        )
        _require_gradient_evaluation(
            "candidate",
            candidate_positions,
            grad_new,
            grad_new_valid,
        )
        finite_drift = grad_new_valid & np.all(np.isfinite(grad_new), axis=1)
        nonfinite_proposal[candidate_indices[~finite_drift]] = True
        candidate_indices = candidate_indices[finite_drift]
        grad_new = grad_new[finite_drift]
        if candidate_indices.size:
            proposal_drift_new = limit_drift(grad_new, dt, method=drift_limiter)
            log_old, log_old_valid = evaluator.log_values(current[candidate_indices])
            log_new, log_new_valid = evaluator.log_values(trial[candidate_indices])
            _require_log_evaluation(
                "current candidate",
                candidate_indices.size,
                log_old,
                log_old_valid,
            )
            _require_log_evaluation(
                "trial candidate",
                candidate_indices.size,
                log_new,
                log_new_valid,
            )
            finite_log = log_old_valid & log_new_valid & np.isfinite(log_old) & np.isfinite(log_new)
            with np.errstate(over="ignore", invalid="ignore"):
                log_acceptance = metropolis_log_acceptance(
                    current[candidate_indices],
                    trial[candidate_indices],
                    log_old,
                    log_new,
                    proposal_drift_old[candidate_indices],
                    proposal_drift_new,
                    dt,
                )
            log_uniform = np.log(rng.random(candidate_indices.size))
            finite_acceptance = np.isfinite(log_acceptance)
            acceptance_evaluable = finite_log & finite_acceptance
            accepted[candidate_indices] = acceptance_evaluable & (
                log_uniform <= np.minimum(log_acceptance, 0.0)
            )
            log_nonfinite_proposal[candidate_indices[~finite_log]] = True
            nonfinite_proposal[candidate_indices[finite_log & (~finite_acceptance)]] = True
    return _mala_result(
        current=current,
        energies=energies,
        trial=trial,
        trial_energies=trial_energies,
        accepted=accepted,
        domain_invalid_proposal=domain_invalid_proposal,
        nonfinite_proposal=nonfinite_proposal,
        log_nonfinite_proposal=log_nonfinite_proposal,
        grad_old=grad_old,
    )


def _validated_mala_state(
    positions: FloatArray,
    local_energies: FloatArray,
    evaluator: _MALAEvaluator,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    current = np.asarray(positions, dtype=float)
    energies = np.asarray(local_energies, dtype=float)
    if current.ndim != 2 or current.shape[0] <= 0 or current.shape[1] <= 0:
        raise ValueError("positions must have shape (walkers, particles)")
    if energies.shape != (current.shape[0],):
        raise ValueError("local_energies must have one value per walker")
    grad_old, _old_energies, old_valid = evaluator.gradient_local_valid(current)
    _require_gradient_evaluation("current", current, grad_old, old_valid)
    if not np.all(old_valid):
        raise ValueError("MALA requires valid input walkers")
    if not np.all(np.isfinite(grad_old)):
        raise ValueError("MALA requires finite guide drift")
    return current, energies, grad_old


def _mala_result(
    *,
    current: FloatArray,
    energies: FloatArray,
    trial: FloatArray,
    trial_energies: FloatArray,
    accepted: NDArray[np.bool_],
    domain_invalid_proposal: NDArray[np.bool_],
    nonfinite_proposal: NDArray[np.bool_],
    log_nonfinite_proposal: NDArray[np.bool_],
    grad_old: FloatArray,
) -> MALAStepResult:
    nonfinite_proposal |= log_nonfinite_proposal
    invalid_proposal = domain_invalid_proposal | (nonfinite_proposal & (~log_nonfinite_proposal))
    vmc_metropolis_rejected = ~accepted & ~domain_invalid_proposal & ~nonfinite_proposal
    vmc_partition = (
        accepted.astype(np.int8)
        + domain_invalid_proposal.astype(np.int8)
        + nonfinite_proposal.astype(np.int8)
        + vmc_metropolis_rejected.astype(np.int8)
    )
    if not np.all(vmc_partition == 1):
        raise RuntimeError("MALA outcomes do not partition walker attempts")
    return MALAStepResult(
        positions=np.where(accepted[:, np.newaxis], trial, current),
        local_energies=np.where(accepted, trial_energies, energies),
        accepted=accepted,
        domain_invalid_proposal=domain_invalid_proposal,
        invalid_proposal=invalid_proposal,
        nonfinite_proposal=nonfinite_proposal,
        metropolis_rejected=~accepted & ~invalid_proposal,
        drift_norm_max=float(np.max(np.linalg.norm(grad_old, axis=1))),
    )


def _require_gradient_evaluation(
    label: str,
    positions: FloatArray,
    gradient: FloatArray,
    valid: BoolArray,
) -> None:
    if gradient.shape != positions.shape:
        raise ValueError(f"{label} guide gradient has the wrong shape")
    if valid.shape != (positions.shape[0],):
        raise ValueError(f"{label} guide validity has the wrong shape")


def _require_scalar_evaluation(
    label: str,
    walkers: int,
    values: FloatArray,
    valid: BoolArray,
) -> None:
    if values.shape != (walkers,):
        raise ValueError(f"{label} guide local energy has the wrong shape")
    if valid.shape != (walkers,):
        raise ValueError(f"{label} guide validity has the wrong shape")


def _require_log_evaluation(
    label: str,
    walkers: int,
    values: FloatArray,
    valid: BoolArray,
) -> None:
    if values.shape != (walkers,):
        raise ValueError(f"{label} guide log amplitude has the wrong shape")
    if valid.shape != (walkers,):
        raise ValueError(f"{label} guide log validity has the wrong shape")
