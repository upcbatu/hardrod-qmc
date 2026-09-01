from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from hrdmc.trial.kernel import reduced_tg_closed_form_local_energy_batch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RelativeWidthSample:
    """Sufficient statistics for correlated sampling over relative alpha."""

    base_local_energy: FloatArray
    internal_norm2: FloatArray

    def __post_init__(self) -> None:
        base = _readonly_vector(self.base_local_energy, "base_local_energy")
        norm2 = _readonly_vector(self.internal_norm2, "internal_norm2")
        if base.shape != norm2.shape:
            raise ValueError("relative-width sufficient statistics must share one shape")
        if base.size == 0:
            raise ValueError("relative-width sample must not be empty")
        if np.any(norm2 < 0.0):
            raise ValueError("internal_norm2 must be non-negative")
        object.__setattr__(self, "base_local_energy", base)
        object.__setattr__(self, "internal_norm2", norm2)

    @property
    def size(self) -> int:
        return int(self.base_local_energy.size)

    def local_energy(
        self,
        *,
        relative_alpha: float,
        omega: float,
        n_particles: int,
    ) -> FloatArray:
        _require_positive_finite("relative_alpha", relative_alpha)
        _require_positive_finite("omega", omega)
        if n_particles < 2:
            raise ValueError("n_particles must be at least two")
        delta = float(relative_alpha) - float(omega)
        constant = 0.5 * delta * (n_particles * n_particles - 1)
        curvature = float(omega) * delta + 0.5 * delta * delta
        return self.base_local_energy + constant - curvature * self.internal_norm2


def relative_width_sample(
    positions: FloatArray,
    *,
    rod_length: float,
    omega: float,
) -> RelativeWidthSample:
    """Reduce guide-squared configurations to the two alpha-dependent statistics."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("positions must have shape (configurations, particles)")
    if not np.all(np.isfinite(values)):
        raise ValueError("positions must be finite")
    if not np.isfinite(rod_length) or rod_length < 0.0:
        raise ValueError("rod_length must be finite and non-negative")
    _require_positive_finite("omega", omega)
    if np.any(np.diff(values, axis=1) < rod_length):
        raise ValueError("positions must satisfy the ordered hard-rod domain")
    n_particles = values.shape[1]
    offsets = rod_length * (
        np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1)
    )
    reduced = values - offsets[np.newaxis, :]
    internal = reduced - np.mean(reduced, axis=1, keepdims=True)
    return RelativeWidthSample(
        base_local_energy=reduced_tg_closed_form_local_energy_batch(
            values,
            rod_length=rod_length,
            omega=omega,
        ),
        internal_norm2=np.sum(internal * internal, axis=1),
    )


def relative_alpha_metrics(
    sample: RelativeWidthSample,
    *,
    omega: float,
    n_particles: int,
    reference_relative_alpha: float,
    relative_alpha: float,
) -> dict[str, float]:
    """Evaluate one alpha with the historical correlated-sampling objective."""
    _require_positive_finite("reference_relative_alpha", reference_relative_alpha)
    _require_positive_finite("relative_alpha", relative_alpha)
    log_weights = -(
        float(relative_alpha) - float(reference_relative_alpha)
    ) * sample.internal_norm2
    shifted = log_weights - float(np.max(log_weights))
    raw_weights = np.exp(shifted)
    weights = raw_weights / float(np.sum(raw_weights))
    reweight_ess = float(1.0 / np.sum(weights * weights))
    local_energy = sample.local_energy(
        relative_alpha=relative_alpha,
        omega=omega,
        n_particles=n_particles,
    )
    mean = float(np.sum(weights * local_energy))
    centered = local_energy - mean
    variance = float(np.sum(weights * centered * centered))
    return {
        "relative_alpha": float(relative_alpha),
        "local_energy_mean": mean,
        "local_energy_variance": variance,
        "local_energy_std": float(math.sqrt(max(variance, 0.0))),
        "reweight_ess": reweight_ess,
        "reweight_ess_fraction": reweight_ess / sample.size,
        "max_normalized_weight": float(np.max(weights)),
    }


def optimize_relative_alpha(
    sample: RelativeWidthSample,
    *,
    omega: float,
    n_particles: int,
    reference_relative_alpha: float,
    alpha_log_half_width: float = 0.2,
    alpha_grid_points: int = 31,
    min_reweight_ess_fraction: float = 0.10,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, Any]]:
    """Minimize reweighted local-energy variance within a bounded alpha window."""
    _require_positive_finite("reference_relative_alpha", reference_relative_alpha)
    _require_positive_finite("alpha_log_half_width", alpha_log_half_width)
    if alpha_grid_points < 5:
        raise ValueError("alpha_grid_points must be at least five")
    if not 0.0 < min_reweight_ess_fraction <= 1.0:
        raise ValueError("min_reweight_ess_fraction must lie in (0, 1]")
    alpha_low = reference_relative_alpha * math.exp(-alpha_log_half_width)
    alpha_high = reference_relative_alpha * math.exp(alpha_log_half_width)
    alpha_values = np.exp(
        np.linspace(math.log(alpha_low), math.log(alpha_high), alpha_grid_points)
    )
    rows = [
        relative_alpha_metrics(
            sample,
            omega=omega,
            n_particles=n_particles,
            reference_relative_alpha=reference_relative_alpha,
            relative_alpha=float(alpha),
        )
        for alpha in alpha_values
    ]
    eligible_rows = [
        row
        for row in rows
        if row["reweight_ess_fraction"] >= min_reweight_ess_fraction
    ]
    grid_best = min(
        eligible_rows or rows,
        key=lambda row: row["local_energy_variance"],
    )

    def objective(log_alpha: float) -> float:
        metrics = relative_alpha_metrics(
            sample,
            omega=omega,
            n_particles=n_particles,
            reference_relative_alpha=reference_relative_alpha,
            relative_alpha=math.exp(log_alpha),
        )
        variance = max(metrics["local_energy_variance"], np.finfo(float).tiny)
        shortfall = max(
            0.0,
            min_reweight_ess_fraction - metrics["reweight_ess_fraction"],
        )
        return float(math.log(variance) + 1.0e4 * shortfall * shortfall)

    refinement = minimize_scalar(
        objective,
        bounds=(math.log(alpha_low), math.log(alpha_high)),
        method="bounded",
        options={"xatol": 1.0e-8, "maxiter": 500},
    )
    refined = relative_alpha_metrics(
        sample,
        omega=omega,
        n_particles=n_particles,
        reference_relative_alpha=reference_relative_alpha,
        relative_alpha=float(np.clip(math.exp(refinement.x), alpha_low, alpha_high)),
    )
    eligible_candidates = [
        candidate
        for candidate in (grid_best, refined)
        if candidate["reweight_ess_fraction"] >= min_reweight_ess_fraction
    ]
    optimum = min(
        eligible_candidates or [grid_best, refined],
        key=lambda candidate: candidate["local_energy_variance"],
    )
    at_boundary = bool(
        optimum["relative_alpha"] <= alpha_low * 1.001
        or optimum["relative_alpha"] >= alpha_high / 1.001
    )
    overlap_ok = optimum["reweight_ess_fraction"] >= min_reweight_ess_fraction
    status = "optimization_candidate"
    if not overlap_ok:
        status = "reweight_overlap_insufficient"
    elif at_boundary:
        status = "reference_recenter_required"
    return rows, optimum, {
        "status": status,
        "alpha_bounds": [alpha_low, alpha_high],
        "relative_alpha_at_boundary": at_boundary,
        "refinement_success": bool(refinement.success),
        "refinement_message": str(refinement.message),
        "refinement_metrics": refined,
        "selected_candidate_source": "refinement" if optimum is refined else "grid",
        "minimum_reweight_ess_fraction": min_reweight_ess_fraction,
    }


def _readonly_vector(values: object, name: str) -> FloatArray:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _require_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
