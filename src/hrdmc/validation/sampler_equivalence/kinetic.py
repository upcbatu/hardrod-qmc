from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class _CutoffFit:
    model: str
    epsilon_count: int
    intercept_by_seed: tuple[float, ...]
    intercept: float
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "epsilon_count": self.epsilon_count,
            "intercept_by_seed": list(self.intercept_by_seed),
            "intercept": self.intercept,
        }
@dataclass(frozen=True)
class GradientCutoffExtrapolation:
    epsilon: tuple[float, ...]
    mean_by_epsilon: tuple[float, ...]
    excluded_probability_by_epsilon: tuple[float, ...]
    monotone_nonincreasing_with_epsilon: bool
    primary: _CutoffFit
    sensitivity_fits: tuple[_CutoffFit, ...]
    model_spread: float
    intercept_standard_error: float
    intercept: float
    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": list(self.epsilon),
            "mean_by_epsilon": list(self.mean_by_epsilon),
            "excluded_probability_by_epsilon": list(self.excluded_probability_by_epsilon),
            "monotone_nonincreasing_with_epsilon": (self.monotone_nonincreasing_with_epsilon),
            "primary": self.primary.to_dict(),
            "sensitivity_fits": [fit.to_dict() for fit in self.sensitivity_fits],
            "model_spread": self.model_spread,
            "intercept_standard_error": self.intercept_standard_error,
            "intercept": self.intercept,
            "uncertainty_scope": (
                "between-seed intercept variation plus separately reported model spread"
            ),
        }
def extrapolate_gradient_cutoff(
    epsilon: FloatArray,
    seed_block_values: FloatArray,
    seed_block_excluded_probability: FloatArray,
) -> GradientCutoffExtrapolation:
    """Extrapolate the bounded gradient estimator to zero free-gap cutoff."""
    eps = np.asarray(epsilon, dtype=float)
    values = np.asarray(seed_block_values, dtype=float)
    excluded = np.asarray(seed_block_excluded_probability, dtype=float)
    if (
        eps.ndim != 1
        or eps.size < 4
        or not np.all(np.isfinite(eps))
        or np.any(eps <= 0.0)
        or not np.all(np.diff(eps) > 0.0)
    ):
        raise ValueError("epsilon must contain at least four finite increasing cutoffs")
    if values.ndim != 3 or values.shape[2] != eps.size:
        raise ValueError("seed_block_values must have shape (seeds, blocks, cutoffs)")
    if excluded.shape != values.shape:
        raise ValueError("excluded probabilities must match seed_block_values")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("cutoff extrapolation requires at least two seeds and blocks")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(excluded))
        or np.any(excluded < 0.0)
        or np.any(excluded > 1.0)
    ):
        raise ValueError("cutoff observations must be finite and probabilities valid")
    primary = _fit(eps, values, degree=1, model="all_point_linear")
    small_linear = _fit(
        eps[:3],
        values[:, :, :3],
        degree=1,
        model="three_smallest_linear",
    )
    quadratic = _fit(eps, values, degree=2, model="all_point_quadratic")
    fits = (small_linear, quadratic)
    model_spread = float(max(abs(fit.intercept - primary.intercept) for fit in fits))
    intercepts = np.asarray(primary.intercept_by_seed, dtype=float)
    standard_error = float(np.std(intercepts, ddof=1) / np.sqrt(intercepts.size))
    mean_values = np.mean(values, axis=(0, 1))
    mean_excluded = np.mean(excluded, axis=(0, 1))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(mean_values))))
    monotone = bool(np.all(np.diff(mean_values) <= tolerance))
    return GradientCutoffExtrapolation(
        epsilon=tuple(float(value) for value in eps),
        mean_by_epsilon=tuple(float(value) for value in mean_values),
        excluded_probability_by_epsilon=tuple(float(value) for value in mean_excluded),
        monotone_nonincreasing_with_epsilon=monotone,
        primary=primary,
        sensitivity_fits=fits,
        model_spread=model_spread,
        intercept_standard_error=standard_error,
        intercept=primary.intercept,
    )
def _fit(
    epsilon: FloatArray,
    values: FloatArray,
    *,
    degree: int,
    model: str,
) -> _CutoffFit:
    intercepts = tuple(
        float(np.polynomial.polynomial.polyfit(epsilon, np.mean(seed, axis=0), degree)[0])
        for seed in values
    )
    return _CutoffFit(
        model=model,
        epsilon_count=int(epsilon.size),
        intercept_by_seed=intercepts,
        intercept=float(np.mean(intercepts)),
    )
