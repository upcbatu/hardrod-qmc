from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]
EQUIVALENT = "equivalent"
NOT_EQUIVALENT = "not_equivalent"
INSUFFICIENT_INFORMATION = "insufficient_information"
BOOTSTRAP_COVERAGE_SCOPE = (
    "nominal nonparametric-bootstrap uncertainty radius with Bonferroni "
    "allocation; not an exact finite-sample coverage guarantee"
)
@dataclass(frozen=True)
class UnpairedScalarEquivalenceResult:
    """Welch interval and practical-equivalence decision for independent seeds."""
    status: str
    first_count: int
    second_count: int
    familywise_confidence: float
    family_size: int
    per_comparison_confidence: float
    practical_margin: float
    observed_difference: float
    standard_error: float
    degrees_of_freedom: float
    critical_value: float
    uncertainty_half_width: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    absolute_upper_bound: float
    @property
    def equivalent(self) -> bool:
        return self.status == EQUIVALENT
    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "equivalent": self.equivalent,
            "first_count": self.first_count,
            "second_count": self.second_count,
            "familywise_confidence": self.familywise_confidence,
            "family_size": self.family_size,
            "per_comparison_confidence": self.per_comparison_confidence,
            "practical_margin": self.practical_margin,
            "observed_difference": self.observed_difference,
            "standard_error": self.standard_error,
            "degrees_of_freedom": self.degrees_of_freedom,
            "critical_value": self.critical_value,
            "uncertainty_half_width": self.uncertainty_half_width,
            "lower_confidence_bound": self.lower_confidence_bound,
            "upper_confidence_bound": self.upper_confidence_bound,
            "absolute_upper_bound": self.absolute_upper_bound,
        }
@dataclass(frozen=True)
class UnpairedVectorEquivalenceResult:
    """Independent-replicate bootstrap bound for a weighted profile norm."""
    status: str
    first_count: int
    second_count: int
    feature_count: int
    familywise_confidence: float
    family_size: int
    per_comparison_confidence: float
    practical_margin: float
    rng_seed: int
    bootstrap_replicates: int
    mean_difference: tuple[float, ...]
    scale_norm: float
    observed_discrepancy: float
    bootstrap_uncertainty_radius: float
    upper_confidence_bound: float
    @property
    def equivalent(self) -> bool:
        return self.status == EQUIVALENT
    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "equivalent": self.equivalent,
            "first_count": self.first_count,
            "second_count": self.second_count,
            "feature_count": self.feature_count,
            "familywise_confidence": self.familywise_confidence,
            "family_size": self.family_size,
            "per_comparison_confidence": self.per_comparison_confidence,
            "practical_margin": self.practical_margin,
            "rng_seed": self.rng_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "mean_difference": list(self.mean_difference),
            "scale_norm": self.scale_norm,
            "observed_discrepancy": self.observed_discrepancy,
            "bootstrap_uncertainty_radius": self.bootstrap_uncertainty_radius,
            "upper_confidence_bound": self.upper_confidence_bound,
            "bootstrap_coverage_scope": BOOTSTRAP_COVERAGE_SCOPE,
        }
def unpaired_scalar_equivalence(
    first_group: FloatArray,
    second_group: FloatArray,
    *,
    practical_margin: float,
    familywise_confidence: float = 0.95,
    family_size: int = 1,
) -> UnpairedScalarEquivalenceResult:
    """Test equivalence of independent scalar seed groups with a Welch interval."""
    margin, confidence, comparisons = _validated_policy(
        practical_margin=practical_margin,
        familywise_confidence=familywise_confidence,
        family_size=family_size,
    )
    first = _canonical_scalars(first_group, "first_group")
    second = _canonical_scalars(second_group, "second_group")
    per_comparison_confidence = 1.0 - (1.0 - confidence) / comparisons
    observed_difference = _difference_if_available(first, second)
    if first.size < 2 or second.size < 2:
        return _insufficient_scalar_result(
            first_count=first.size,
            second_count=second.size,
            confidence=confidence,
            comparisons=comparisons,
            per_comparison_confidence=per_comparison_confidence,
            margin=margin,
            observed_difference=observed_difference,
        )
    first_variance = float(np.var(first, ddof=1))
    second_variance = float(np.var(second, ddof=1))
    first_component = first_variance / first.size
    second_component = second_variance / second.size
    standard_error_squared = first_component + second_component
    standard_error = float(np.sqrt(standard_error_squared))
    if standard_error_squared == 0.0:
        degrees_of_freedom = float("inf")
    else:
        denominator = 0.0
        if first_component > 0.0:
            denominator += first_component * first_component / (first.size - 1)
        if second_component > 0.0:
            denominator += second_component * second_component / (second.size - 1)
        degrees_of_freedom = float(standard_error_squared**2 / denominator)
    tail_probability = (1.0 - confidence) / (2.0 * comparisons)
    critical_value = float(student_t.ppf(1.0 - tail_probability, degrees_of_freedom))
    uncertainty_half_width = float(critical_value * standard_error)
    lower = float(observed_difference - uncertainty_half_width)
    upper = float(observed_difference + uncertainty_half_width)
    absolute_upper = float(max(abs(lower), abs(upper)))
    status = _three_way_status(
        observed_discrepancy=abs(observed_difference),
        upper_bound=absolute_upper,
        practical_margin=margin,
    )
    return UnpairedScalarEquivalenceResult(
        status=status,
        first_count=int(first.size),
        second_count=int(second.size),
        familywise_confidence=confidence,
        family_size=comparisons,
        per_comparison_confidence=per_comparison_confidence,
        practical_margin=margin,
        observed_difference=observed_difference,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        critical_value=critical_value,
        uncertainty_half_width=uncertainty_half_width,
        lower_confidence_bound=lower,
        upper_confidence_bound=upper,
        absolute_upper_bound=absolute_upper,
    )
def unpaired_vector_equivalence(
    first_group: FloatArray,
    second_group: FloatArray,
    *,
    feature_weights: FloatArray,
    scale_profile: FloatArray,
    practical_margin: float,
    rng_seed: int,
    bootstrap_replicates: int = 10_000,
    familywise_confidence: float = 0.95,
    family_size: int = 1,
) -> UnpairedVectorEquivalenceResult:
    """Bound an independent-group weighted relative profile discrepancy."""
    margin, confidence, comparisons = _validated_policy(
        practical_margin=practical_margin,
        familywise_confidence=familywise_confidence,
        family_size=family_size,
    )
    if not isinstance(rng_seed, (int, np.integer)) or int(rng_seed) < 0:
        raise ValueError("rng_seed must be an explicit non-negative integer")
    if not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be an integer of at least 100")
    first, second = _canonical_profiles(first_group, second_group)
    feature_count = first.shape[1]
    weights = np.asarray(feature_weights, dtype=float)
    scale = np.asarray(scale_profile, dtype=float)
    if weights.shape != (feature_count,) or scale.shape != (feature_count,):
        raise ValueError("feature_weights and scale_profile must match the profile width")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(scale)) or np.any(weights <= 0.0):
        raise ValueError("profile weights must be positive and all scale inputs must be finite")
    scale_norm = _weighted_norm(scale, weights)
    if scale_norm <= 0.0:
        raise ValueError("scale_profile must have a positive weighted norm")
    per_comparison_confidence = 1.0 - (1.0 - confidence) / comparisons
    mean_difference = _profile_difference_if_available(first, second, feature_count)
    observed_discrepancy = _weighted_norm(mean_difference, weights) / scale_norm
    if first.shape[0] < 2 or second.shape[0] < 2:
        return UnpairedVectorEquivalenceResult(
            status=INSUFFICIENT_INFORMATION,
            first_count=int(first.shape[0]),
            second_count=int(second.shape[0]),
            feature_count=feature_count,
            familywise_confidence=confidence,
            family_size=comparisons,
            per_comparison_confidence=per_comparison_confidence,
            practical_margin=margin,
            rng_seed=int(rng_seed),
            bootstrap_replicates=int(bootstrap_replicates),
            mean_difference=tuple(float(value) for value in mean_difference),
            scale_norm=scale_norm,
            observed_discrepancy=observed_discrepancy,
            bootstrap_uncertainty_radius=float("nan"),
            upper_confidence_bound=float("nan"),
        )
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    rng = np.random.default_rng(int(rng_seed))
    error_norms = np.empty(int(bootstrap_replicates), dtype=float)
    chunk_size = 256
    for start in range(0, int(bootstrap_replicates), chunk_size):
        stop = min(start + chunk_size, int(bootstrap_replicates))
        count = stop - start
        first_indices = rng.integers(0, first.shape[0], size=(count, first.shape[0]))
        second_indices = rng.integers(0, second.shape[0], size=(count, second.shape[0]))
        first_bootstrap = np.mean(first[first_indices], axis=1)
        second_bootstrap = np.mean(second[second_indices], axis=1)
        errors = (first_bootstrap - first_mean) - (second_bootstrap - second_mean)
        error_norms[start:stop] = np.sqrt(np.sum(errors * errors * weights, axis=1)) / scale_norm
    uncertainty_radius = float(np.quantile(error_norms, per_comparison_confidence, method="higher"))
    upper_bound = float(observed_discrepancy + uncertainty_radius)
    status = _three_way_status(
        observed_discrepancy=observed_discrepancy,
        upper_bound=upper_bound,
        practical_margin=margin,
    )
    return UnpairedVectorEquivalenceResult(
        status=status,
        first_count=int(first.shape[0]),
        second_count=int(second.shape[0]),
        feature_count=feature_count,
        familywise_confidence=confidence,
        family_size=comparisons,
        per_comparison_confidence=per_comparison_confidence,
        practical_margin=margin,
        rng_seed=int(rng_seed),
        bootstrap_replicates=int(bootstrap_replicates),
        mean_difference=tuple(float(value) for value in mean_difference),
        scale_norm=scale_norm,
        observed_discrepancy=observed_discrepancy,
        bootstrap_uncertainty_radius=uncertainty_radius,
        upper_confidence_bound=upper_bound,
    )
def _validated_policy(
    *,
    practical_margin: float,
    familywise_confidence: float,
    family_size: int,
) -> tuple[float, float, int]:
    if not np.isfinite(practical_margin) or practical_margin < 0.0:
        raise ValueError("practical_margin must be finite and non-negative")
    if not np.isfinite(familywise_confidence) or not 0.0 < familywise_confidence < 1.0:
        raise ValueError("familywise_confidence must lie strictly between zero and one")
    if not isinstance(family_size, (int, np.integer)) or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    return float(practical_margin), float(familywise_confidence), int(family_size)
def _canonical_scalars(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional seed array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.sort(array)
def _canonical_profiles(
    first_group: FloatArray,
    second_group: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    first = np.asarray(first_group, dtype=float)
    second = np.asarray(second_group, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != second.shape[1:]:
        raise ValueError("profile groups must be matrices with one common feature width")
    if first.shape[1] < 1:
        raise ValueError("profile groups must contain at least one feature")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("profile groups must contain only finite values")
    return _sort_rows(first), _sort_rows(second)
def _sort_rows(values: FloatArray) -> FloatArray:
    if values.shape[0] < 2:
        return values.copy()
    keys = tuple(values[:, index] for index in range(values.shape[1] - 1, -1, -1))
    return values[np.lexsort(keys)]
def _difference_if_available(first: FloatArray, second: FloatArray) -> float:
    if first.size == 0 or second.size == 0:
        return float("nan")
    return float(np.mean(first) - np.mean(second))
def _profile_difference_if_available(
    first: FloatArray,
    second: FloatArray,
    feature_count: int,
) -> FloatArray:
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.full(feature_count, np.nan)
    return np.asarray(np.mean(first, axis=0) - np.mean(second, axis=0), dtype=float)
def _weighted_norm(values: FloatArray, weights: FloatArray) -> float:
    return float(np.sqrt(np.sum(np.asarray(values, dtype=float) ** 2 * weights)))
def _three_way_status(
    *,
    observed_discrepancy: float,
    upper_bound: float,
    practical_margin: float,
) -> str:
    if not np.isfinite(observed_discrepancy) or not np.isfinite(upper_bound):
        return INSUFFICIENT_INFORMATION
    if upper_bound <= practical_margin:
        return EQUIVALENT
    if observed_discrepancy > practical_margin:
        return NOT_EQUIVALENT
    return INSUFFICIENT_INFORMATION
def _insufficient_scalar_result(
    *,
    first_count: int,
    second_count: int,
    confidence: float,
    comparisons: int,
    per_comparison_confidence: float,
    margin: float,
    observed_difference: float,
) -> UnpairedScalarEquivalenceResult:
    return UnpairedScalarEquivalenceResult(
        status=INSUFFICIENT_INFORMATION,
        first_count=int(first_count),
        second_count=int(second_count),
        familywise_confidence=confidence,
        family_size=comparisons,
        per_comparison_confidence=per_comparison_confidence,
        practical_margin=margin,
        observed_difference=observed_difference,
        standard_error=float("nan"),
        degrees_of_freedom=float("nan"),
        critical_value=float("nan"),
        uncertainty_half_width=float("nan"),
        lower_confidence_bound=float("nan"),
        upper_confidence_bound=float("nan"),
        absolute_upper_bound=float("nan"),
    )
