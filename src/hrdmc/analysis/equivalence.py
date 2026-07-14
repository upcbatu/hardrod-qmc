from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PairwiseEquivalenceBound:
    first_index: int
    second_index: int
    mean_difference: float
    standard_error: float
    upper_bound: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "mean_difference": self.mean_difference,
            "standard_error": self.standard_error,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class SimultaneousPairwiseEquivalenceResult:
    equivalent: bool
    replicate_count: int
    condition_count: int
    pair_count: int
    confidence_level: float
    critical_value: float
    equivalence_margin: float
    observed_max_difference: float
    simultaneous_upper_bound: float
    pairwise_bounds: tuple[PairwiseEquivalenceBound, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "equivalent": self.equivalent,
            "replicate_count": self.replicate_count,
            "condition_count": self.condition_count,
            "pair_count": self.pair_count,
            "confidence_level": self.confidence_level,
            "critical_value": self.critical_value,
            "equivalence_margin": self.equivalence_margin,
            "observed_max_difference": self.observed_max_difference,
            "simultaneous_upper_bound": self.simultaneous_upper_bound,
            "pairwise_bounds": [bound.to_dict() for bound in self.pairwise_bounds],
        }


@dataclass(frozen=True)
class PairwiseNormEquivalenceBound:
    first_index: int
    second_index: int
    mean_relative_norm: float
    standard_error: float
    upper_bound: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "mean_relative_norm": self.mean_relative_norm,
            "standard_error": self.standard_error,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class SimultaneousPairwiseNormEquivalenceResult:
    equivalent: bool
    replicate_count: int
    condition_count: int
    pair_count: int
    confidence_level: float
    critical_value: float
    equivalence_margin: float
    observed_max_relative_norm: float
    simultaneous_upper_bound: float
    pairwise_bounds: tuple[PairwiseNormEquivalenceBound, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "equivalent": self.equivalent,
            "replicate_count": self.replicate_count,
            "condition_count": self.condition_count,
            "pair_count": self.pair_count,
            "confidence_level": self.confidence_level,
            "critical_value": self.critical_value,
            "equivalence_margin": self.equivalence_margin,
            "observed_max_relative_norm": self.observed_max_relative_norm,
            "simultaneous_upper_bound": self.simultaneous_upper_bound,
            "pairwise_bounds": [bound.to_dict() for bound in self.pairwise_bounds],
        }


def simultaneous_pairwise_equivalence(
    replicate_values: FloatArray,
    *,
    equivalence_margin: float,
    confidence_level: float = 0.95,
) -> SimultaneousPairwiseEquivalenceResult:
    """Bound every paired condition difference with family-wise coverage.

    Rows are independent replicates and columns are repeated conditions on the
    same replicate.  Each two-sided paired Student-t interval receives a
    Bonferroni share of the requested family-wise error rate.  Equivalence is
    established only when the largest absolute-difference upper bound lies
    inside the declared practical margin.
    """

    values = np.asarray(replicate_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("replicate_values must be a two-dimensional array")
    replicate_count, condition_count = values.shape
    if replicate_count < 2:
        raise ValueError("pairwise equivalence requires at least two independent replicates")
    if condition_count < 2:
        raise ValueError("pairwise equivalence requires at least two conditions")
    if not np.all(np.isfinite(values)):
        raise ValueError("replicate_values must be finite")
    if not np.isfinite(equivalence_margin) or equivalence_margin < 0.0:
        raise ValueError("equivalence_margin must be finite and non-negative")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    pairs = tuple(combinations(range(condition_count), 2))
    pair_count = len(pairs)
    familywise_alpha = 1.0 - confidence_level
    critical_value = float(
        student_t.ppf(
            1.0 - familywise_alpha / (2.0 * pair_count),
            df=replicate_count - 1,
        )
    )
    bounds: list[PairwiseEquivalenceBound] = []
    for first_index, second_index in pairs:
        differences = values[:, first_index] - values[:, second_index]
        mean_difference = float(np.mean(differences))
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(float(replicate_count)))
        upper_bound = float(abs(mean_difference) + critical_value * standard_error)
        bounds.append(
            PairwiseEquivalenceBound(
                first_index=first_index,
                second_index=second_index,
                mean_difference=mean_difference,
                standard_error=standard_error,
                upper_bound=upper_bound,
            )
        )

    observed_max_difference = max(abs(bound.mean_difference) for bound in bounds)
    simultaneous_upper_bound = max(bound.upper_bound for bound in bounds)
    return SimultaneousPairwiseEquivalenceResult(
        equivalent=simultaneous_upper_bound <= equivalence_margin,
        replicate_count=replicate_count,
        condition_count=condition_count,
        pair_count=pair_count,
        confidence_level=float(confidence_level),
        critical_value=critical_value,
        equivalence_margin=float(equivalence_margin),
        observed_max_difference=float(observed_max_difference),
        simultaneous_upper_bound=float(simultaneous_upper_bound),
        pairwise_bounds=tuple(bounds),
    )


def simultaneous_pairwise_norm_equivalence(
    replicate_vectors: FloatArray,
    *,
    feature_weights: FloatArray,
    scale_vector: FloatArray,
    equivalence_margin: float,
    confidence_level: float = 0.95,
) -> SimultaneousPairwiseNormEquivalenceResult:
    """Bound paired relative vector distances with family-wise coverage.

    Rows are independent replicates, the second axis contains repeated
    conditions, and the final axis contains a vector observable. For each
    condition pair, the per-replicate weighted L2 distance is normalized by
    the declared common scale vector. A Bonferroni paired-Student upper bound
    must lie inside the practical relative-norm margin for every pair.
    """

    values = np.asarray(replicate_vectors, dtype=float)
    weights = np.asarray(feature_weights, dtype=float)
    scale = np.asarray(scale_vector, dtype=float)
    if values.ndim != 3:
        raise ValueError("replicate_vectors must be a three-dimensional array")
    replicate_count, condition_count, feature_count = values.shape
    if replicate_count < 2:
        raise ValueError("norm equivalence requires at least two independent replicates")
    if condition_count < 2:
        raise ValueError("norm equivalence requires at least two conditions")
    if weights.shape != (feature_count,) or scale.shape != (feature_count,):
        raise ValueError("feature weights and scale vector must match the vector shape")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(scale))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("norm-equivalence inputs must be finite with positive weights")
    if not np.isfinite(equivalence_margin) or equivalence_margin < 0.0:
        raise ValueError("equivalence_margin must be finite and non-negative")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    scale_norm = float(np.sqrt(np.sum(scale * scale * weights)))
    if not np.isfinite(scale_norm) or scale_norm <= 0.0:
        raise ValueError("scale vector must have a positive weighted norm")

    pairs = tuple(combinations(range(condition_count), 2))
    pair_count = len(pairs)
    familywise_alpha = 1.0 - confidence_level
    critical_value = float(
        student_t.ppf(
            1.0 - familywise_alpha / (2.0 * pair_count),
            df=replicate_count - 1,
        )
    )
    bounds: list[PairwiseNormEquivalenceBound] = []
    for first_index, second_index in pairs:
        differences = values[:, first_index, :] - values[:, second_index, :]
        distances = np.sqrt(np.sum(differences * differences * weights, axis=1)) / scale_norm
        mean_distance = float(np.mean(distances))
        standard_error = float(np.std(distances, ddof=1) / np.sqrt(float(replicate_count)))
        upper_bound = float(mean_distance + critical_value * standard_error)
        bounds.append(
            PairwiseNormEquivalenceBound(
                first_index=first_index,
                second_index=second_index,
                mean_relative_norm=mean_distance,
                standard_error=standard_error,
                upper_bound=upper_bound,
            )
        )

    observed_max_relative_norm = max(bound.mean_relative_norm for bound in bounds)
    simultaneous_upper_bound = max(bound.upper_bound for bound in bounds)
    return SimultaneousPairwiseNormEquivalenceResult(
        equivalent=simultaneous_upper_bound <= equivalence_margin,
        replicate_count=replicate_count,
        condition_count=condition_count,
        pair_count=pair_count,
        confidence_level=float(confidence_level),
        critical_value=critical_value,
        equivalence_margin=float(equivalence_margin),
        observed_max_relative_norm=float(observed_max_relative_norm),
        simultaneous_upper_bound=float(simultaneous_upper_bound),
        pairwise_bounds=tuple(bounds),
    )
