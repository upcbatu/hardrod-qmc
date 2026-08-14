from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from hrdmc.statistics.population_bound import (
    PopulationDifferenceBound,
    PopulationEnergyPoint,
    _bounded_at_resolution,
    _student_critical_value,
    _validate_reporting_controls,
    population_difference_bound,
)


@dataclass(frozen=True)
class _InversePopulationFit:
    intercept: float
    slope: float
    intercept_stderr: float
    slope_stderr: float
    seed_intercept_standard_error: float
    seed_slope_standard_error: float
    intercept_source_run_quadrature_standard_error: float
    slope_source_run_quadrature_standard_error: float
    intercept_worst_case_arbitrary_covariance_standard_error_envelope: float
    slope_worst_case_arbitrary_covariance_standard_error_envelope: float
    residual_contrast_mean: float
    residual_contrast_seed_standard_error: float
    residual_contrast_source_run_quadrature_standard_error: float
    residual_contrast_worst_case_arbitrary_covariance_standard_error_envelope: float
    residual_contrast_standard_error: float
    residual_zero_test_pvalue: float
    residual_zero_test_alpha: float
    residual_zero_test_status: str
    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "model": "E(W) = E_infinity + c / W"}

@dataclass(frozen=True)
class _RichardsonWindowAssessment:
    low_population_intercept: float
    high_population_intercept: float
    mean_difference: float
    observed_absolute_difference: float
    paired_standard_error: float
    source_run_quadrature_standard_error: float
    worst_case_arbitrary_covariance_standard_error_envelope: float
    conservative_standard_error: float
    confidence_level: float
    degrees_of_freedom: int
    critical_value: float
    upper_allowance: float
    reporting_resolution: float
    bounded_below_reporting_resolution: bool
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class _PopulationLimitCorrection:
    reference_walkers: int
    value: float
    matched_seed_standard_error: float
    source_run_quadrature_standard_error: float
    worst_case_arbitrary_covariance_standard_error_envelope: float
    conservative_standard_error: float
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class PopulationLadderAssessment:
    points: tuple[PopulationEnergyPoint, ...]
    reference_walkers: int
    last_doubling: PopulationDifferenceBound
    inverse_population_fit: _InversePopulationFit | None
    richardson_window: _RichardsonWindowAssessment | None
    population_limit_correction: _PopulationLimitCorrection | None
    classification: str
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "classification": self.classification,
            "reference_walkers": self.reference_walkers,
            "points": [point.to_dict() for point in self.points],
            "last_doubling": self.last_doubling.to_dict(),
            "inverse_population_fit": (
                None
                if self.inverse_population_fit is None
                else self.inverse_population_fit.to_dict()
            ),
            "richardson_window": (
                None if self.richardson_window is None else self.richardson_window.to_dict()
            ),
            "population_limit_correction": (
                None
                if self.population_limit_correction is None
                else self.population_limit_correction.to_dict()
            ),
        }
        if self.classification == "accepted_population_limit":
            assert self.inverse_population_fit is not None
            assert self.richardson_window is not None
            assert self.population_limit_correction is not None
            payload.update(
                {
                    "candidate_population_limit_energy_at_fixed_timestep": (
                        self.inverse_population_fit.intercept
                    ),
                    "candidate_population_limit_energy_statistical_stderr": (
                        self.inverse_population_fit.intercept_stderr
                    ),
                    "candidate_population_limit_model_window_upper_allowance": (
                        self.richardson_window.upper_allowance
                    ),
                    "candidate_population_limit_correction_at_reference_population": (
                        self.population_limit_correction.value
                    ),
                    "candidate_population_limit_correction_statistical_stderr": (
                        self.population_limit_correction.conservative_standard_error
                    ),
                }
            )
        return payload

@dataclass(frozen=True)
class TimeStepPopulationInteraction:
    fine_timestep_difference: PopulationDifferenceBound
    coarse_timestep_difference: PopulationDifferenceBound
    interaction_difference: float
    observed_absolute_interaction: float
    fine_difference_standard_error: float
    coarse_difference_standard_error: float
    statistical_error_method: str
    interaction_statistical_standard_error: float
    interaction_source_run_quadrature_standard_error: float
    interaction_worst_case_arbitrary_covariance_standard_error_envelope: float
    interaction_standard_error: float
    confidence_level: float
    degrees_of_freedom: int
    critical_value: float
    upper_allowance: float
    reporting_resolution: float
    bounded_below_reporting_resolution: bool
    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fine_timestep_difference": self.fine_timestep_difference.to_dict(),
            "coarse_timestep_difference": self.coarse_timestep_difference.to_dict(),
        }

def analyze_population_ladder(
    points: Sequence[PopulationEnergyPoint],
    *,
    reporting_resolution: float,
    confidence_level: float = 0.95,
    fit_alpha: float = 0.05,
) -> PopulationLadderAssessment:
    """Assess a ``W, 2W`` or ``W/2, W, 2W`` energy ladder."""
    validated = _validated_points(points)
    _validate_reporting_controls(
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
        fit_alpha=fit_alpha,
    )
    if len(validated) not in {2, 3}:
        raise ValueError("population ladder requires two or three walker populations")
    if len(validated) == 2:
        reference, doubled = validated
        _require_doubling(reference.walkers, doubled.walkers)
        last_doubling = population_difference_bound(
            reference,
            doubled,
            reporting_resolution=reporting_resolution,
            confidence_level=confidence_level,
        )
        classification = (
            "accepted_finite_population_bound"
            if last_doubling.bounded_below_reporting_resolution
            else "additional_population_point_required"
        )
        return PopulationLadderAssessment(
            points=validated,
            reference_walkers=reference.walkers,
            last_doubling=last_doubling,
            inverse_population_fit=None,
            richardson_window=None,
            population_limit_correction=None,
            classification=classification,
        )
    half, reference, doubled = validated
    _require_doubling(half.walkers, reference.walkers)
    _require_doubling(reference.walkers, doubled.walkers)
    if not (half.seed_ids == reference.seed_ids == doubled.seed_ids):
        raise ValueError("all three population points must use identical seed ids")
    last_doubling = population_difference_bound(
        reference,
        doubled,
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
    )
    fit, population_limit_correction = _inverse_population_fit(
        validated,
        fit_alpha=fit_alpha,
    )
    richardson = _richardson_window_assessment(
        half,
        reference,
        doubled,
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
    )
    if (
        fit.residual_zero_test_status == "residual_not_statistically_resolved"
        and richardson.bounded_below_reporting_resolution
    ):
        classification = "accepted_population_limit"
    elif last_doubling.bounded_below_reporting_resolution:
        classification = "accepted_finite_population_bound"
    else:
        classification = "population_sensitive"
    return PopulationLadderAssessment(
        points=validated,
        reference_walkers=reference.walkers,
        last_doubling=last_doubling,
        inverse_population_fit=fit,
        richardson_window=richardson,
        population_limit_correction=population_limit_correction,
        classification=classification,
    )

def analyze_timestep_population_interaction(
    fine_points: Sequence[PopulationEnergyPoint],
    coarse_points: Sequence[PopulationEnergyPoint],
    *,
    reporting_resolution: float,
    confidence_level: float = 0.95,
) -> TimeStepPopulationInteraction:
    """Bound the difference between fine- and coarse-step population shifts."""
    fine = _validated_points(fine_points)
    coarse = _validated_points(coarse_points)
    if len(fine) != 2 or len(coarse) != 2:
        raise ValueError("timestep-population interaction requires two points at each timestep")
    _require_doubling(fine[0].walkers, fine[1].walkers)
    _require_doubling(coarse[0].walkers, coarse[1].walkers)
    if (fine[0].walkers, fine[1].walkers) != (
        coarse[0].walkers,
        coarse[1].walkers,
    ):
        raise ValueError("fine and coarse interaction corners must use the same populations")
    fine_bound = population_difference_bound(
        fine[0],
        fine[1],
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
    )
    coarse_bound = population_difference_bound(
        coarse[0],
        coarse[1],
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
    )
    fine_seed_ids = fine[0].seed_ids
    coarse_seed_ids = coarse[0].seed_ids
    if fine_seed_ids == coarse_seed_ids:
        fine_seed_differences = fine[1].seed_energies - fine[0].seed_energies
        coarse_seed_differences = coarse[1].seed_energies - coarse[0].seed_energies
        interaction_seed_differences = coarse_seed_differences - fine_seed_differences
        interaction_statistical_standard_error = float(
            np.std(interaction_seed_differences, ddof=1)
            / math.sqrt(float(interaction_seed_differences.size))
        )
        statistical_error_method = "direct_same_seed_four_corner_sem"
        degrees_of_freedom = interaction_seed_differences.size - 1
    elif set(fine_seed_ids).isdisjoint(coarse_seed_ids):
        interaction_statistical_standard_error = math.hypot(
            fine_bound.paired_standard_error,
            coarse_bound.paired_standard_error,
        )
        statistical_error_method = "quadrature_of_independent_within_timestep_paired_sems"
        degrees_of_freedom = min(
            fine_bound.degrees_of_freedom,
            coarse_bound.degrees_of_freedom,
        )
    else:
        raise ValueError(
            "fine and coarse interaction pairs require identical ordered seed ids "
            "or disjoint seed sets"
        )
    interaction = coarse_bound.mean_difference - fine_bound.mean_difference
    interaction_source_run_quadrature_standard_error = math.sqrt(
        math.fsum(point.conservative_stderr**2 for point in (*fine, *coarse))
    )
    interaction_worst_case_arbitrary_covariance_standard_error_envelope = math.fsum(
        point.conservative_stderr for point in (*fine, *coarse)
    )
    interaction_standard_error = max(
        interaction_statistical_standard_error,
        interaction_source_run_quadrature_standard_error,
    )
    critical_value = _student_critical_value(
        confidence_level,
        degrees_of_freedom=degrees_of_freedom,
    )
    observed = abs(interaction)
    upper_allowance = float(observed + critical_value * interaction_standard_error)
    return TimeStepPopulationInteraction(
        fine_timestep_difference=fine_bound,
        coarse_timestep_difference=coarse_bound,
        interaction_difference=interaction,
        observed_absolute_interaction=observed,
        fine_difference_standard_error=fine_bound.paired_standard_error,
        coarse_difference_standard_error=coarse_bound.paired_standard_error,
        statistical_error_method=statistical_error_method,
        interaction_statistical_standard_error=interaction_statistical_standard_error,
        interaction_source_run_quadrature_standard_error=(
            interaction_source_run_quadrature_standard_error
        ),
        interaction_worst_case_arbitrary_covariance_standard_error_envelope=(
            interaction_worst_case_arbitrary_covariance_standard_error_envelope
        ),
        interaction_standard_error=interaction_standard_error,
        confidence_level=confidence_level,
        degrees_of_freedom=degrees_of_freedom,
        critical_value=critical_value,
        upper_allowance=upper_allowance,
        reporting_resolution=reporting_resolution,
        bounded_below_reporting_resolution=_bounded_at_resolution(
            upper_allowance,
            reporting_resolution,
        ),
    )

def _inverse_population_fit(
    points: tuple[PopulationEnergyPoint, ...],
    *,
    fit_alpha: float,
) -> tuple[_InversePopulationFit, _PopulationLimitCorrection]:
    populations = np.asarray([point.walkers for point in points], dtype=np.float64)
    seed_energies = np.column_stack([point.seed_energies for point in points])
    run_errors = np.asarray([point.conservative_stderr for point in points], dtype=np.float64)
    design = np.column_stack((np.ones_like(populations), 1.0 / populations))
    information = design.T @ design
    if not np.all(np.isfinite(information)) or np.linalg.matrix_rank(design) != 2:
        raise ValueError("inverse-population fit is singular")
    coefficient_map = np.linalg.inv(information) @ design.T
    seed_coefficients = seed_energies @ coefficient_map.T
    coefficient_mean = np.mean(seed_coefficients, axis=0)
    coefficient_sem = np.std(seed_coefficients, axis=0, ddof=1) / math.sqrt(
        float(seed_coefficients.shape[0])
    )
    coefficient_source_floors = np.sqrt(
        np.sum((coefficient_map * run_errors[np.newaxis, :]) ** 2, axis=1)
    )
    coefficient_worst_case_envelopes = np.sum(
        np.abs(coefficient_map) * run_errors[np.newaxis, :],
        axis=1,
    )
    coefficient_stderr = np.maximum(coefficient_sem, coefficient_source_floors)
    correction_map = coefficient_map[0].copy()
    correction_map[1] -= 1.0
    seed_corrections = seed_energies @ correction_map
    correction_mean = float(np.mean(seed_corrections))
    # The matched-seed linear contrast is authoritative.  Reconstructing it
    # from separately averaged, large absolute energies adds cancellation
    # error without testing an independent invariant.
    correction_seed_sem = float(
        np.std(seed_corrections, ddof=1) / math.sqrt(float(seed_corrections.size))
    )
    correction_source_floor = float(np.sqrt(np.sum((correction_map * run_errors) ** 2)))
    correction_worst_case_envelope = float(np.sum(np.abs(correction_map) * run_errors))
    correction_stderr = max(correction_seed_sem, correction_source_floor)
    contrast = seed_energies[:, 0] - 3.0 * seed_energies[:, 1] + 2.0 * seed_energies[:, 2]
    contrast_mean = float(np.mean(contrast))
    contrast_sem = float(np.std(contrast, ddof=1) / math.sqrt(float(contrast.size)))
    contrast_source_floor = math.sqrt(
        math.fsum(
            (
                points[0].conservative_stderr ** 2,
                (3.0 * points[1].conservative_stderr) ** 2,
                (2.0 * points[2].conservative_stderr) ** 2,
            )
        )
    )
    contrast_worst_case_envelope = math.fsum(
        (
            points[0].conservative_stderr,
            3.0 * points[1].conservative_stderr,
            2.0 * points[2].conservative_stderr,
        )
    )
    contrast_stderr = max(contrast_sem, contrast_source_floor)
    if contrast_stderr == 0.0:
        pvalue = 1.0 if contrast_mean == 0.0 else 0.0
    else:
        statistic = abs(contrast_mean) / contrast_stderr
        pvalue = float(2.0 * student_t.sf(statistic, df=contrast.size - 1))
    fit = _InversePopulationFit(
        intercept=float(coefficient_mean[0]),
        slope=float(coefficient_mean[1]),
        intercept_stderr=float(coefficient_stderr[0]),
        slope_stderr=float(coefficient_stderr[1]),
        seed_intercept_standard_error=float(coefficient_sem[0]),
        seed_slope_standard_error=float(coefficient_sem[1]),
        intercept_source_run_quadrature_standard_error=float(coefficient_source_floors[0]),
        slope_source_run_quadrature_standard_error=float(coefficient_source_floors[1]),
        intercept_worst_case_arbitrary_covariance_standard_error_envelope=float(
            coefficient_worst_case_envelopes[0]
        ),
        slope_worst_case_arbitrary_covariance_standard_error_envelope=float(
            coefficient_worst_case_envelopes[1]
        ),
        residual_contrast_mean=contrast_mean,
        residual_contrast_seed_standard_error=contrast_sem,
        residual_contrast_source_run_quadrature_standard_error=contrast_source_floor,
        residual_contrast_worst_case_arbitrary_covariance_standard_error_envelope=(
            contrast_worst_case_envelope
        ),
        residual_contrast_standard_error=contrast_stderr,
        residual_zero_test_pvalue=pvalue,
        residual_zero_test_alpha=fit_alpha,
        residual_zero_test_status=(
            "residual_not_statistically_resolved"
            if pvalue >= fit_alpha
            else "residual_statistically_resolved"
        ),
    )
    correction = _PopulationLimitCorrection(
        reference_walkers=points[1].walkers,
        value=correction_mean,
        matched_seed_standard_error=correction_seed_sem,
        source_run_quadrature_standard_error=correction_source_floor,
        worst_case_arbitrary_covariance_standard_error_envelope=(correction_worst_case_envelope),
        conservative_standard_error=correction_stderr,
    )
    return fit, correction

def _richardson_window_assessment(
    half: PopulationEnergyPoint,
    reference: PopulationEnergyPoint,
    doubled: PopulationEnergyPoint,
    *,
    reporting_resolution: float,
    confidence_level: float,
) -> _RichardsonWindowAssessment:
    low_seed = 2.0 * reference.seed_energies - half.seed_energies
    high_seed = 2.0 * doubled.seed_energies - reference.seed_energies
    differences = high_seed - low_seed
    low_intercept = float(2.0 * reference.energy - half.energy)
    high_intercept = float(2.0 * doubled.energy - reference.energy)
    mean_difference = float(np.mean(differences))
    paired_standard_error = float(np.std(differences, ddof=1) / math.sqrt(float(differences.size)))
    source_run_quadrature_standard_error = math.sqrt(
        math.fsum(
            (
                half.conservative_stderr**2,
                (3.0 * reference.conservative_stderr) ** 2,
                (2.0 * doubled.conservative_stderr) ** 2,
            )
        )
    )
    worst_case_arbitrary_covariance_standard_error_envelope = math.fsum(
        (
            half.conservative_stderr,
            3.0 * reference.conservative_stderr,
            2.0 * doubled.conservative_stderr,
        )
    )
    conservative_standard_error = max(
        paired_standard_error,
        source_run_quadrature_standard_error,
    )
    degrees_of_freedom = differences.size - 1
    critical_value = _student_critical_value(
        confidence_level,
        degrees_of_freedom=degrees_of_freedom,
    )
    observed = abs(mean_difference)
    upper_allowance = float(observed + critical_value * conservative_standard_error)
    return _RichardsonWindowAssessment(
        low_population_intercept=low_intercept,
        high_population_intercept=high_intercept,
        mean_difference=mean_difference,
        observed_absolute_difference=observed,
        paired_standard_error=paired_standard_error,
        source_run_quadrature_standard_error=source_run_quadrature_standard_error,
        worst_case_arbitrary_covariance_standard_error_envelope=(
            worst_case_arbitrary_covariance_standard_error_envelope
        ),
        conservative_standard_error=conservative_standard_error,
        confidence_level=confidence_level,
        degrees_of_freedom=degrees_of_freedom,
        critical_value=critical_value,
        upper_allowance=upper_allowance,
        reporting_resolution=reporting_resolution,
        bounded_below_reporting_resolution=_bounded_at_resolution(
            upper_allowance,
            reporting_resolution,
        ),
    )

def _validated_points(
    points: Sequence[PopulationEnergyPoint],
) -> tuple[PopulationEnergyPoint, ...]:
    values = tuple(sorted(points, key=lambda point: point.walkers))
    if len({point.walkers for point in values}) != len(values):
        raise ValueError("walker populations must be unique within a timestep")
    return values

def _require_doubling(first: int, second: int) -> None:
    if second != 2 * first:
        raise ValueError("population comparisons require exact walker-count doublings")
