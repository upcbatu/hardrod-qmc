from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationEnergyPoint:
    """One mixed-energy estimate at a fixed walker population."""

    walkers: int
    energy: float
    conservative_stderr: float
    seed_ids: tuple[int, ...]
    seed_energies: FloatArray
    label: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.walkers, bool)
            or not isinstance(self.walkers, (int, np.integer))
            or self.walkers <= 0
        ):
            raise ValueError("walkers must be a positive integer")
        if not math.isfinite(self.energy):
            raise ValueError("population-point energy must be finite")
        if not math.isfinite(self.conservative_stderr) or self.conservative_stderr <= 0.0:
            raise ValueError("population-point conservative stderr must be finite and positive")
        if (
            len(self.seed_ids) < 2
            or len(set(self.seed_ids)) != len(self.seed_ids)
            or any(
                isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                for seed in self.seed_ids
            )
        ):
            raise ValueError("population points require at least two unique seeds")
        values = np.asarray(self.seed_energies, dtype=np.float64)
        if values.shape != (len(self.seed_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("seed_energies must contain one finite value per seed")
        if not math.isclose(
            self.energy,
            float(np.mean(values)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("population-point energy must equal the mean seed energy")
        object.__setattr__(self, "seed_energies", values.copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "walkers": self.walkers,
            "energy": self.energy,
            "conservative_stderr": self.conservative_stderr,
            "seed_ids": list(self.seed_ids),
            "seed_energies": self.seed_energies.tolist(),
            "label": self.label,
        }


@dataclass(frozen=True)
class PopulationDifferenceBound:
    first_walkers: int
    second_walkers: int
    mean_difference: float
    observed_absolute_difference: float
    paired_standard_error: float
    first_run_conservative_stderr: float
    second_run_conservative_stderr: float
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
        return {
            "first_walkers": self.first_walkers,
            "second_walkers": self.second_walkers,
            "mean_difference": self.mean_difference,
            "observed_absolute_difference": self.observed_absolute_difference,
            "paired_standard_error": self.paired_standard_error,
            "first_run_conservative_stderr": self.first_run_conservative_stderr,
            "second_run_conservative_stderr": self.second_run_conservative_stderr,
            "source_run_quadrature_standard_error": (self.source_run_quadrature_standard_error),
            "source_run_quadrature_standard_error_rule": (
                "sqrt(first_run_stderr**2 + second_run_stderr**2)"
            ),
            "worst_case_arbitrary_covariance_standard_error_envelope": (
                self.worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "worst_case_arbitrary_covariance_envelope_rule": (
                "first_run_stderr + second_run_stderr; diagnostic only and not "
                "used as a calibrated confidence standard error"
            ),
            "conservative_standard_error": self.conservative_standard_error,
            "conservative_standard_error_rule": (
                "max(paired seed-difference SEM, source-run quadrature standard error)"
            ),
            "confidence_level": self.confidence_level,
            "degrees_of_freedom": self.degrees_of_freedom,
            "critical_value": self.critical_value,
            "upper_allowance": self.upper_allowance,
            "reporting_resolution": self.reporting_resolution,
            "bounded_below_reporting_resolution": (self.bounded_below_reporting_resolution),
        }


@dataclass(frozen=True)
class InversePopulationFit:
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
        return {
            "model": "E(W) = E_infinity + c / W",
            "intercept": self.intercept,
            "slope": self.slope,
            "intercept_stderr": self.intercept_stderr,
            "slope_stderr": self.slope_stderr,
            "seed_intercept_standard_error": self.seed_intercept_standard_error,
            "seed_slope_standard_error": self.seed_slope_standard_error,
            "intercept_source_run_quadrature_standard_error": (
                self.intercept_source_run_quadrature_standard_error
            ),
            "slope_source_run_quadrature_standard_error": (
                self.slope_source_run_quadrature_standard_error
            ),
            "intercept_worst_case_arbitrary_covariance_standard_error_envelope": (
                self.intercept_worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "slope_worst_case_arbitrary_covariance_standard_error_envelope": (
                self.slope_worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "coefficient_standard_error_rule": (
                "max(matched-seed coefficient SEM, root-sum-square of coefficient-"
                "weighted source-run conservative stderrs)"
            ),
            "residual_contrast": "E(W/2) - 3*E(W) + 2*E(2W)",
            "residual_contrast_mean": self.residual_contrast_mean,
            "residual_contrast_seed_standard_error": (self.residual_contrast_seed_standard_error),
            "residual_contrast_source_run_quadrature_standard_error": (
                self.residual_contrast_source_run_quadrature_standard_error
            ),
            "residual_contrast_worst_case_arbitrary_covariance_standard_error_envelope": (
                self.residual_contrast_worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "residual_contrast_standard_error": (self.residual_contrast_standard_error),
            "residual_zero_test_pvalue": self.residual_zero_test_pvalue,
            "residual_zero_test_alpha": self.residual_zero_test_alpha,
            "residual_zero_test_status": self.residual_zero_test_status,
            "fit_method": (
                "ordinary least squares in 1/W fitted independently for each "
                "matched seed, then aggregated across seeds"
            ),
            "uncertainty_scope": (
                "matched-seed statistical coefficient errors; the separate "
                "Richardson-window upper allowance qualifies model-window sensitivity"
            ),
        }


@dataclass(frozen=True)
class RichardsonWindowAssessment:
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
        return {
            "low_population_intercept": self.low_population_intercept,
            "high_population_intercept": self.high_population_intercept,
            "mean_difference": self.mean_difference,
            "observed_absolute_difference": self.observed_absolute_difference,
            "paired_standard_error": self.paired_standard_error,
            "source_run_quadrature_standard_error": (self.source_run_quadrature_standard_error),
            "source_run_quadrature_standard_error_rule": (
                "sqrt(sigma_W_over_2**2 + 9*sigma_W**2 + 4*sigma_2W**2)"
            ),
            "worst_case_arbitrary_covariance_standard_error_envelope": (
                self.worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "worst_case_arbitrary_covariance_envelope_rule": (
                "sigma_W_over_2 + 3*sigma_W + 2*sigma_2W; diagnostic only"
            ),
            "conservative_standard_error": self.conservative_standard_error,
            "conservative_standard_error_rule": (
                "max(paired Richardson-window SEM, source-run quadrature standard error)"
            ),
            "confidence_level": self.confidence_level,
            "degrees_of_freedom": self.degrees_of_freedom,
            "critical_value": self.critical_value,
            "upper_allowance": self.upper_allowance,
            "reporting_resolution": self.reporting_resolution,
            "bounded_below_reporting_resolution": (self.bounded_below_reporting_resolution),
        }


@dataclass(frozen=True)
class PopulationLadderAssessment:
    points: tuple[PopulationEnergyPoint, ...]
    reference_walkers: int
    last_doubling: PopulationDifferenceBound
    inverse_population_fit: InversePopulationFit | None
    richardson_window: RichardsonWindowAssessment | None
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
            "uncertainty_component_combination": (
                "statistical fit errors, last-doubling upper allowance, and "
                "Richardson-window upper allowance are reported separately"
            ),
            "population_limit_promotion_rule": (
                "the inverse-1/W residual must be statistically unresolved and the "
                "adjacent Richardson-window difference must be bounded below the "
                "reporting resolution"
            ),
        }
        if self.classification == "accepted_population_limit":
            assert self.inverse_population_fit is not None
            assert self.richardson_window is not None
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
            "fine_timestep_difference": self.fine_timestep_difference.to_dict(),
            "coarse_timestep_difference": self.coarse_timestep_difference.to_dict(),
            "interaction_difference": self.interaction_difference,
            "observed_absolute_interaction": self.observed_absolute_interaction,
            "fine_difference_standard_error": self.fine_difference_standard_error,
            "coarse_difference_standard_error": self.coarse_difference_standard_error,
            "statistical_error_method": self.statistical_error_method,
            "interaction_statistical_standard_error": (self.interaction_statistical_standard_error),
            "interaction_source_run_quadrature_standard_error": (
                self.interaction_source_run_quadrature_standard_error
            ),
            "interaction_source_run_quadrature_standard_error_rule": (
                "root-sum-square of the four signed-contrast source-run stderrs"
            ),
            "interaction_worst_case_arbitrary_covariance_standard_error_envelope": (
                self.interaction_worst_case_arbitrary_covariance_standard_error_envelope
            ),
            "interaction_worst_case_arbitrary_covariance_envelope_rule": (
                "sum of the four source-run conservative stderrs; diagnostic only"
            ),
            "interaction_standard_error": self.interaction_standard_error,
            "interaction_standard_error_rule": (
                "max(four-corner statistical SEM, four-run quadrature source floor); "
                "the Student-t multiplier is applied once"
            ),
            "confidence_level": self.confidence_level,
            "degrees_of_freedom": self.degrees_of_freedom,
            "critical_value": self.critical_value,
            "upper_allowance": self.upper_allowance,
            "reporting_resolution": self.reporting_resolution,
            "bounded_below_reporting_resolution": (self.bounded_below_reporting_resolution),
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
    fit = _inverse_population_fit(validated, fit_alpha=fit_alpha)
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
        classification=classification,
    )


def population_difference_bound(
    first: PopulationEnergyPoint,
    second: PopulationEnergyPoint,
    *,
    reporting_resolution: float,
    confidence_level: float = 0.95,
) -> PopulationDifferenceBound:
    """Bound a paired population-energy difference without hiding its error inputs."""

    _validate_reporting_controls(
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
        fit_alpha=0.05,
    )
    if first.seed_ids != second.seed_ids:
        raise ValueError("population differences require identical ordered seed ids")
    differences = second.seed_energies - first.seed_energies
    mean_difference = float(np.mean(differences))
    aggregate_difference = second.energy - first.energy
    if not math.isclose(
        mean_difference,
        aggregate_difference,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("paired seed difference disagrees with aggregate energy difference")
    paired_standard_error = float(np.std(differences, ddof=1) / math.sqrt(float(differences.size)))
    source_run_quadrature_standard_error = math.hypot(
        first.conservative_stderr,
        second.conservative_stderr,
    )
    worst_case_arbitrary_covariance_standard_error_envelope = (
        first.conservative_stderr + second.conservative_stderr
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
    return PopulationDifferenceBound(
        first_walkers=first.walkers,
        second_walkers=second.walkers,
        mean_difference=mean_difference,
        observed_absolute_difference=observed,
        paired_standard_error=paired_standard_error,
        first_run_conservative_stderr=first.conservative_stderr,
        second_run_conservative_stderr=second.conservative_stderr,
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
) -> InversePopulationFit:
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
    return InversePopulationFit(
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


def _richardson_window_assessment(
    half: PopulationEnergyPoint,
    reference: PopulationEnergyPoint,
    doubled: PopulationEnergyPoint,
    *,
    reporting_resolution: float,
    confidence_level: float,
) -> RichardsonWindowAssessment:
    low_seed = 2.0 * reference.seed_energies - half.seed_energies
    high_seed = 2.0 * doubled.seed_energies - reference.seed_energies
    differences = high_seed - low_seed
    low_intercept = float(2.0 * reference.energy - half.energy)
    high_intercept = float(2.0 * doubled.energy - reference.energy)
    mean_difference = float(np.mean(differences))
    if not math.isclose(
        mean_difference,
        high_intercept - low_intercept,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Richardson seed windows disagree with aggregate intercepts")
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
    return RichardsonWindowAssessment(
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


def _validate_reporting_controls(
    *,
    reporting_resolution: float,
    confidence_level: float,
    fit_alpha: float,
) -> None:
    if not math.isfinite(reporting_resolution) or reporting_resolution <= 0.0:
        raise ValueError("reporting_resolution must be finite and positive")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not math.isfinite(fit_alpha) or not 0.0 < fit_alpha < 1.0:
        raise ValueError("fit_alpha must lie strictly between zero and one")


def _student_critical_value(confidence_level: float, *, degrees_of_freedom: int) -> float:
    value = float(
        student_t.ppf(
            1.0 - (1.0 - confidence_level) / 2.0,
            df=degrees_of_freedom,
        )
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Student-t critical value is unavailable")
    return value


def _bounded_at_resolution(upper_allowance: float, reporting_resolution: float) -> bool:
    return upper_allowance <= reporting_resolution or math.isclose(
        upper_allowance,
        reporting_resolution,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )
