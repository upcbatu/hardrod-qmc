from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]

LEADING_LINEAR = "leading_linear"
LEADING_QUADRATIC = "leading_quadratic"
CURVATURE = "curvature"
MINIMUM_LEADING_POINT_COUNT = 3
MINIMUM_CURVATURE_POINT_COUNT = 4


@dataclass(frozen=True)
class TimeStepPoint:
    """One energy estimate at a positive DMC time step."""

    dt: float
    energy: float
    conservative_stderr: float
    label: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "dt": self.dt,
            "energy": self.energy,
            "conservative_stderr": self.conservative_stderr,
            "label": self.label,
        }


@dataclass(frozen=True)
class WeightedTimeStepFit:
    """Weighted least-squares fit against an explicit set of time-step powers."""

    model: str
    basis_powers: tuple[int, ...]
    point_count: int
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    formal_covariance: tuple[tuple[float, ...], ...]
    parameter_stderr: tuple[float, ...]
    formal_parameter_stderr: tuple[float, ...]
    intercept_weights: tuple[float, ...]
    covariance_scale: float
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    goodness_of_fit_pvalue: float
    goodness_of_fit_alpha: float
    goodness_of_fit_status: str
    weighted_design_condition_number: float

    @property
    def intercept(self) -> float:
        return self.coefficients[0]

    @property
    def intercept_stderr(self) -> float:
        return self.parameter_stderr[0]

    @property
    def formal_intercept_stderr(self) -> float:
        return self.formal_parameter_stderr[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "point_count": self.point_count,
            "basis_powers": list(self.basis_powers),
            "polynomial_convention": ("E(dt) = sum_j coefficients[j] * dt**basis_powers[j]"),
            "coefficients": list(self.coefficients),
            "covariance": [list(row) for row in self.covariance],
            "formal_covariance": [list(row) for row in self.formal_covariance],
            "parameter_stderr": list(self.parameter_stderr),
            "formal_parameter_stderr": list(self.formal_parameter_stderr),
            "intercept_weights": list(self.intercept_weights),
            "intercept_weight_sum": math.fsum(self.intercept_weights),
            "covariance_scale": self.covariance_scale,
            "covariance_scale_rule": "max(1, chi_squared / degrees_of_freedom)",
            "intercept": self.intercept,
            "intercept_stderr": self.intercept_stderr,
            "formal_intercept_stderr": self.formal_intercept_stderr,
            "chi_squared": self.chi_squared,
            "degrees_of_freedom": self.degrees_of_freedom,
            "reduced_chi_squared": self.reduced_chi_squared,
            "goodness_of_fit_pvalue": self.goodness_of_fit_pvalue,
            "goodness_of_fit_alpha": self.goodness_of_fit_alpha,
            "goodness_of_fit_status": self.goodness_of_fit_status,
            "goodness_of_fit_uses_declared_errors": True,
            "weighted_design_condition_number": self.weighted_design_condition_number,
        }


@dataclass(frozen=True)
class LargestTimeStepStability:
    available: bool
    model: str
    removed_dt: float
    full_intercept: float
    leave_one_out_intercept: float | None
    absolute_shift: float | None
    comparison_uncertainty: float | None
    reduced_fit_goodness_of_fit_pvalue: float | None
    reduced_fit_goodness_of_fit_status: str | None
    sensitivity_sigma: float
    classification: str
    reason: str | None = None

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "available": self.available,
            "model": self.model,
            "removed_dt": self.removed_dt,
            "full_intercept": self.full_intercept,
            "leave_one_out_intercept": self.leave_one_out_intercept,
            "absolute_shift": self.absolute_shift,
            "comparison_uncertainty": self.comparison_uncertainty,
            "comparison_uncertainty_source": (
                "exact diagonal-WLS estimator-difference propagation"
            ),
            "sensitivity_sigma": self.sensitivity_sigma,
            "reduced_fit_goodness_of_fit_pvalue": (self.reduced_fit_goodness_of_fit_pvalue),
            "reduced_fit_goodness_of_fit_status": (self.reduced_fit_goodness_of_fit_status),
            "classification": self.classification,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LeadingModelSensitivity:
    linear_intercept: float
    quadratic_intercept: float
    absolute_spread: float
    relative_spread: float
    comparison_uncertainty: float
    sensitivity_sigma: float
    classification: str
    reason: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "linear_intercept": self.linear_intercept,
            "quadratic_intercept": self.quadratic_intercept,
            "absolute_spread": self.absolute_spread,
            "relative_spread": self.relative_spread,
            "comparison_uncertainty": self.comparison_uncertainty,
            "comparison_uncertainty_source": (
                "exact diagonal-WLS estimator-difference propagation"
            ),
            "sensitivity_sigma": self.sensitivity_sigma,
            "classification": self.classification,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AbsoluteDifferenceUpperAllowance:
    """Normal-approximation upper allowance for an absolute estimator difference."""

    absolute_difference_estimate: float
    comparison_uncertainty: float
    confidence_level: float
    normal_quantile: float
    upper_allowance: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "absolute_difference_estimate": self.absolute_difference_estimate,
            "comparison_uncertainty": self.comparison_uncertainty,
            "confidence_level": self.confidence_level,
            "normal_quantile": self.normal_quantile,
            "upper_allowance": self.upper_allowance,
            "allowance_rule": (
                "absolute_difference_estimate + normal_quantile * comparison_uncertainty"
            ),
            "interpretation": (
                "two-sided normal-approximation upper allowance for the declared "
                "estimator difference"
            ),
        }


@dataclass(frozen=True)
class TimeStepExtrapolation:
    points: tuple[TimeStepPoint, ...]
    leading_linear_fit: WeightedTimeStepFit
    leading_quadratic_fit: WeightedTimeStepFit
    curvature_fit: WeightedTimeStepFit | None
    leading_linear_largest_point_stability: LargestTimeStepStability
    leading_quadratic_largest_point_stability: LargestTimeStepStability
    curvature_largest_point_stability: LargestTimeStepStability | None
    leading_model_sensitivity: LeadingModelSensitivity
    fit_window_status: str
    fit_alpha: float
    classification: str

    def to_dict(self) -> dict[str, Any]:
        model_spread = self.leading_model_sensitivity.absolute_spread
        return {
            "classification": self.classification,
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
            "reference_model": LEADING_LINEAR,
            "reference_model_basis": "E(dt) = E0 + c1 * dt",
            "competing_leading_model": LEADING_QUADRATIC,
            "competing_leading_model_basis": "E(dt) = E0 + c2 * dt**2",
            "curvature_model_role": ("E(dt) = E0 + c1 * dt + c2 * dt**2; range diagnostic only"),
            "fit_alpha": self.fit_alpha,
            "fit_window_status": self.fit_window_status,
            "candidate_zero_step_energy": self.leading_linear_fit.intercept,
            "candidate_zero_step_energy_statistical_stderr": (
                self.leading_linear_fit.intercept_stderr
            ),
            "leading_model_intercept_spread": model_spread,
            "model_spread_interpretation": (
                "observed leading-model intercept difference; it is neither a "
                "statistical standard error nor a complete time-step systematic bound"
            ),
            "leading_linear_fit": self.leading_linear_fit.to_dict(),
            "leading_quadratic_fit": self.leading_quadratic_fit.to_dict(),
            "curvature_fit": (None if self.curvature_fit is None else self.curvature_fit.to_dict()),
            "largest_point_leave_one_out": {
                LEADING_LINEAR: self.leading_linear_largest_point_stability.to_dict(),
                LEADING_QUADRATIC: (self.leading_quadratic_largest_point_stability.to_dict()),
                CURVATURE: (
                    None
                    if self.curvature_largest_point_stability is None
                    else self.curvature_largest_point_stability.to_dict()
                ),
            },
            "leading_linear_vs_leading_quadratic": (self.leading_model_sensitivity.to_dict()),
            "model_selection_uses_reference_energy": False,
            "uncertainty_assumption": (
                "diagonal weighted least squares uses the input conservative "
                "standard errors; cross-time-step covariance is unavailable and omitted"
            ),
        }


def absolute_difference_upper_allowance(
    absolute_difference_estimate: float,
    comparison_uncertainty: float,
    *,
    confidence_level: float,
) -> AbsoluteDifferenceUpperAllowance:
    """Return a conservative two-sided normal allowance for ``|theta_1-theta_2|``."""

    if not math.isfinite(absolute_difference_estimate) or absolute_difference_estimate < 0.0:
        raise ValueError("absolute_difference_estimate must be finite and nonnegative")
    if not math.isfinite(comparison_uncertainty) or comparison_uncertainty < 0.0:
        raise ValueError("comparison_uncertainty must be finite and nonnegative")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    normal_quantile = float(norm.ppf(0.5 + 0.5 * confidence_level))
    upper_allowance = absolute_difference_estimate + normal_quantile * comparison_uncertainty
    return AbsoluteDifferenceUpperAllowance(
        absolute_difference_estimate=float(absolute_difference_estimate),
        comparison_uncertainty=float(comparison_uncertainty),
        confidence_level=float(confidence_level),
        normal_quantile=normal_quantile,
        upper_allowance=float(upper_allowance),
    )


def weighted_leading_time_step_fit(
    points: Sequence[TimeStepPoint],
    *,
    leading_power: int,
    fit_alpha: float = 0.05,
) -> WeightedTimeStepFit:
    """Fit either ``E0 + c1*dt`` or ``E0 + c2*dt**2``."""

    if leading_power == 1:
        model = LEADING_LINEAR
    elif leading_power == 2:
        model = LEADING_QUADRATIC
    else:
        raise ValueError("leading time-step power must be one or two")
    return _weighted_basis_fit(
        points,
        model=model,
        basis_powers=(0, leading_power),
        minimum_point_count=MINIMUM_LEADING_POINT_COUNT,
        fit_alpha=fit_alpha,
    )


def weighted_curvature_time_step_fit(
    points: Sequence[TimeStepPoint],
    *,
    fit_alpha: float = 0.05,
) -> WeightedTimeStepFit:
    """Fit ``E0 + c1*dt + c2*dt**2`` as a sampled-range diagnostic."""

    return _weighted_basis_fit(
        points,
        model=CURVATURE,
        basis_powers=(0, 1, 2),
        minimum_point_count=MINIMUM_CURVATURE_POINT_COUNT,
        fit_alpha=fit_alpha,
    )


def analyze_time_step_extrapolation(
    points: Sequence[TimeStepPoint],
    *,
    sensitivity_sigma: float = 2.0,
    fit_alpha: float = 0.05,
) -> TimeStepExtrapolation:
    """Compare leading linear and quadratic time-step hypotheses.

    The reference linear goodness of fit has precedence. A rejected competing
    leading-quadratic hypothesis makes the leading-order comparison model
    sensitive. The full quadratic polynomial is reported only as a curvature
    diagnostic and does not choose the extrapolated energy.
    """

    validated = _validated_points(points)
    _validate_sensitivity_sigma(sensitivity_sigma)
    _validate_fit_alpha(fit_alpha)
    linear_fit = weighted_leading_time_step_fit(
        validated,
        leading_power=1,
        fit_alpha=fit_alpha,
    )
    quadratic_fit = weighted_leading_time_step_fit(
        validated,
        leading_power=2,
        fit_alpha=fit_alpha,
    )
    curvature_fit = (
        weighted_curvature_time_step_fit(validated, fit_alpha=fit_alpha)
        if len(validated) >= MINIMUM_CURVATURE_POINT_COUNT
        else None
    )
    linear_stability = _largest_point_stability(
        validated,
        full_fit=linear_fit,
        minimum_point_count=MINIMUM_LEADING_POINT_COUNT,
        sensitivity_sigma=sensitivity_sigma,
        fit_alpha=fit_alpha,
    )
    quadratic_stability = _largest_point_stability(
        validated,
        full_fit=quadratic_fit,
        minimum_point_count=MINIMUM_LEADING_POINT_COUNT,
        sensitivity_sigma=sensitivity_sigma,
        fit_alpha=fit_alpha,
    )
    curvature_stability = (
        _largest_point_stability(
            validated,
            full_fit=curvature_fit,
            minimum_point_count=MINIMUM_CURVATURE_POINT_COUNT,
            sensitivity_sigma=sensitivity_sigma,
            fit_alpha=fit_alpha,
        )
        if curvature_fit is not None
        else None
    )
    sensitivity = _leading_model_sensitivity(
        validated,
        linear_fit,
        quadratic_fit,
        sensitivity_sigma=sensitivity_sigma,
    )

    fit_window_unresolved = any(
        not stability.available or stability.reduced_fit_goodness_of_fit_status == "fit_inadequate"
        for stability in (linear_stability, quadratic_stability)
    )
    if linear_fit.goodness_of_fit_status == "fit_inadequate":
        classification = "fit_inadequate"
    elif quadratic_fit.goodness_of_fit_status == "fit_inadequate":
        classification = "model_sensitive"
    elif sensitivity.classification == "model_sensitive" or any(
        stability.classification == "largest_point_sensitive"
        for stability in (linear_stability, quadratic_stability)
    ):
        classification = "model_sensitive"
    elif fit_window_unresolved:
        classification = "fit_window_unresolved"
    else:
        classification = "model_consistent"
    return TimeStepExtrapolation(
        points=tuple(validated),
        leading_linear_fit=linear_fit,
        leading_quadratic_fit=quadratic_fit,
        curvature_fit=curvature_fit,
        leading_linear_largest_point_stability=linear_stability,
        leading_quadratic_largest_point_stability=quadratic_stability,
        curvature_largest_point_stability=curvature_stability,
        leading_model_sensitivity=sensitivity,
        fit_window_status=("unresolved" if fit_window_unresolved else "accepted"),
        fit_alpha=fit_alpha,
        classification=classification,
    )


def _weighted_basis_fit(
    points: Sequence[TimeStepPoint],
    *,
    model: str,
    basis_powers: tuple[int, ...],
    minimum_point_count: int,
    fit_alpha: float,
) -> WeightedTimeStepFit:
    validated = _validated_points(points)
    _validate_fit_alpha(fit_alpha)
    if len(validated) < minimum_point_count:
        raise ValueError(f"{model} time-step fit requires at least {minimum_point_count} points")

    dt = np.asarray([point.dt for point in validated], dtype=float)
    energy = np.asarray([point.energy for point in validated], dtype=float)
    stderr = np.asarray(
        [point.conservative_stderr for point in validated],
        dtype=float,
    )
    dt_scale = float(np.max(dt))
    scaled_dt = dt / dt_scale
    design = np.column_stack([np.power(scaled_dt, power) for power in basis_powers])
    weighted_design = design / stderr[:, None]
    weighted_mean = float(np.average(energy, weights=1.0 / np.square(stderr)))
    weighted_response = (energy - weighted_mean) / stderr
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )
    if int(rank) != len(basis_powers):
        raise ValueError("time-step fit design matrix is rank deficient")
    scaled_coefficients[0] += weighted_mean

    fitted = design @ scaled_coefficients
    residuals = (energy - fitted) / stderr
    chi_squared = float(residuals @ residuals)
    degrees_of_freedom = len(validated) - len(basis_powers)
    reduced_chi_squared = chi_squared / float(degrees_of_freedom)
    goodness_of_fit_pvalue = float(chi2.sf(chi_squared, degrees_of_freedom))

    formal_scaled_covariance = np.linalg.inv(weighted_design.T @ weighted_design)
    coefficient_estimator = formal_scaled_covariance @ (design.T / np.square(stderr)[None, :])
    intercept_weights = coefficient_estimator[0]
    coefficient_transform = np.diag([dt_scale ** (-power) for power in basis_powers])
    coefficients = coefficient_transform @ scaled_coefficients
    formal_covariance = coefficient_transform @ formal_scaled_covariance @ coefficient_transform.T
    covariance_scale = max(1.0, reduced_chi_squared)
    covariance = formal_covariance * covariance_scale
    formal_stderr = np.sqrt(np.diag(formal_covariance))
    parameter_stderr = np.sqrt(np.diag(covariance))
    return WeightedTimeStepFit(
        model=model,
        basis_powers=basis_powers,
        point_count=len(validated),
        coefficients=_vector_tuple(np.asarray(coefficients, dtype=np.float64)),
        covariance=_matrix_tuple(covariance),
        formal_covariance=_matrix_tuple(formal_covariance),
        parameter_stderr=_vector_tuple(parameter_stderr),
        formal_parameter_stderr=_vector_tuple(formal_stderr),
        intercept_weights=_vector_tuple(np.asarray(intercept_weights, dtype=np.float64)),
        covariance_scale=float(covariance_scale),
        chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=float(reduced_chi_squared),
        goodness_of_fit_pvalue=goodness_of_fit_pvalue,
        goodness_of_fit_alpha=fit_alpha,
        goodness_of_fit_status=(
            "accepted" if goodness_of_fit_pvalue >= fit_alpha else "fit_inadequate"
        ),
        weighted_design_condition_number=float(np.linalg.cond(weighted_design)),
    )


def _largest_point_stability(
    points: Sequence[TimeStepPoint],
    *,
    full_fit: WeightedTimeStepFit,
    minimum_point_count: int,
    sensitivity_sigma: float,
    fit_alpha: float,
) -> LargestTimeStepStability:
    ordered = sorted(points, key=lambda point: point.dt)
    remaining = ordered[:-1]
    removed_dt = ordered[-1].dt
    if len(remaining) < minimum_point_count:
        return LargestTimeStepStability(
            available=False,
            model=full_fit.model,
            removed_dt=removed_dt,
            full_intercept=full_fit.intercept,
            leave_one_out_intercept=None,
            absolute_shift=None,
            comparison_uncertainty=None,
            reduced_fit_goodness_of_fit_pvalue=None,
            reduced_fit_goodness_of_fit_status=None,
            sensitivity_sigma=sensitivity_sigma,
            classification="largest_point_check_unavailable",
            reason=(
                f"removing the largest time step leaves fewer than "
                f"{minimum_point_count} points for {full_fit.model}"
            ),
        )
    reduced_fit = _weighted_basis_fit(
        remaining,
        model=full_fit.model,
        basis_powers=full_fit.basis_powers,
        minimum_point_count=minimum_point_count,
        fit_alpha=fit_alpha,
    )
    shift = abs(full_fit.intercept - reduced_fit.intercept)
    extended_reduced_weights = (*reduced_fit.intercept_weights, 0.0)
    comparison_uncertainty = _estimator_difference_uncertainty(
        full_fit.intercept_weights,
        extended_reduced_weights,
        ordered,
    )
    if reduced_fit.goodness_of_fit_status == "fit_inadequate":
        classification = "largest_point_fit_inadequate"
    elif shift <= sensitivity_sigma * comparison_uncertainty:
        classification = "largest_point_stable"
    else:
        classification = "largest_point_sensitive"
    return LargestTimeStepStability(
        available=True,
        model=full_fit.model,
        removed_dt=removed_dt,
        full_intercept=full_fit.intercept,
        leave_one_out_intercept=reduced_fit.intercept,
        absolute_shift=float(shift),
        comparison_uncertainty=float(comparison_uncertainty),
        reduced_fit_goodness_of_fit_pvalue=(reduced_fit.goodness_of_fit_pvalue),
        reduced_fit_goodness_of_fit_status=(reduced_fit.goodness_of_fit_status),
        sensitivity_sigma=sensitivity_sigma,
        classification=classification,
    )


def _leading_model_sensitivity(
    points: Sequence[TimeStepPoint],
    linear_fit: WeightedTimeStepFit,
    quadratic_fit: WeightedTimeStepFit,
    *,
    sensitivity_sigma: float,
) -> LeadingModelSensitivity:
    spread = abs(linear_fit.intercept - quadratic_fit.intercept)
    comparison_uncertainty = _estimator_difference_uncertainty(
        linear_fit.intercept_weights,
        quadratic_fit.intercept_weights,
        points,
    )
    reason: str | None = None
    if linear_fit.goodness_of_fit_status == "fit_inadequate":
        classification = "reference_fit_inadequate"
        reason = "the leading-linear reference model fails its goodness-of-fit check"
    elif quadratic_fit.goodness_of_fit_status == "fit_inadequate":
        classification = "model_sensitive"
        reason = "the competing leading-quadratic model fails its goodness-of-fit check"
    elif spread > sensitivity_sigma * comparison_uncertainty:
        classification = "model_sensitive"
        reason = "leading-model intercept spread exceeds declared-error uncertainty"
    else:
        classification = "model_consistent"
    relative_spread = spread / abs(linear_fit.intercept) if linear_fit.intercept else math.inf
    return LeadingModelSensitivity(
        linear_intercept=linear_fit.intercept,
        quadratic_intercept=quadratic_fit.intercept,
        absolute_spread=float(spread),
        relative_spread=float(relative_spread),
        comparison_uncertainty=float(comparison_uncertainty),
        sensitivity_sigma=sensitivity_sigma,
        classification=classification,
        reason=reason,
    )


def _estimator_difference_uncertainty(
    first_weights: Sequence[float],
    second_weights: Sequence[float],
    points: Sequence[TimeStepPoint],
) -> float:
    if len(first_weights) != len(points) or len(second_weights) != len(points):
        raise ValueError("intercept weights must match the time-step point count")
    variance = math.fsum(
        (first - second) ** 2 * point.conservative_stderr**2
        for first, second, point in zip(
            first_weights,
            second_weights,
            points,
            strict=True,
        )
    )
    return math.sqrt(max(0.0, variance))


def _validated_points(points: Sequence[TimeStepPoint]) -> list[TimeStepPoint]:
    validated = sorted(points, key=lambda point: point.dt)
    if len(validated) < MINIMUM_LEADING_POINT_COUNT:
        raise ValueError("time-step extrapolation requires at least three points")
    for point in validated:
        if not math.isfinite(point.dt) or point.dt <= 0.0:
            raise ValueError("time steps must be finite and positive")
        if not math.isfinite(point.energy):
            raise ValueError("energies must be finite")
        if not math.isfinite(point.conservative_stderr) or point.conservative_stderr <= 0.0:
            raise ValueError("conservative energy standard errors must be finite and positive")
    if len({point.dt for point in validated}) != len(validated):
        raise ValueError("time-step extrapolation requires distinct time steps")
    return validated


def _validate_sensitivity_sigma(sensitivity_sigma: float) -> None:
    if not math.isfinite(sensitivity_sigma) or sensitivity_sigma <= 0.0:
        raise ValueError("sensitivity_sigma must be finite and positive")


def _validate_fit_alpha(fit_alpha: float) -> None:
    if not math.isfinite(fit_alpha) or not 0.0 < fit_alpha < 1.0:
        raise ValueError("fit_alpha must lie strictly between zero and one")


def _vector_tuple(values: FloatArray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _matrix_tuple(values: FloatArray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)
