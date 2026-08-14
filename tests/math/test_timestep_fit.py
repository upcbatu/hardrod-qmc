from __future__ import annotations

import math

import pytest

from hrdmc.statistics.timestep_fit import (
    TimeStepPoint,
    absolute_difference_upper_allowance,
    analyze_time_step_extrapolation,
    weighted_leading_time_step_fit,
)


def test_absolute_difference_allowance_keeps_the_confidence_margin_separate() -> None:
    result = absolute_difference_upper_allowance(0.001064, 0.000453, confidence_level=0.95)

    assert result.normal_quantile == pytest.approx(1.9599639845)
    assert result.upper_allowance == pytest.approx(0.00195186308498)
    assert result.upper_allowance > result.absolute_difference_estimate


def test_weighted_linear_fit_recovers_the_zero_step_intercept() -> None:
    points = tuple(
        TimeStepPoint(dt=dt, energy=4.0 + 2.0 * dt, conservative_stderr=0.1)
        for dt in (0.01, 0.02, 0.04)
    )

    fit = weighted_leading_time_step_fit(points, leading_power=1)

    assert fit.intercept == pytest.approx(4.0)
    assert fit.coefficients[1] == pytest.approx(2.0)
    assert fit.chi_squared == pytest.approx(0.0, abs=1.0e-24)


def test_intercept_weights_propagate_shared_point_uncertainty() -> None:
    result = analyze_time_step_extrapolation(
        tuple(TimeStepPoint(dt=dt, energy=2.0, conservative_stderr=0.5) for dt in (1.0, 2.0, 3.0))
    )

    assert result.leading_linear_fit.intercept_weights == pytest.approx(
        (4.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0)
    )
    assert result.leading_quadratic_fit.intercept_weights == pytest.approx(
        (6.0 / 7.0, 3.0 / 7.0, -2.0 / 7.0)
    )
    assert result.leading_model_sensitivity.comparison_uncertainty == pytest.approx(
        math.sqrt(2.0 / 21.0)
    )


def test_largest_point_comparison_uses_correlated_weight_difference() -> None:
    result = analyze_time_step_extrapolation(
        tuple(
            TimeStepPoint(
                dt=dt,
                energy=10.0 if dt == 4.0 else 0.0,
                conservative_stderr=1.0,
            )
            for dt in (1.0, 2.0, 3.0, 4.0)
        )
    )

    stability = result.leading_linear_largest_point_stability
    assert result.leading_linear_fit.intercept_weights == pytest.approx((1.0, 0.5, 0.0, -0.5))
    assert stability.comparison_uncertainty == pytest.approx(math.sqrt(5.0 / 6.0))


def test_bad_alternating_data_are_not_promoted_to_a_fit() -> None:
    result = analyze_time_step_extrapolation(
        tuple(
            TimeStepPoint(
                dt=dt,
                energy=10.0 if index % 2 else 0.0,
                conservative_stderr=0.01,
            )
            for index, dt in enumerate((0.01, 0.02, 0.03, 0.04, 0.05))
        )
    )

    assert result.leading_linear_fit.goodness_of_fit_pvalue < 0.05
