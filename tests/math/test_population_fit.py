from __future__ import annotations

import numpy as np
import pytest

from hrdmc.statistics.population_bound import PopulationEnergyPoint, population_difference_bound
from hrdmc.statistics.population_fit import (
    analyze_population_ladder,
    analyze_timestep_population_interaction,
)


def _point(
    walkers: int,
    energy: float,
    *,
    seeds: tuple[int, ...] = (11, 12, 13, 14, 15),
    spread: float = 2.0e-4,
    stderr: float = 1.0e-4,
) -> PopulationEnergyPoint:
    return PopulationEnergyPoint(
        walkers=walkers,
        energy=energy,
        conservative_stderr=stderr,
        seed_ids=seeds,
        seed_energies=energy + np.linspace(-spread, spread, len(seeds)),
    )


def test_last_population_doubling_uses_paired_seed_uncertainty() -> None:
    result = analyze_population_ladder(
        [_point(256, 5.0), _point(512, 5.0)], reporting_resolution=0.01
    )

    assert result.last_doubling.bounded_below_reporting_resolution is True
    assert result.last_doubling.conservative_standard_error == pytest.approx(np.sqrt(2.0) * 1.0e-4)
    assert result.last_doubling.worst_case_arbitrary_covariance_standard_error_envelope == (
        pytest.approx(2.0e-4)
    )


def test_seed_spread_prevents_a_false_population_equivalence() -> None:
    first = PopulationEnergyPoint(
        walkers=256,
        energy=5.0,
        conservative_stderr=1.0e-4,
        seed_ids=(1, 2, 3, 4, 5),
        seed_energies=np.asarray([4.98, 4.99, 5.0, 5.01, 5.02]),
    )
    second = PopulationEnergyPoint(
        walkers=512,
        energy=5.0,
        conservative_stderr=1.0e-4,
        seed_ids=(1, 2, 3, 4, 5),
        seed_energies=np.asarray([5.02, 5.01, 5.0, 4.99, 4.98]),
    )

    result = analyze_population_ladder([first, second], reporting_resolution=0.01)

    assert result.last_doubling.mean_difference == pytest.approx(0.0)
    assert result.last_doubling.paired_standard_error > 0.01


def test_inverse_population_fit_recovers_a_known_limit() -> None:
    result = analyze_population_ladder(
        [_point(walkers, 5.0 + 10.0 / walkers) for walkers in (128, 256, 512)],
        reporting_resolution=0.01,
    )

    assert result.inverse_population_fit is not None
    assert result.population_limit_correction is not None
    assert result.inverse_population_fit.intercept == pytest.approx(5.0)
    assert result.population_limit_correction.value == pytest.approx(-10.0 / 256.0)


def test_same_seed_timestep_population_interaction_uses_the_four_corner_sem() -> None:
    result = analyze_timestep_population_interaction(
        [_point(256, 5.10), _point(512, 5.05)],
        [_point(256, 5.20), _point(512, 5.15)],
        reporting_resolution=0.01,
    )

    assert result.interaction_difference == pytest.approx(0.0)
    assert result.interaction_source_run_quadrature_standard_error == pytest.approx(2.0e-4)
    assert result.interaction_worst_case_arbitrary_covariance_standard_error_envelope == (
        pytest.approx(4.0e-4)
    )


def test_common_mode_offset_does_not_change_the_paired_contrast() -> None:
    first_values = np.asarray([4535.8615058516125, 4535.87813527751, 4535.846345893767])
    second_values = np.asarray([4535.853517044063, 4535.860332666248, 4535.869568067359])
    first = PopulationEnergyPoint(
        walkers=256,
        energy=float(np.mean(first_values)),
        conservative_stderr=1.0e-3,
        seed_ids=(8204, 8205, 8206),
        seed_energies=first_values,
    )
    second = PopulationEnergyPoint(
        walkers=512,
        energy=float(np.mean(second_values)),
        conservative_stderr=1.0e-3,
        seed_ids=(8204, 8205, 8206),
        seed_energies=second_values,
    )

    result = population_difference_bound(first, second, reporting_resolution=0.01)

    assert result.mean_difference == float(np.mean(second_values - first_values))
    assert result.mean_difference != second.energy - first.energy


def test_population_fit_uses_coefficient_weighted_quadrature() -> None:
    result = analyze_population_ladder(
        [_point(walkers, 5.0 + 10.0 / walkers) for walkers in (128, 256, 512)],
        reporting_resolution=0.01,
    )

    assert result.inverse_population_fit is not None
    fit = result.inverse_population_fit
    assert fit.intercept_source_run_quadrature_standard_error == pytest.approx(
        0.00012247448713915887
    )
    assert fit.slope_source_run_quadrature_standard_error == pytest.approx(0.02370099455417731)
    assert fit.residual_contrast_source_run_quadrature_standard_error == pytest.approx(
        0.00037416573867739413
    )


def test_population_bound_accepts_the_exact_reporting_boundary() -> None:
    first = _point(256, 5.0)
    shift = 0.01 - 2.7764451051977987 * np.sqrt(2.0) * 1.0e-4
    second = _point(512, 5.0 + shift)

    result = population_difference_bound(first, second, reporting_resolution=0.01)

    assert result.upper_allowance == pytest.approx(0.01, abs=2.0e-15)
    assert result.bounded_below_reporting_resolution is True
