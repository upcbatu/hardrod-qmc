from __future__ import annotations

import numpy as np
import pytest

from hrdmc.estimators import (
    estimate_weighted_observables,
    filter_weighted_configurations,
    integrate_density_profile,
    weighted_density_profile_on_grid,
    weighted_energy,
    weighted_r2_radius,
    weighted_rms_radius,
)


def test_filter_weighted_configurations_excludes_invalid_zero_and_nonfinite_rows() -> None:
    samples = np.array(
        [
            [-1.0, 1.0],
            [-2.0, 2.0],
            [0.0, 1.0],
            [np.nan, 0.0],
        ]
    )
    local_energies = np.array([1.0, 3.0, 5.0, 7.0])
    weights = np.array([2.0, 1.0, 0.0, 1.0])
    valid = np.array([True, True, False, True])

    filtered = filter_weighted_configurations(samples, local_energies, weights, valid)

    assert filtered.included_sample_count == 2
    assert filtered.zero_weight_excluded_count == 1
    assert filtered.nonfinite_excluded_count == 1
    np.testing.assert_allclose(filtered.normalized_weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    np.testing.assert_allclose(
        weighted_energy(filtered.local_energies, filtered.normalized_weights),
        5.0 / 3.0,
    )


def test_weighted_density_profile_integrates_particle_count() -> None:
    samples = np.array([[-0.5, 0.5], [-0.5, 1.5]])
    weights = np.array([0.25, 0.75])
    grid = np.array([-0.5, 0.5, 1.5])

    density, diagnostics = weighted_density_profile_on_grid(
        samples,
        weights,
        grid,
        n_particles=2,
    )

    np.testing.assert_allclose(integrate_density_profile(density), 2.0)
    np.testing.assert_allclose(diagnostics["density_integral_abs_error"], 0.0)
    np.testing.assert_allclose(density.n_x, np.array([1.0, 0.25, 0.75]))


def test_estimate_weighted_observables_reports_energy_radius_and_density() -> None:
    samples = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    local_energies = np.array([1.0, 3.0])
    weights = np.array([0.25, 0.75])
    valid = np.array([True, True])
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = estimate_weighted_observables(
        samples,
        local_energies,
        weights,
        valid,
        grid,
        center=0.0,
        n_particles=2,
    )

    np.testing.assert_allclose(result.mixed_energy, 2.5)
    np.testing.assert_allclose(result.r2_radius, weighted_r2_radius(samples, weights))
    np.testing.assert_allclose(result.rms_radius, weighted_rms_radius(samples, weights))
    np.testing.assert_allclose(integrate_density_profile(result.density), 2.0)


def test_weighted_estimators_reject_empty_positive_valid_set() -> None:
    with pytest.raises(ValueError, match="no finite positive-weight valid"):
        filter_weighted_configurations(
            np.array([[0.0, 1.0]]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([True]),
        )
