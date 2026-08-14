from __future__ import annotations

import numpy as np

from hrdmc.statistics.density import relative_density_l2_error, unit_mass_shell_average


def test_relative_density_l2_error_uses_the_reference_norm() -> None:
    grid = np.linspace(0.0, 1.0, 11)

    assert relative_density_l2_error(grid, np.ones_like(grid), 2.0 * np.ones_like(grid)) == 0.5


def test_shell_average_preserves_a_uniform_density() -> None:
    result = unit_mass_shell_average(np.linspace(0.0, 4.0, 9), np.ones(8), particle_count=4)

    np.testing.assert_allclose(result.boundaries, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.centers, [1.5, 2.5])
    np.testing.assert_allclose(result.values, [1.0, 1.0])


def test_shell_average_transforms_replicates_independently() -> None:
    edges = np.linspace(-2.0, 2.0, 9)
    aggregate = np.ones(8)
    perturbation = np.asarray([0.10, -0.10, 0.10, -0.10, -0.10, 0.10, -0.10, 0.10])
    result = unit_mass_shell_average(
        edges,
        aggregate,
        particle_count=4,
        replicate_densities=np.stack((aggregate + perturbation, aggregate - perturbation)),
    )

    assert result.replicate_values is not None
    assert result.stderr is not None
    np.testing.assert_allclose(result.centers, -result.centers[::-1])
    np.testing.assert_allclose(result.values, result.values[::-1])
    assert np.all(result.stderr >= 0.0)
