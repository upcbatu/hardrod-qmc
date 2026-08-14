from __future__ import annotations

import numpy as np
import pytest

from hrdmc.theory.finite_a_n2 import trapped_n2_finite_a_reference


def _energy_ladder(rod_length: float) -> np.ndarray:
    return np.asarray(
        [
            trapped_n2_finite_a_reference(
                rod_length=rod_length,
                omega=1.0,
                grid_points=points,
                y_max=12.0,
            ).total_energy
            for points in (256, 512, 1024)
        ]
    )


def test_zero_diameter_n2_reference_converges_to_the_tonks_girardeau_limit() -> None:
    energies = _energy_ladder(0.0)
    errors = np.abs(energies - 2.0)

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1.1e-5
    assert errors[1] / errors[2] > 3.9


def test_unit_diameter_n2_reference_has_a_convergent_exact_node_energy() -> None:
    energies = _energy_ladder(1.0)
    errors = np.abs(energies - 3.0)

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1.5e-5
    assert errors[1] / errors[2] > 3.9


def test_finite_a_n2_density_integrates_to_two_particles() -> None:
    reference = trapped_n2_finite_a_reference(
        rod_length=1.0, omega=1.0, grid_points=512, y_max=12.0
    )
    grid = np.linspace(-10.0, 10.0, 4001)

    assert np.trapezoid(reference.density_profile(grid), grid) == pytest.approx(2.0, abs=2.0e-12)
    assert np.sum(reference.relative_probability_mass) == pytest.approx(1.0)
