from __future__ import annotations

import numpy as np
import pytest

from hrdmc.theory.tonks_girardeau import (
    trapped_tg_density_profile,
    trapped_tg_energy_total,
    trapped_tg_r2_radius,
    trapped_tg_rms_radius,
)


def test_tonks_girardeau_energy_and_radius_match_occupied_oscillator_levels() -> None:
    assert trapped_tg_energy_total(10, 1.0) == 50.0
    assert trapped_tg_r2_radius(10, 1.0) == 5.0
    assert trapped_tg_rms_radius(10, 1.0) == pytest.approx(np.sqrt(5.0))


def test_one_particle_density_is_the_harmonic_ground_state() -> None:
    grid = np.linspace(-3.0, 3.0, 101)
    density = trapped_tg_density_profile(grid, n_particles=1, omega=1.0)

    np.testing.assert_allclose(density, np.exp(-(grid**2)) / np.sqrt(np.pi))


def test_tonks_girardeau_density_integrates_to_particle_number() -> None:
    grid = np.linspace(-10.0, 10.0, 4001)
    density = trapped_tg_density_profile(grid, n_particles=10, omega=1.0)

    assert np.trapezoid(density, grid) == pytest.approx(10.0, abs=1.0e-11)
    np.testing.assert_allclose(density, density[::-1], rtol=0.0, atol=2.0e-14)
