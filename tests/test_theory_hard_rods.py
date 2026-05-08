from __future__ import annotations

import numpy as np
import pytest

from hrdmc.theory import (
    excluded_length,
    hard_rod_chemical_potential,
    hard_rod_energy_density,
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
    invert_hard_rod_chemical_potential,
    lda_density_profile,
    lda_total_energy,
)


def test_hard_rod_eos_positive() -> None:
    rho = 0.3
    rod_length = 0.5
    assert excluded_length(n_particles=21, length=70.0, rod_length=rod_length) > 0
    assert hard_rod_finite_ring_energy_per_particle(21, 70.0, rod_length) > 0
    assert hard_rod_energy_per_particle(rho, rod_length) > 0
    assert hard_rod_energy_density(rho, rod_length) > 0
    assert hard_rod_chemical_potential(rho, rod_length) > 0


def test_chemical_potential_inverse_roundtrip() -> None:
    rho = 0.4
    rod_length = 0.3
    mu = hard_rod_chemical_potential(rho, rod_length)
    recovered = invert_hard_rod_chemical_potential(mu, rod_length)
    np.testing.assert_allclose(recovered, rho, rtol=1e-10, atol=1e-10)


def test_chemical_potential_is_monotone() -> None:
    rod_length = 0.2
    densities = np.linspace(0.05, 0.8 / rod_length, 25)
    mu = np.asarray([hard_rod_chemical_potential(float(rho), rod_length) for rho in densities])
    assert np.all(np.diff(mu) > 0)


def test_point_core_chemical_potential_limit() -> None:
    rho = 0.7
    mu = hard_rod_chemical_potential(rho, rod_length=0.0)
    np.testing.assert_allclose(mu, np.pi**2 * rho**2)
    np.testing.assert_allclose(invert_hard_rod_chemical_potential(mu, 0.0), rho)


def test_lda_profile_normalizes_particles() -> None:
    x = np.linspace(-16.0, 16.0, 1201)
    potential = 0.5 * 0.1**2 * x**2
    profile = lda_density_profile(x, potential, n_particles=5.0, rod_length=0.2)
    assert np.isclose(profile.integrated_particles, 5.0, rtol=1e-8, atol=1e-8)
    assert profile.chemical_potential > np.min(potential)
    assert np.all(profile.n_x >= 0.0)
    assert profile.n_x[np.argmin(np.abs(x))] > profile.n_x[0]
    assert lda_total_energy(profile, rod_length=0.2) > 0.0


def test_lda_profile_rejects_truncated_grid() -> None:
    x = np.linspace(-4.0, 4.0, 401)
    potential = 0.5 * 0.1**2 * x**2

    with pytest.raises(ValueError, match="does not contain the density cloud"):
        lda_density_profile(x, potential, n_particles=5.0, rod_length=0.2)
