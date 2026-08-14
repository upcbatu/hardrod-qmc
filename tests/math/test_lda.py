from __future__ import annotations

import numpy as np

from hrdmc.theory.lda import (
    excluded_length,
    hard_rod_chemical_potential,
    hard_rod_energy_density,
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
    invert_hard_rod_chemical_potential,
    lda_density_profile,
    lda_mean_square_radius,
    lda_rms_radius,
    lda_support_edges,
    lda_total_energy,
)


def test_hard_rod_equation_of_state_is_positive_in_its_domain() -> None:
    density = 0.3
    rod_length = 0.5

    assert excluded_length(n_particles=21, length=70.0, rod_length=rod_length) > 0.0
    assert hard_rod_finite_ring_energy_per_particle(21, 70.0, rod_length) > 0.0
    assert hard_rod_energy_per_particle(density, rod_length) > 0.0
    assert hard_rod_energy_density(density, rod_length) > 0.0
    assert hard_rod_chemical_potential(density, rod_length) > 0.0


def test_chemical_potential_inverse_round_trip() -> None:
    density = 0.4
    rod_length = 0.3
    chemical_potential = hard_rod_chemical_potential(density, rod_length)

    np.testing.assert_allclose(
        invert_hard_rod_chemical_potential(chemical_potential, rod_length),
        density,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_point_core_chemical_potential_matches_the_fermi_limit() -> None:
    density = 0.7
    chemical_potential = hard_rod_chemical_potential(density, rod_length=0.0)

    np.testing.assert_allclose(chemical_potential, 0.5 * np.pi**2 * density**2)
    np.testing.assert_allclose(invert_hard_rod_chemical_potential(chemical_potential, 0.0), density)


def test_lda_profile_normalizes_and_has_symmetric_support() -> None:
    grid = np.linspace(-16.0, 16.0, 1201)
    potential = 0.5 * 0.1**2 * grid**2
    profile = lda_density_profile(grid, potential, n_particles=5.0, rod_length=0.2)
    left, right = lda_support_edges(profile)

    assert np.isclose(profile.integrated_particles, 5.0, rtol=1.0e-8, atol=1.0e-8)
    assert np.all(profile.n_x >= 0.0)
    assert lda_total_energy(profile, rod_length=0.2) > 0.0
    assert lda_mean_square_radius(profile) > 0.0
    assert lda_rms_radius(profile) > 0.0
    assert left is not None and right is not None
    assert left < 0.0 < right
