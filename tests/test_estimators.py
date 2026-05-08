from __future__ import annotations

import numpy as np

from hrdmc.estimators import (
    estimate_density_profile,
    estimate_local_energy,
    estimate_open_line_density_profile,
    estimate_pair_distribution,
    estimate_static_structure_factor,
)
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import hard_rod_finite_ring_energy_per_particle
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial


def test_pair_distribution_shapes() -> None:
    system = HardRodSystem(n_particles=7, length=21.0, rod_length=0.5)
    snapshots = np.stack([system.initial_lattice(jitter=0.05, seed=i) for i in range(5)])
    result = estimate_pair_distribution(snapshots, system, n_bins=20)
    assert result.r.shape == (20,)
    assert result.g_r.shape == (20,)
    assert result.counts.shape == (20,)


def test_pair_distribution_normalization_sum_rule() -> None:
    system = HardRodSystem(n_particles=5, length=20.0, rod_length=0.5)
    snapshots = np.stack(
        [
            system.initial_lattice(jitter=0.05, seed=11),
            system.initial_lattice(jitter=0.05, seed=12),
            system.initial_lattice(jitter=0.05, seed=13),
        ]
    )
    result = estimate_pair_distribution(snapshots, system, n_bins=25)
    dr = np.diff(result.bin_edges)

    np.testing.assert_allclose(np.sum(result.counts), snapshots.shape[0] * 5 * 4 / 2)
    np.testing.assert_allclose(
        np.sum(result.g_r * dr),
        system.length * (system.n_particles - 1) / (2.0 * system.n_particles),
    )


def test_structure_factor_shapes() -> None:
    system = HardRodSystem(n_particles=7, length=21.0, rod_length=0.5)
    snapshots = np.stack([system.initial_lattice(jitter=0.05, seed=i) for i in range(5)])
    result = estimate_static_structure_factor(snapshots, system, n_modes=8)
    assert result.k.shape == (8,)
    assert result.s_k.shape == (8,)
    assert result.stderr.shape == (8,)


def test_structure_factor_matches_lattice_reference() -> None:
    system = HardRodSystem(n_particles=5, length=10.0, rod_length=0.5)
    snapshots = np.asarray([np.linspace(0.0, system.length, system.n_particles, endpoint=False)])
    result = estimate_static_structure_factor(snapshots, system, n_modes=5)

    np.testing.assert_allclose(result.s_k[:4], np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(result.s_k[4], system.n_particles, atol=1e-12)
    np.testing.assert_allclose(result.stderr, np.zeros(5), atol=1e-12)


def test_all_pair_local_energy_matches_finite_ring_reference() -> None:
    system = HardRodSystem(n_particles=7, length=21.0, rod_length=0.5)
    trial = HardRodJastrowTrial(system=system, power=1.0, nearest_neighbor_only=False)
    snapshots = np.stack([system.initial_lattice(jitter=0.05, seed=i) for i in range(5)])
    result = estimate_local_energy(snapshots, trial)
    exact_total = system.n_particles * hard_rod_finite_ring_energy_per_particle(
        system.n_particles,
        system.length,
        system.rod_length,
    )
    np.testing.assert_allclose(result.values, exact_total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.mean, exact_total, rtol=1e-12, atol=1e-12)


def test_periodic_density_profile_integrates_particle_count() -> None:
    system = HardRodSystem(n_particles=4, length=8.0, rod_length=0.5)
    snapshots = np.array(
        [
            [-0.25, 1.0, 3.0, 7.5],
            [0.25, 2.0, 4.0, 8.25],
        ]
    )
    result = estimate_density_profile(snapshots, system=system, n_bins=16)
    integral = np.sum(result.n_x * np.diff(result.bin_edges))
    np.testing.assert_allclose(integral, system.n_particles)


def test_open_line_density_profile_integrates_particle_count() -> None:
    snapshots = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    result = estimate_open_line_density_profile(snapshots, x_min=-2.0, x_max=2.0, n_bins=40)
    integral = np.sum(result.n_x * np.diff(result.bin_edges))
    np.testing.assert_allclose(integral, 3.0)
