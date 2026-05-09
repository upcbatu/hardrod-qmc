from __future__ import annotations

import numpy as np

from hrdmc.estimators import (
    density_support_edges,
    estimate_cloud_moments,
    estimate_density_profile,
    estimate_local_energy,
    estimate_open_line_density_profile,
    estimate_pair_distribution,
    estimate_static_structure_factor,
    estimate_trapped_local_energy,
    integrate_density_profile,
    trapped_hard_rod_local_energy,
)
from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import hard_rod_finite_ring_energy_per_particle
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial
from hrdmc.wavefunctions.trapped import TrappedHardRodTrial


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


def test_trapped_local_kinetic_energy_matches_finite_difference() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    trial = TrappedHardRodTrial(system=system, gaussian_alpha=0.07, contact_power=1.2)
    trap = HarmonicTrap(omega=0.1)
    positions = np.array([-1.4, 0.15, 1.35])
    step = 1e-5
    log0 = trial.log_value(positions)

    laplacian_over_value = 0.0
    for particle in range(system.n_particles):
        plus = positions.copy()
        minus = positions.copy()
        plus[particle] += step
        minus[particle] -= step
        grad_log = (trial.log_value(plus) - trial.log_value(minus)) / (2.0 * step)
        lap_log = (trial.log_value(plus) - 2.0 * log0 + trial.log_value(minus)) / step**2
        laplacian_over_value += lap_log + grad_log**2

    finite_difference_kinetic = -laplacian_over_value
    local_energy = trapped_hard_rod_local_energy(positions, trial, trap)

    np.testing.assert_allclose(local_energy.kinetic, finite_difference_kinetic, rtol=5e-6)


def test_trapped_local_energy_uses_harmonic_trap_convention() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    trial = TrappedHardRodTrial(system=system, gaussian_alpha=0.05)
    trap = HarmonicTrap(omega=0.3, center=0.2)
    positions = np.array([-1.3, -0.2, 1.1])

    local_energy = trapped_hard_rod_local_energy(positions, trial, trap)

    np.testing.assert_allclose(local_energy.trap, trap.total(positions))
    np.testing.assert_allclose(local_energy.total, local_energy.kinetic + trap.total(positions))


def test_trapped_local_energy_rejects_invalid_hard_rod_configuration() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.5)
    trial = TrappedHardRodTrial(system=system, gaussian_alpha=0.05)
    trap = HarmonicTrap(omega=0.1)
    invalid_positions = np.array([-0.1, 0.2, 1.4])

    with np.testing.assert_raises(ValueError):
        trapped_hard_rod_local_energy(invalid_positions, trial, trap)
    with np.testing.assert_raises(ValueError):
        estimate_trapped_local_energy(np.asarray([invalid_positions]), trial, trap)


def test_periodic_density_profile_integrates_particle_count() -> None:
    system = HardRodSystem(n_particles=4, length=8.0, rod_length=0.5)
    snapshots = np.array(
        [
            [-0.25, 1.0, 3.0, 7.5],
            [0.25, 2.0, 4.0, 8.25],
        ]
    )
    result = estimate_density_profile(snapshots, system=system, n_bins=16)
    integral = integrate_density_profile(result)
    np.testing.assert_allclose(integral, system.n_particles)


def test_density_profile_integral_uses_histogram_edges() -> None:
    result = estimate_open_line_density_profile(
        np.array([[0.1, 1.1, 2.1]]),
        x_min=0.0,
        x_max=3.0,
        n_bins=3,
    )

    np.testing.assert_allclose(integrate_density_profile(result), 3.0)
    assert np.trapezoid(result.n_x, result.x) != 3.0


def test_density_support_edges() -> None:
    result = estimate_open_line_density_profile(
        np.array([[-1.0, 0.0, 2.0]]),
        x_min=-2.0,
        x_max=3.0,
        n_bins=5,
    )
    assert density_support_edges(result) == (-0.5, 2.5)


def test_cloud_moments_from_snapshots() -> None:
    snapshots = np.array(
        [
            [-1.0, 1.0],
            [-2.0, 2.0],
        ]
    )
    result = estimate_cloud_moments(snapshots)
    np.testing.assert_allclose(result.mean_square_radius, 2.5)
    np.testing.assert_allclose(result.rms_radius, np.sqrt(2.5))
    assert result.mean_square_radius_stderr > 0.0


def test_open_line_density_profile_integrates_particle_count() -> None:
    snapshots = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    result = estimate_open_line_density_profile(snapshots, x_min=-2.0, x_max=2.0, n_bins=40)
    integral = integrate_density_profile(result)
    np.testing.assert_allclose(integral, 3.0)
