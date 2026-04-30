from __future__ import annotations

import numpy as np

from hrdmc.estimators import (
    estimate_local_energy,
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


def test_structure_factor_shapes() -> None:
    system = HardRodSystem(n_particles=7, length=21.0, rod_length=0.5)
    snapshots = np.stack([system.initial_lattice(jitter=0.05, seed=i) for i in range(5)])
    result = estimate_static_structure_factor(snapshots, system, n_modes=8)
    assert result.k.shape == (8,)
    assert result.s_k.shape == (8,)
    assert result.stderr.shape == (8,)


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
