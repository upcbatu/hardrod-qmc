from __future__ import annotations

import numpy as np

from hrdmc.estimators import estimate_pair_distribution, estimate_static_structure_factor
from hrdmc.systems.hard_rods import HardRodSystem


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
