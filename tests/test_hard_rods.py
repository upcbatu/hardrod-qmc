from __future__ import annotations

import numpy as np
import pytest

from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.systems.hard_rods import HardRodSystem


def test_invalid_packing_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        HardRodSystem(n_particles=10, length=5.0, rod_length=0.5)


def test_initial_lattice_is_valid() -> None:
    system = HardRodSystem(n_particles=11, length=30.0, rod_length=0.5)
    x = system.initial_lattice(jitter=0.1, seed=1)
    assert system.is_valid(x)


def test_min_image_pair_count() -> None:
    system = HardRodSystem(n_particles=5, length=20.0, rod_length=0.5)
    x = system.initial_lattice()
    d = system.pair_distances_min_image(x)
    assert d.shape == (5 * 4 // 2,)
    assert np.all(d >= 0)


def test_open_line_hard_rods_are_not_periodic() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    x = system.initial_lattice(spacing=1.0, jitter=0.05, seed=1)
    assert system.is_valid(x)
    assert not system.is_valid(np.array([0.0, 0.25, 1.0, 2.0]))
    proposed = system.propose_single_particle(x, 0, displacement=-100.0)
    assert proposed[0] < -50.0


def test_harmonic_trap_values() -> None:
    trap = HarmonicTrap(omega=0.2)
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(trap.values(x), np.array([0.02, 0.0, 0.02]))
    assert trap.total(x) == pytest.approx(0.04)
