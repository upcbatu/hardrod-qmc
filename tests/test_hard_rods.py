from __future__ import annotations

import numpy as np
import pytest

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

