from __future__ import annotations

import numpy as np
import pytest

from hrdmc.sampling.initial_conditions import (
    hard_core_preserving_breathing_scale,
    initial_walkers_with_metadata,
)
from hrdmc.system.geometry import (
    HarmonicTrap,
    OpenLineHardRodSystem,
    lattice_spacing_for_target_rms,
)


def test_lattice_spacing_reproduces_the_requested_rms() -> None:
    n_particles = 8
    target_rms = 25.0
    spacing = lattice_spacing_for_target_rms(n_particles, target_rms)
    offsets = np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1)

    assert np.sqrt(np.mean((spacing * offsets) ** 2)) == pytest.approx(target_rms)


def test_breathing_scale_preserves_the_hard_core_and_scales_free_gaps() -> None:
    positions = np.asarray([-2.0, -0.75, 0.6, 2.1])
    scaled = hard_core_preserving_breathing_scale(positions, rod_length=0.5, scale=1.7, anchor=0.0)

    assert np.all(np.diff(scaled) >= 0.5)
    np.testing.assert_allclose(np.diff(scaled) - 0.5, 1.7 * (np.diff(positions) - 0.5))


def test_lda_rms_initializer_targets_the_requested_width() -> None:
    system = OpenLineHardRodSystem(n_particles=8, rod_length=0.5)
    batch = initial_walkers_with_metadata(
        system,
        walkers=16,
        rng=np.random.default_rng(123),
        initialization_mode="lda-rms-lattice",
        target_initial_rms=25.0,
    )

    initial_rms = batch.metadata["initial_rms_mean"]
    assert isinstance(initial_rms, int | float)
    assert float(initial_rms) == pytest.approx(25.0, abs=0.2)
    assert np.all(np.diff(batch.positions, axis=1) > system.rod_length)


def test_open_line_geometry_has_no_periodic_wraparound() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    positions = system.initial_lattice(spacing=1.0, jitter=0.05, seed=1)

    assert system.is_valid(positions)
    assert not system.is_valid(np.asarray([0.0, 0.25, 1.0, 2.0]))
    assert system.propose_single_particle(positions, 0, displacement=-100.0)[0] < -50.0


def test_harmonic_trap_matches_one_half_omega_squared_x_squared() -> None:
    trap = HarmonicTrap(omega=0.2)
    positions = np.asarray([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(trap.values(positions), [0.02, 0.0, 0.02])
    assert trap.total(positions) == pytest.approx(0.04)
