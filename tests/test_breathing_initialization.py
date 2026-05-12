from __future__ import annotations

import numpy as np

from hrdmc.systems.open_line import OpenLineHardRodSystem, lattice_spacing_for_target_rms
from hrdmc.workflows.dmc.rn_block import RNCase, build_case_objects
from hrdmc.workflows.dmc.rn_block_initial_conditions.geometry import (
    hard_core_preserving_breathing_scale,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions.lattice import (
    initial_walkers_with_metadata,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions.preburn import (
    breathing_preburn_walkers,
)


def test_lattice_spacing_for_target_rms_reproduces_equally_spaced_rms() -> None:
    n_particles = 8
    target_rms = 25.0
    spacing = lattice_spacing_for_target_rms(n_particles, target_rms)
    offsets = np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1)
    positions = spacing * offsets
    rms = np.sqrt(np.mean(positions * positions))
    assert np.isclose(rms, target_rms)


def test_hard_core_preserving_breathing_scale_preserves_domain() -> None:
    positions = np.asarray([-2.0, -0.75, 0.6, 2.1], dtype=float)
    rod_length = 0.5
    scaled = hard_core_preserving_breathing_scale(
        positions,
        rod_length=rod_length,
        scale=1.7,
        anchor=0.0,
    )
    assert np.all(np.diff(scaled) >= rod_length)
    free_old = np.diff(np.sort(positions)) - rod_length
    free_new = np.diff(scaled) - rod_length
    assert np.allclose(free_new, 1.7 * free_old)


def test_breathing_scale_identity_returns_sorted_positions() -> None:
    positions = np.asarray([2.1, -2.0, 0.6, -0.75], dtype=float)
    scaled = hard_core_preserving_breathing_scale(
        positions,
        rod_length=0.5,
        scale=1.0,
        anchor=0.0,
    )
    assert np.allclose(scaled, np.sort(positions))


def test_breathing_scale_below_one_keeps_free_gaps_nonnegative() -> None:
    positions = np.asarray([-2.0, -0.75, 0.6, 2.1], dtype=float)
    scaled = hard_core_preserving_breathing_scale(
        positions,
        rod_length=0.5,
        scale=0.25,
        anchor=0.0,
    )
    assert np.all(np.diff(scaled) >= 0.5)


def test_lda_rms_initializer_targets_requested_rms() -> None:
    system = OpenLineHardRodSystem(n_particles=8, rod_length=0.5)
    batch = initial_walkers_with_metadata(
        system,
        walkers=16,
        rng=np.random.default_rng(123),
        initialization_mode="lda-rms-lattice",
        target_initial_rms=25.0,
    )
    assert abs(float(batch.metadata["initial_rms_mean"]) - 25.0) < 0.2
    assert batch.metadata["initialization_mode"] == "lda-rms-lattice"


def test_breathing_preburn_metadata_is_reported_when_enabled() -> None:
    case = RNCase(n_particles=4, rod_length=0.5, omega=0.1)
    system, _trap, guide, _target, _proposal = build_case_objects(case)
    batch = initial_walkers_with_metadata(
        system,
        walkers=4,
        rng=np.random.default_rng(123),
        initialization_mode="tight-lattice",
    )
    _walkers, metadata = breathing_preburn_walkers(
        batch.positions,
        guide,
        np.random.default_rng(456),
        steps=2,
        log_step=0.02,
    )
    assert metadata["breathing_preburn_steps"] == 2
    assert "breathing_preburn_acceptance_rate" in metadata
    assert metadata["breathing_preburn_jacobian_dimension"] == case.n_particles
    assert metadata["preburn_gap_min"] >= system.rod_length
