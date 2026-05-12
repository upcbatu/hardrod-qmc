from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrdmc.systems.open_line import OpenLineHardRodSystem, lattice_spacing_for_target_rms
from hrdmc.workflows.dmc.rn_block_initial_conditions.controls import INITIALIZATION_MODES
from hrdmc.workflows.dmc.rn_block_initial_conditions.geometry import (
    array_min_or_none,
    rms_radius_rows,
)


@dataclass(frozen=True)
class InitialWalkerBatch:
    positions: np.ndarray
    metadata: dict[str, float | int | str | None]


def initial_walkers(
    system: OpenLineHardRodSystem,
    walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if walkers <= 0:
        raise ValueError("walkers must be positive")
    return np.vstack(
        [
            system.initial_lattice(
                spacing=max(1.25, 2.5 * system.rod_length),
                jitter=0.05,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for _ in range(walkers)
        ]
    )


def initial_walkers_with_metadata(
    system: OpenLineHardRodSystem,
    walkers: int,
    rng: np.random.Generator,
    *,
    initialization_mode: str = "tight-lattice",
    target_initial_rms: float | None = None,
    init_width_log_sigma: float = 0.10,
) -> InitialWalkerBatch:
    if initialization_mode not in INITIALIZATION_MODES:
        raise ValueError(f"unknown initialization_mode: {initialization_mode}")
    if walkers <= 0:
        raise ValueError("walkers must be positive")
    if init_width_log_sigma < 0.0:
        raise ValueError("init_width_log_sigma must be non-negative")
    if initialization_mode == "tight-lattice":
        spacings = np.full(walkers, max(1.25, 2.5 * system.rod_length), dtype=float)
        target = None if target_initial_rms is None else float(target_initial_rms)
    else:
        if target_initial_rms is None:
            raise ValueError("target_initial_rms is required for LDA-RMS initialization")
        spacing = lattice_spacing_for_target_rms(system.n_particles, target_initial_rms)
        spacing = max(spacing, system.rod_length * 1.05)
        target = float(target_initial_rms)
        if initialization_mode == "lda-rms-logspread":
            spacings = spacing * np.exp(rng.normal(0.0, init_width_log_sigma, size=walkers))
            spacings = np.maximum(spacings, system.rod_length * 1.05)
        else:
            spacings = np.full(walkers, spacing, dtype=float)
    positions = np.vstack(
        [
            system.initial_lattice_with_spacing(
                spacing=float(spacing),
                jitter=0.05,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for spacing in spacings
        ]
    )
    rms = rms_radius_rows(positions, center=system.center)
    gaps = np.diff(np.sort(positions, axis=1), axis=1)
    return InitialWalkerBatch(
        positions=positions,
        metadata={
            "initialization_mode": initialization_mode,
            "target_initial_rms": target,
            "initial_spacing_mean": float(np.mean(spacings)),
            "initial_spacing_std": float(np.std(spacings, ddof=1)) if walkers > 1 else 0.0,
            "initial_rms_mean": float(np.mean(rms)),
            "initial_rms_std": float(np.std(rms, ddof=1)) if walkers > 1 else 0.0,
            "initial_gap_min": array_min_or_none(gaps),
            "init_width_log_sigma": float(init_width_log_sigma),
            "initializer_scope": (
                "initial-condition preconditioner only; production RN-DMC unchanged"
            ),
        },
    )
