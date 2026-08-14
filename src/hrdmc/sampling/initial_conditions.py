from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hrdmc.system.geometry import OpenLineHardRodSystem, lattice_spacing_for_target_rms
from hrdmc.trial.guide import DMCGuide

INITIALIZATION_MODES = ("tight-lattice", "lda-rms-lattice", "lda-rms-logspread")
@dataclass(frozen=True)
class InitializationControls:
    mode: str = "tight-lattice"
    init_width_log_sigma: float = 0.10
    breathing_preburn_steps: int = 0
    breathing_preburn_log_step: float = 0.04
    def validate(self) -> None:
        if self.mode not in INITIALIZATION_MODES:
            raise ValueError(f"unknown initialization mode: {self.mode}")
        if self.init_width_log_sigma < 0.0:
            raise ValueError("init_width_log_sigma must be non-negative")
        if self.breathing_preburn_steps < 0:
            raise ValueError("breathing_preburn_steps must be non-negative")
        if self.breathing_preburn_log_step < 0.0:
            raise ValueError("breathing_preburn_log_step must be non-negative")

def _rms_radius_rows(positions: np.ndarray, *, center: float) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(positions, dtype=float) - center) ** 2, axis=1))
def _reduced_open_line_coordinates(sorted_positions: np.ndarray, rod_length: float) -> np.ndarray:
    x = np.asarray(sorted_positions, dtype=float)
    if x.ndim != 1:
        raise ValueError("sorted_positions must be one-dimensional")
    offsets = rod_length * (np.arange(x.size, dtype=float) - 0.5 * (x.size - 1))
    return x - offsets
def _physical_from_reduced_open_line(
    reduced_positions: np.ndarray, rod_length: float
) -> np.ndarray:
    u = np.asarray(reduced_positions, dtype=float)
    if u.ndim != 1:
        raise ValueError("reduced_positions must be one-dimensional")
    offsets = rod_length * (np.arange(u.size, dtype=float) - 0.5 * (u.size - 1))
    return u + offsets
def hard_core_preserving_breathing_scale(
    positions: np.ndarray,
    rod_length: float,
    scale: float,
    anchor: float,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    x = np.sort(np.asarray(positions, dtype=float))
    reduced = _reduced_open_line_coordinates(x, rod_length)
    scaled = anchor + scale * (reduced - anchor)
    return _physical_from_reduced_open_line(scaled, rod_length)
def _array_min_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return float(values.min().item())

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
        minimum_free_gap = 0.05
        spacing = max(spacing, system.rod_length + minimum_free_gap)
        target = float(target_initial_rms)
        if initialization_mode == "lda-rms-logspread":
            free_spacing = spacing - system.rod_length
            free_spacings = free_spacing * np.exp(
                rng.normal(0.0, init_width_log_sigma, size=walkers)
            )
            spacings = system.rod_length + np.maximum(free_spacings, minimum_free_gap)
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
    rms = _rms_radius_rows(positions, center=system.center)
    gaps = np.diff(np.sort(positions, axis=1), axis=1)
    return InitialWalkerBatch(
        positions=positions,
        metadata={
            "initialization_mode": initialization_mode,
            "target_initial_rms": target,
            "initial_spacing_mean": float(np.mean(spacings)),
            "initial_spacing_std": float(np.std(spacings, ddof=1)) if walkers > 1 else 0.0,
            "initial_free_spacing_mean": float(np.mean(spacings - system.rod_length)),
            "initial_free_spacing_std": float(np.std(spacings - system.rod_length, ddof=1))
            if walkers > 1
            else 0.0,
            "initial_rms_mean": float(np.mean(rms)),
            "initial_rms_std": float(np.std(rms, ddof=1)) if walkers > 1 else 0.0,
            "initial_gap_min": _array_min_or_none(gaps),
            "init_width_log_sigma": float(init_width_log_sigma),
            "initializer_scope": (
                "initial-condition preconditioner only; production DMC unchanged"
            ),
        },
    )

class _SystemBackedGuide(DMCGuide, Protocol):
    @property
    def system(self) -> OpenLineHardRodSystem: ...
def breathing_preburn_walkers(
    walkers: np.ndarray,
    guide: _SystemBackedGuide,
    rng: np.random.Generator,
    *,
    steps: int,
    log_step: float,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    if steps < 0:
        raise ValueError("breathing_preburn_steps must be non-negative")
    if log_step < 0.0:
        raise ValueError("breathing_preburn_log_step must be non-negative")
    positions = np.asarray(walkers, dtype=float).copy()
    if steps == 0:
        rms = _rms_radius_rows(positions, center=guide.system.center)
        gaps = np.diff(np.sort(positions, axis=1), axis=1)
        return positions, {
            "breathing_preburn_steps": 0,
            "breathing_preburn_log_step": float(log_step),
            "breathing_preburn_acceptance_rate": None,
            "breathing_preburn_jacobian_dimension": guide.system.n_particles,
            "preburn_rms_mean": float(np.mean(rms)),
            "preburn_rms_std": float(np.std(rms, ddof=1)) if rms.size > 1 else 0.0,
            "preburn_gap_min": _array_min_or_none(gaps),
        }
    accepted = 0
    attempted = 0
    dimension = guide.system.n_particles
    for _step in range(steps):
        for index in range(positions.shape[0]):
            old = positions[index]
            scale = float(np.exp(rng.normal(0.0, log_step)))
            proposed = hard_core_preserving_breathing_scale(
                old,
                guide.system.rod_length,
                scale,
                anchor=guide.system.center,
            )
            attempted += 1
            old_log = guide.log_value(old)
            new_log = guide.log_value(proposed)
            if not np.isfinite(old_log) or not np.isfinite(new_log):
                continue
            log_accept = 2.0 * (new_log - old_log) + dimension * np.log(scale)
            if np.log(rng.random()) < min(0.0, float(log_accept)):
                positions[index] = proposed
                accepted += 1
    rms = _rms_radius_rows(positions, center=guide.system.center)
    gaps = np.diff(np.sort(positions, axis=1), axis=1)
    return positions, {
        "breathing_preburn_steps": int(steps),
        "breathing_preburn_log_step": float(log_step),
        "breathing_preburn_acceptance_rate": float(accepted / attempted) if attempted else None,
        "breathing_preburn_jacobian_dimension": dimension,
        "preburn_rms_mean": float(np.mean(rms)),
        "preburn_rms_std": float(np.std(rms, ddof=1)) if rms.size > 1 else 0.0,
        "preburn_gap_min": _array_min_or_none(gaps),
    }

def prepare_initial_walkers(
    system: OpenLineHardRodSystem,
    guide: _SystemBackedGuide,
    walkers: int,
    rng: np.random.Generator,
    *,
    controls: InitializationControls,
    target_initial_rms: float | None = None,
) -> InitialWalkerBatch:
    controls.validate()
    initial = initial_walkers_with_metadata(
        system,
        walkers,
        rng,
        initialization_mode=controls.mode,
        target_initial_rms=target_initial_rms,
        init_width_log_sigma=controls.init_width_log_sigma,
    )
    positions, preburn_metadata = breathing_preburn_walkers(
        initial.positions,
        guide,
        rng,
        steps=controls.breathing_preburn_steps,
        log_step=controls.breathing_preburn_log_step,
    )
    metadata = dict(initial.metadata)
    metadata.update(preburn_metadata)
    return InitialWalkerBatch(positions=positions, metadata=metadata)
