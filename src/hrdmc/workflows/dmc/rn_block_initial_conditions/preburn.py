from __future__ import annotations

from typing import Protocol

import numpy as np

from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.api import DMCGuide
from hrdmc.workflows.dmc.rn_block_initial_conditions.geometry import (
    array_min_or_none,
    hard_core_preserving_breathing_scale,
    rms_radius_rows,
)


class SystemBackedGuide(DMCGuide, Protocol):
    @property
    def system(self) -> OpenLineHardRodSystem: ...


def breathing_preburn_walkers(
    walkers: np.ndarray,
    guide: SystemBackedGuide,
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
        rms = rms_radius_rows(positions, center=guide.system.center)
        gaps = np.diff(np.sort(positions, axis=1), axis=1)
        return positions, {
            "breathing_preburn_steps": 0,
            "breathing_preburn_log_step": float(log_step),
            "breathing_preburn_acceptance_rate": None,
            "breathing_preburn_jacobian_dimension": guide.system.n_particles,
            "preburn_rms_mean": float(np.mean(rms)),
            "preburn_rms_std": float(np.std(rms, ddof=1)) if rms.size > 1 else 0.0,
            "preburn_gap_min": array_min_or_none(gaps),
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
    rms = rms_radius_rows(positions, center=guide.system.center)
    gaps = np.diff(np.sort(positions, axis=1), axis=1)
    return positions, {
        "breathing_preburn_steps": int(steps),
        "breathing_preburn_log_step": float(log_step),
        "breathing_preburn_acceptance_rate": float(accepted / attempted)
        if attempted
        else None,
        "breathing_preburn_jacobian_dimension": dimension,
        "preburn_rms_mean": float(np.mean(rms)),
        "preburn_rms_std": float(np.std(rms, ddof=1)) if rms.size > 1 else 0.0,
        "preburn_gap_min": array_min_or_none(gaps),
    }
