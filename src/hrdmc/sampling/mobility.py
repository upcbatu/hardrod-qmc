from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.statistics.streaming import weighted_quantile

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class LocalStepMobility:
    """Whole-configuration mobility diagnostics for one ensemble transition."""
    configuration_esjd: float
    r2_esjd: float
    weighted_free_gap_esjd: float
def local_step_mobility(
    old_positions: FloatArray,
    new_positions: FloatArray,
    *,
    center: float,
    rod_length: float,
) -> LocalStepMobility:
    """Return whole-configuration mobility diagnostics for one local step."""
    old = np.asarray(old_positions, dtype=float)
    new = np.asarray(new_positions, dtype=float)
    if old.shape != new.shape or old.ndim != 2:
        raise ValueError("mobility positions must have matching walker matrices")
    displacement = new - old
    configuration_esjd = float(np.mean(np.sum(displacement * displacement, axis=1)))
    old_r2 = np.mean((old - center) ** 2, axis=1)
    new_r2 = np.mean((new - center) ** 2, axis=1)
    r2_esjd = float(np.mean((new_r2 - old_r2) ** 2))
    old_gap = _weighted_free_gap_coordinate(old, rod_length=rod_length)
    new_gap = _weighted_free_gap_coordinate(new, rod_length=rod_length)
    return LocalStepMobility(
        configuration_esjd=configuration_esjd,
        r2_esjd=r2_esjd,
        weighted_free_gap_esjd=float(np.mean((new_gap - old_gap) ** 2)),
    )
def _weighted_free_gap_coordinate(
    positions: FloatArray,
    *,
    rod_length: float,
) -> FloatArray:
    """Return the trap-weighted mean free gap for every ordered walker."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("positions must have shape (n_walkers, n_particles)")
    n_particles = values.shape[1]
    if n_particles < 2:
        return np.zeros(values.shape[0], dtype=float)
    k = np.arange(1, n_particles, dtype=float)
    coefficients = k * (n_particles - k)
    coefficients /= float(np.sum(coefficients))
    free_gaps = np.diff(values, axis=1) - float(rod_length)
    return free_gaps @ coefficients
def free_gap_batch_diagnostics(
    positions: FloatArray,
    normalized_weights: FloatArray,
    *,
    rod_length: float,
) -> dict[str, float]:
    """Summarize the physical free-gap coordinate at one stored batch."""
    values = np.asarray(positions, dtype=float)
    weights = np.asarray(normalized_weights, dtype=float).reshape(-1)
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("free-gap diagnostics require one weight per walker")
    if values.shape[1] < 2:
        return {
            "weighted_free_gap_mean": 0.0,
            "free_gap_min": float("nan"),
            "free_gap_p01": float("nan"),
        }
    free_gaps = np.diff(values, axis=1) - float(rod_length)
    walker_minimum = np.min(free_gaps, axis=1)
    coordinate = _weighted_free_gap_coordinate(values, rod_length=rod_length)
    return {
        "weighted_free_gap_mean": float(np.sum(weights * coordinate)),
        "free_gap_min": float(np.min(free_gaps)),
        "free_gap_p01": weighted_quantile(walker_minimum, weights, 0.01),
    }
