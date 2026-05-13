from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_position_batch(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("configuration array must be one- or two-dimensional")
    return arr


def validate_particle_shape(x: FloatArray, n_particles: int) -> None:
    if x.ndim != 2 or x.shape[1] != n_particles:
        raise ValueError("configuration particle count does not match system")


def positions_from_center_and_gaps(
    q_relative: FloatArray,
    gaps: FloatArray,
    *,
    center: float,
) -> FloatArray:
    offsets = np.zeros((gaps.shape[0], gaps.shape[1] + 1), dtype=float)
    offsets[:, 1:] = np.cumsum(gaps, axis=1)
    offsets -= np.mean(offsets, axis=1)[:, np.newaxis]
    return center + q_relative[:, np.newaxis] + offsets
