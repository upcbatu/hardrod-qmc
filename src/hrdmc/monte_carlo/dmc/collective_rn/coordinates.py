from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.open_line import OpenLineHardRodSystem

FloatArray = NDArray[np.float64]


def hardrod_offsets(system: OpenLineHardRodSystem) -> FloatArray:
    return system.rod_length * (
        np.arange(system.n_particles, dtype=float) - 0.5 * (system.n_particles - 1)
    )


def to_reduced(system: OpenLineHardRodSystem, positions: FloatArray) -> FloatArray:
    return np.asarray(positions, dtype=float) - hardrod_offsets(system)


def from_reduced(system: OpenLineHardRodSystem, reduced_positions: FloatArray) -> FloatArray:
    return np.asarray(reduced_positions, dtype=float) + hardrod_offsets(system)


def scale_reduced_cloud(
    system: OpenLineHardRodSystem,
    positions: FloatArray,
    *,
    log_scale: float,
) -> FloatArray:
    scale = float(np.exp(log_scale))
    reduced = to_reduced(system, positions)
    return from_reduced(system, system.center + scale * (reduced - system.center))


def inverse_scale_reduced_cloud(
    system: OpenLineHardRodSystem,
    positions: FloatArray,
    *,
    log_scale: float,
) -> FloatArray:
    return scale_reduced_cloud(system, positions, log_scale=-log_scale)


def breathing_log_jacobian(system: OpenLineHardRodSystem, log_scale: float) -> float:
    return float(system.n_particles * log_scale)
