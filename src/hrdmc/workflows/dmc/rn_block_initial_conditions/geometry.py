from __future__ import annotations

import numpy as np


def rms_radius_rows(positions: np.ndarray, *, center: float) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(positions, dtype=float) - center) ** 2, axis=1))


def reduced_open_line_coordinates(sorted_positions: np.ndarray, rod_length: float) -> np.ndarray:
    x = np.asarray(sorted_positions, dtype=float)
    if x.ndim != 1:
        raise ValueError("sorted_positions must be one-dimensional")
    offsets = rod_length * (np.arange(x.size, dtype=float) - 0.5 * (x.size - 1))
    return x - offsets


def physical_from_reduced_open_line(reduced_positions: np.ndarray, rod_length: float) -> np.ndarray:
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
    reduced = reduced_open_line_coordinates(x, rod_length)
    scaled = anchor + scale * (reduced - anchor)
    return physical_from_reduced_open_line(scaled, rod_length)


def array_min_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return float(values.min().item())
