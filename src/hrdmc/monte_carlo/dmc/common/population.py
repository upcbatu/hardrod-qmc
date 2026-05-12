from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def normalize_weights(weights: FloatArray) -> FloatArray:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return weights / total


def systematic_resample(weights: FloatArray, rng: np.random.Generator) -> IntArray:
    """Systematic resampling indices for DMC population control."""
    w = normalize_weights(weights)
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(w)
    return np.searchsorted(cumulative, positions).astype(np.int64)


def normalize_log_weights(log_weights: FloatArray) -> FloatArray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("all walkers have zero weight")
    shifted = np.full_like(log_weights, -np.inf)
    shifted[finite] = log_weights[finite] - float(np.max(log_weights[finite]))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def effective_sample_size(log_weights: FloatArray) -> float:
    weights = normalize_log_weights(log_weights)
    return float(1.0 / np.sum(weights * weights))


def maybe_resample_population(
    positions: FloatArray,
    local_energies: FloatArray,
    log_weights: FloatArray,
    rng: np.random.Generator,
    *,
    threshold_fraction: float,
) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    if threshold_fraction <= 0.0:
        return positions, local_energies, log_weights, False
    weights = normalize_log_weights(log_weights)
    ess = float(1.0 / np.sum(weights * weights))
    threshold = threshold_fraction * positions.shape[0]
    if ess >= threshold:
        return positions, local_energies, log_weights, False
    indices = systematic_resample(weights, rng)
    return (
        positions[indices].copy(),
        local_energies[indices].copy(),
        np.zeros(positions.shape[0], dtype=float),
        True,
    )


def maybe_resample_population_with_indices(
    positions: FloatArray,
    local_energies: FloatArray,
    log_weights: FloatArray,
    rng: np.random.Generator,
    *,
    threshold_fraction: float,
) -> tuple[FloatArray, FloatArray, FloatArray, bool, IntArray]:
    if threshold_fraction <= 0.0:
        return (
            positions,
            local_energies,
            log_weights,
            False,
            np.arange(positions.shape[0], dtype=np.int64),
        )
    weights = normalize_log_weights(log_weights)
    ess = float(1.0 / np.sum(weights * weights))
    threshold = threshold_fraction * positions.shape[0]
    if ess >= threshold:
        return (
            positions,
            local_energies,
            log_weights,
            False,
            np.arange(positions.shape[0], dtype=np.int64),
        )
    indices = systematic_resample(weights, rng)
    return (
        positions[indices].copy(),
        local_energies[indices].copy(),
        np.zeros(positions.shape[0], dtype=float),
        True,
        indices,
    )


def recenter_log_weights(log_weights: FloatArray) -> FloatArray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("all walkers have zero weight")
    out = log_weights.copy()
    out[finite] -= float(np.max(out[finite]))
    return out


def require_live_weight(log_weights: FloatArray) -> None:
    if not np.any(np.isfinite(log_weights)):
        raise RuntimeError("all walkers have zero weight")
