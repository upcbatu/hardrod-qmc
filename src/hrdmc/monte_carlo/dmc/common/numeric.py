from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def finite_mean(values: FloatArray | list[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_variance(values: FloatArray | list[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.var(arr, ddof=1))


def finite_min(values: FloatArray | list[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def finite_max(values: FloatArray | list[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def log_weight_span(log_weights: FloatArray) -> float:
    finite = np.asarray(log_weights, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.max(finite) - np.min(finite))


def weighted_quantile(
    values: FloatArray,
    weights: FloatArray,
    quantile: float,
) -> float:
    """Return a finite weighted empirical quantile.

    The definition is the first ordered sample whose cumulative normalized
    weight reaches ``quantile``. It is intentionally deterministic and is used
    only for compact DMC diagnostics at stored production batches.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    data = np.asarray(values, dtype=float).reshape(-1)
    sample_weights = np.asarray(weights, dtype=float).reshape(-1)
    if data.shape != sample_weights.shape:
        raise ValueError("values and weights must have the same shape")
    finite = np.isfinite(data) & np.isfinite(sample_weights) & (sample_weights > 0.0)
    if not np.any(finite):
        return float("nan")
    data = data[finite]
    sample_weights = sample_weights[finite]
    order = np.argsort(data, kind="stable")
    ordered = data[order]
    cumulative = np.cumsum(sample_weights[order])
    target = quantile * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), ordered.size - 1)
    return float(ordered[index])
