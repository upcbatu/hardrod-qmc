from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bias(estimate: float, reference: float) -> float:
    return float(estimate - reference)


def mean_squared_error(bias_value: float, variance: float) -> float:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    return float(bias_value**2 + variance)


def density_l2_error(x: FloatArray, estimate: FloatArray, reference: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if estimate.shape != x.shape or reference.shape != x.shape:
        raise ValueError("density arrays must match x shape")
    if x.size < 2:
        raise ValueError("x must contain at least two points")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")
    return float(np.trapezoid((estimate - reference) ** 2, x))


def relative_density_l2_error(x: FloatArray, estimate: FloatArray, reference: FloatArray) -> float:
    """Return ||estimate-reference||_2 / ||reference||_2 on a fixed grid."""
    numerator = density_l2_error(x, estimate, reference)
    denominator = density_l2_error(x, np.zeros_like(np.asarray(reference, dtype=float)), reference)
    if denominator <= 0.0:
        raise ValueError("reference density norm must be positive")
    return float(np.sqrt(numerator / denominator))
