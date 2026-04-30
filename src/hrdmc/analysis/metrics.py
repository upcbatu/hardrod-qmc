from __future__ import annotations


def bias(estimate: float, reference: float) -> float:
    return float(estimate - reference)


def mean_squared_error(bias_value: float, variance: float) -> float:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    return float(bias_value**2 + variance)
