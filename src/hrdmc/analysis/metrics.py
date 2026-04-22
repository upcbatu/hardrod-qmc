from __future__ import annotations


def bias(estimate: float, reference: float) -> float:
    return float(estimate - reference)


def mean_squared_error(bias_value: float, variance: float) -> float:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    return float(bias_value**2 + variance)


def cost_score(mse: float, cpu_seconds: float) -> float:
    if mse < 0:
        raise ValueError("mse must be non-negative")
    if cpu_seconds <= 0:
        raise ValueError("cpu_seconds must be positive")
    return float(mse * cpu_seconds)
