from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class StreamingBatchObservables(TypedDict):
    samples: FloatArray
    weights: FloatArray
    normalized_weights: FloatArray
    weight_sum: float
    energy_numerator: float
    r2_numerator: float
    mixed_energy: float
    r2_radius: float
    local_energy_variance: float
    local_energy_median: float
    local_energy_mad: float
    local_energy_p001: float
    local_energy_p01: float
    local_energy_p99: float
    local_energy_p999: float
    finite_sample_count: int
    valid_sample_count: int
    included_sample_count: int
    total_sample_count: int


def streaming_batch_observables(
    positions: FloatArray,
    local_energies: FloatArray,
    weights: FloatArray,
    valid_mask: NDArray[np.bool_],
    *,
    center: float,
) -> StreamingBatchObservables:
    """Reduce one weighted DMC batch into mixed-estimator observables."""

    finite = (
        np.all(np.isfinite(positions), axis=1) & np.isfinite(local_energies) & np.isfinite(weights)
    )
    included = finite & valid_mask & (weights > 0.0)
    weight_sum = float(np.sum(weights[included]))
    if weight_sum <= 0.0:
        raise RuntimeError("stored batch has no positive valid weight")
    samples = positions[included]
    energies = local_energies[included]
    raw_weights = weights[included]
    normalized_weights = raw_weights / weight_sum
    per_configuration_r2 = np.mean((samples - center) ** 2, axis=1)
    energy_numerator = float(np.sum(raw_weights * energies))
    r2_numerator = float(np.sum(raw_weights * per_configuration_r2))
    batch_energy = float(np.sum(normalized_weights * energies))
    batch_r2 = float(np.sum(normalized_weights * per_configuration_r2))
    centered_energy = energies - batch_energy
    local_energy_variance = float(np.sum(normalized_weights * centered_energy * centered_energy))
    local_energy_median = _weighted_quantile(energies, normalized_weights, 0.5)
    local_energy_mad = _weighted_quantile(
        np.abs(energies - local_energy_median),
        normalized_weights,
        0.5,
    )
    return {
        "samples": samples,
        "weights": raw_weights,
        "normalized_weights": normalized_weights,
        "weight_sum": weight_sum,
        "energy_numerator": energy_numerator,
        "r2_numerator": r2_numerator,
        "mixed_energy": batch_energy,
        "r2_radius": batch_r2,
        "local_energy_variance": local_energy_variance,
        "local_energy_median": local_energy_median,
        "local_energy_mad": local_energy_mad,
        "local_energy_p001": _weighted_quantile(energies, normalized_weights, 0.001),
        "local_energy_p01": _weighted_quantile(energies, normalized_weights, 0.01),
        "local_energy_p99": _weighted_quantile(energies, normalized_weights, 0.99),
        "local_energy_p999": _weighted_quantile(energies, normalized_weights, 0.999),
        "finite_sample_count": int(np.count_nonzero(finite)),
        "valid_sample_count": int(np.count_nonzero(finite & valid_mask)),
        "included_sample_count": int(np.count_nonzero(included)),
        "total_sample_count": int(positions.shape[0]),
    }


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
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
