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
    finite = (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(local_energies)
        & np.isfinite(weights)
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
        "finite_sample_count": int(np.count_nonzero(finite)),
        "valid_sample_count": int(np.count_nonzero(finite & valid_mask)),
        "included_sample_count": int(np.count_nonzero(included)),
        "total_sample_count": int(positions.shape[0]),
    }
