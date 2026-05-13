from __future__ import annotations

import numpy as np

from hrdmc.estimators.pure.forward_walking.protocols import FloatArray, IntArray
from hrdmc.estimators.pure.forward_walking.results import (
    PureWalkingStatus,
    TransportedLagResult,
)


def overall_status(results: tuple[TransportedLagResult, ...]) -> PureWalkingStatus:
    if any(result.schema_status != "SCHEMA_GO" for result in results):
        return "PURE_WALKING_SCHEMA_NO_GO"
    if all(result.plateau_status == "PLATEAU_FOUND" for result in results):
        return "PURE_WALKING_GO"
    if any(result.plateau_status == "NO_BLOCKS" for result in results):
        return "PURE_WALKING_NO_BLOCKS_NO_GO"
    if any(result.plateau_status == "INSUFFICIENT_BLOCKS" for result in results):
        return "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO"
    if any(result.plateau_status == "INSUFFICIENT_WEIGHT_ESS" for result in results):
        return "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO"
    return "PURE_WALKING_PLATEAU_NO_GO"


def normalized_weights(log_weights: FloatArray) -> FloatArray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("transport event has no finite post-step log weights")
    shifted = np.full_like(log_weights, -np.inf, dtype=float)
    shifted[finite] = log_weights[finite] - float(np.max(log_weights[finite]))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("transport event has invalid post-step weights")
    return weights / total


def is_identity_parent_map(parent_indices: IntArray) -> bool:
    return bool(np.array_equal(parent_indices, np.arange(parent_indices.size)))
