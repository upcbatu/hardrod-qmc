from __future__ import annotations

import numpy as np

from hrdmc.estimators.pure.forward_walking.protocols import FloatArray, IntArray
from hrdmc.estimators.pure.forward_walking.results import (
    GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    GENEALOGY_SOURCE_FAMILY_DOMINANCE,
    PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    PLATEAU_INSUFFICIENT_BLOCKS,
    PLATEAU_NO_BLOCKS,
    PLATEAU_RESOLVED,
    PURE_STATUS_ACCEPTED,
    PURE_STATUS_INSUFFICIENT_SAMPLES,
    PURE_STATUS_NO_BLOCKS,
    PURE_STATUS_PLATEAU_UNRESOLVED,
    PURE_STATUS_SCHEMA_INVALID,
    SCHEMA_VALID,
    PureWalkingStatus,
    TransportedLagResult,
)


def overall_status(results: tuple[TransportedLagResult, ...]) -> PureWalkingStatus:
    if any(result.schema_status != SCHEMA_VALID for result in results):
        return PURE_STATUS_SCHEMA_INVALID
    if any(
        result.genealogy_status
        in {
            GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
            GENEALOGY_SOURCE_FAMILY_DOMINANCE,
        }
        for result in results
    ):
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    if all(result.plateau_status == PLATEAU_RESOLVED for result in results):
        return PURE_STATUS_ACCEPTED
    if any(result.plateau_status == PLATEAU_NO_BLOCKS for result in results):
        return PURE_STATUS_NO_BLOCKS
    if any(result.plateau_status == PLATEAU_INSUFFICIENT_BLOCKS for result in results):
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    if any(
        result.plateau_status == PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM for result in results
    ):
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    return PURE_STATUS_PLATEAU_UNRESOLVED


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
