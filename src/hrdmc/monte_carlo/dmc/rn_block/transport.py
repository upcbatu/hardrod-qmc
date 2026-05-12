from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RNTransportConvention:
    """Explicit branching contract for transported auxiliary estimators."""

    weight_convention: str = "post_step_normalized_log_weights"
    parent_convention: str = "post_resample_parent_indices"
    gauge_convention: str = "log_weights_pre_resample_are_recentered_max_subtracted"
    snapshot_alignment: str = "on_every_dmc_step"
    parent_map_scope: str = "single_dmc_step"
    collection_mode: str = "single_point"
    future_collection_modes: tuple[str, ...] = ("sliding_window",)
    required_invariants: tuple[str, ...] = (
        "lag0_identity",
        "deterministic_parent_map",
        "weight_gauge_shift_cancellation",
        "composed_parent_map_associativity",
    )


@dataclass(frozen=True)
class RNTransportEvent:
    """One DMC-step transport event emitted by the RN-block engine.

    The event is intentionally algorithmic. Estimator formulas live in
    ``hrdmc.estimators`` and consume this contract. Positions and local
    energies are the post-step population after optional resampling. The
    pre-resampling log weights are kept for audit; post-resampling log weights
    define the normalized estimator weights for the emitted positions.
    """

    step_id: int
    production_step_id: int | None
    block_id: int
    positions: FloatArray
    local_energy_per_walker: FloatArray
    r2_rb_per_walker: FloatArray | None
    log_weights_pre_resample: FloatArray
    log_weights_post_resample: FloatArray
    parent_indices: IntArray
    resampled: bool
    weight_gauge_shift: float
    convention: RNTransportConvention = RNTransportConvention()


class RNTransportObserver(Protocol):
    def record_transport_event(self, event: RNTransportEvent) -> None:
        """Consume a transport event without storing full histories by default."""


def identity_parent_indices(n_walkers: int) -> IntArray:
    return np.arange(n_walkers, dtype=np.int64)


def com_rao_blackwell_r2_per_walker(
    positions: FloatArray,
    *,
    center: float,
    com_variance: float,
) -> FloatArray:
    """Compute per-walker COM Rao-Blackwell R2 for transport payloads."""
    values = np.asarray(positions, dtype=float)
    centered = values - center
    relative = centered - np.mean(centered, axis=1, keepdims=True)
    return np.mean(relative * relative, axis=1) + float(com_variance)
