from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.estimators.pure.forward_walking.assembly import assemble_observable_result
from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.contributions import event_contribution_matrix
from hrdmc.estimators.pure.forward_walking.results import (
    PureWalkingResult,
    PureWalkingStatus,
    TransportedLagResult,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TransportConventionLike(Protocol):
    @property
    def weight_convention(self) -> str: ...

    @property
    def parent_convention(self) -> str: ...

    @property
    def parent_map_scope(self) -> str: ...


class TransportEventLike(Protocol):
    @property
    def production_step_id(self) -> int | None: ...

    @property
    def positions(self) -> FloatArray: ...

    @property
    def r2_rb_per_walker(self) -> FloatArray | None: ...

    @property
    def log_weights_post_resample(self) -> FloatArray: ...

    @property
    def parent_indices(self) -> IntArray: ...

    @property
    def convention(self) -> TransportConventionLike: ...


class TransportedAuxiliaryForwardWalking:
    """Online transported auxiliary-variable FW estimator."""

    def __init__(self, config: PureWalkingConfig):
        config.validate()
        self.config = config
        self._n_walkers: int | None = None
        self._lag_states: dict[str, dict[int, _LagState]] = {}
        self._event_count = 0
        self._production_event_count = 0

    def record_transport_event(self, event: TransportEventLike) -> None:
        self._validate_event_convention(event)
        self._event_count += 1
        if event.production_step_id is None:
            return
        values_by_observable = {
            observable: event_contribution_matrix(
                event,
                observable=observable,
                center=self.config.center,
                observable_source=self.config.observable_source,
                density_bin_edges=self.config.density_bin_edges,
                pair_bin_edges=self.config.pair_bin_edges,
                structure_k_values=self.config.structure_k_values,
            )
            for observable in self.config.observables
        }
        if any(not np.all(np.isfinite(values)) for values in values_by_observable.values()):
            raise ValueError("non-finite observable value in transported FW event")
        parent_indices = np.asarray(event.parent_indices, dtype=np.int64)
        n_walkers = int(event.positions.shape[0])
        if parent_indices.shape != (n_walkers,):
            raise ValueError("parent_indices must have one entry per walker")
        if np.any(parent_indices < 0) or np.any(parent_indices >= n_walkers):
            raise ValueError("parent_indices out of range")
        normalized_weights = _normalized_weights(event.log_weights_post_resample)
        if self._n_walkers is None:
            self._n_walkers = n_walkers
            self._lag_states = {
                observable: {
                    lag: _LagState(
                        lag_steps=lag,
                        block_size_steps=self.config.block_size_steps,
                        n_walkers=n_walkers,
                        dimension=values_by_observable[observable].shape[1],
                    )
                    for lag in self.config.lag_steps
                }
                for observable in self.config.observables
            }
        elif n_walkers != self._n_walkers:
            raise ValueError("walker count changed inside transported FW stream")
        for observable, values in values_by_observable.items():
            for state in self._lag_states[observable].values():
                state.step(
                    parent_indices=parent_indices,
                    observable_values=values,
                    normalized_weights=normalized_weights,
                )
        self._production_event_count += 1

    def result(
        self,
        *,
        mixed_r2_reference: float | None = None,
        mixed_rms_radius_reference: float | None = None,
    ) -> PureWalkingResult:
        if self._n_walkers is None:
            return PureWalkingResult(
                status="PURE_WALKING_NO_BLOCKS_NO_GO",
                observable_results={},
                metadata=self._metadata(),
            )
        observable_results: dict[str, TransportedLagResult] = {}
        for observable in self.config.observables:
            block_values = {
                lag: state.block_values.copy()
                for lag, state in self._lag_states[observable].items()
            }
            block_weight_ess = {
                lag: state.block_weight_ess.copy()
                for lag, state in self._lag_states[observable].items()
            }
            if observable == "r2":
                observable_results[observable] = assemble_observable_result(
                    config=self.config,
                    observable=observable,
                    block_values_by_lag=block_values,
                    block_weight_ess_by_lag=block_weight_ess,
                    mixed_r2_reference=mixed_r2_reference,
                    mixed_rms_radius_reference=mixed_rms_radius_reference,
                )
            else:
                observable_results[observable] = assemble_observable_result(
                    config=self.config,
                    observable=observable,
                    block_values_by_lag=block_values,
                    block_weight_ess_by_lag=block_weight_ess,
                    mixed_r2_reference=None,
                    mixed_rms_radius_reference=None,
                )
        return PureWalkingResult(
            status=_overall_status(tuple(observable_results.values())),
            observable_results=observable_results,
            metadata=self._metadata(),
        )

    def _metadata(self) -> dict[str, object]:
        return {
            "estimator": "transported_auxiliary_forward_walking",
            "pure_estimator_tier": "transported_auxiliary_fw",
            "raw_descendant_count_role": "diagnostic_only",
            "lag_steps": list(self.config.lag_steps),
            "lag_unit": self.config.lag_unit,
            "block_size_steps": self.config.block_size_steps,
            "observable_source": self.config.observable_source,
            "observables": list(self.config.observables),
            "transport_mode": self.config.transport_mode,
            "collection_mode": self.config.collection_mode,
            "weight_convention": "post_step_normalized_log_weights",
            "parent_convention": "post_resample_parent_indices",
            "rms_semantics": "paper_rms_radius=sqrt(aggregated_pure_r2)",
            "transport_invariant_tests_passed": list(
                self.config.transport_invariant_tests_passed
            ),
            "event_count": self._event_count,
            "production_event_count": self._production_event_count,
        }

    def _validate_event_convention(self, event: TransportEventLike) -> None:
        convention = event.convention
        if convention.weight_convention != "post_step_normalized_log_weights":
            raise ValueError("unsupported event weight convention")
        if convention.parent_convention != "post_resample_parent_indices":
            raise ValueError("unsupported event parent convention")
        if convention.parent_map_scope != "single_dmc_step":
            raise ValueError("unsupported event parent map scope")


def estimate_transported_auxiliary_forward_walking(
    events: Sequence[TransportEventLike],
    config: PureWalkingConfig,
    *,
    mixed_r2_reference: float | None = None,
    mixed_rms_radius_reference: float | None = None,
) -> PureWalkingResult:
    estimator = TransportedAuxiliaryForwardWalking(config)
    for event in events:
        estimator.record_transport_event(event)
    return estimator.result(
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_radius_reference,
    )


@dataclass
class _LagState:
    lag_steps: int
    block_size_steps: int
    n_walkers: int
    dimension: int
    auxiliary: FloatArray = field(init=False)
    weighted_collect_sum: FloatArray = field(init=False)
    collect_count: int = 0
    delay_count: int = 0
    block_values: list[FloatArray] = field(default_factory=list)
    block_weight_ess: list[float] = field(default_factory=list)
    min_weight_ess: float = float("inf")

    def __post_init__(self) -> None:
        self.auxiliary = np.zeros((self.n_walkers, self.dimension), dtype=float)
        self.weighted_collect_sum = np.zeros(self.dimension, dtype=float)

    def step(
        self,
        *,
        parent_indices: IntArray,
        observable_values: FloatArray,
        normalized_weights: FloatArray,
    ) -> None:
        weight_ess = float(1.0 / np.sum(normalized_weights * normalized_weights))
        self.min_weight_ess = min(self.min_weight_ess, weight_ess)
        self.auxiliary = self.auxiliary[parent_indices]
        if self.collect_count < self.block_size_steps:
            self.auxiliary += observable_values
            self.weighted_collect_sum += np.sum(
                normalized_weights[:, np.newaxis] * observable_values,
                axis=0,
            )
            self.collect_count += 1
        elif self.delay_count < self.lag_steps:
            self.delay_count += 1
        if self.collect_count == self.block_size_steps and self.delay_count == self.lag_steps:
            if self.lag_steps == 0:
                self.block_values.append(self.weighted_collect_sum / self.block_size_steps)
            else:
                self.block_values.append(
                    np.sum(normalized_weights[:, np.newaxis] * self.auxiliary, axis=0)
                    / self.block_size_steps
                )
            self.block_weight_ess.append(self.min_weight_ess)
            self.auxiliary = np.zeros((self.n_walkers, self.dimension), dtype=float)
            self.weighted_collect_sum = np.zeros(self.dimension, dtype=float)
            self.collect_count = 0
            self.delay_count = 0
            self.min_weight_ess = float("inf")


def _overall_status(results: tuple[TransportedLagResult, ...]) -> PureWalkingStatus:
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


def _normalized_weights(log_weights: FloatArray) -> FloatArray:
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
