from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hrdmc.estimators.pure.forward_walking.assembly import (
    assemble_observable_result_from_stats,
)
from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.contributions import (
    event_contribution_matrix,
    weighted_density_profile,
)
from hrdmc.estimators.pure.forward_walking.lag_state import LagState, SlidingLagState
from hrdmc.estimators.pure.forward_walking.protocols import FloatArray, TransportEventLike
from hrdmc.estimators.pure.forward_walking.results import (
    LagValue,
    PureWalkingResult,
    TransportedLagResult,
)
from hrdmc.estimators.pure.forward_walking.status import (
    is_identity_parent_map,
    normalized_weights,
    overall_status,
)


class TransportedAuxiliaryForwardWalking:
    """Online transported auxiliary-variable FW estimator."""

    def __init__(self, config: PureWalkingConfig):
        config.validate()
        self.config = config
        self._n_walkers: int | None = None
        self._lag_states: dict[str, dict[int, LagState]] = {}
        self._observable_dimensions: dict[str, int] = {}
        self._direct_reference_sum: dict[str, FloatArray] = {}
        self._event_count = 0
        self._production_event_count = 0

    def record_transport_event(self, event: TransportEventLike) -> None:
        self._validate_event_convention(event)
        self._event_count += 1
        if event.production_step_id is None:
            return
        parent_indices = np.asarray(event.parent_indices, dtype=np.int64)
        n_walkers = int(event.positions.shape[0])
        if parent_indices.shape != (n_walkers,):
            raise ValueError("parent_indices must have one entry per walker")
        if np.any(parent_indices < 0) or np.any(parent_indices >= n_walkers):
            raise ValueError("parent_indices out of range")
        parent_is_identity = is_identity_parent_map(parent_indices)
        normalized = normalized_weights(event.log_weights_post_resample)
        if self._n_walkers is None:
            self._n_walkers = n_walkers
            self._observable_dimensions = {
                observable: self._observable_dimension(observable)
                for observable in self.config.observables
            }
            self._lag_states = {
                observable: {
                    lag: self._new_lag_state(
                        lag=lag,
                        n_walkers=n_walkers,
                        dimension=self._observable_dimensions[observable],
                    )
                    for lag in self.config.lag_steps
                }
                for observable in self.config.observables
            }
            self._direct_reference_sum = {
                observable: np.zeros(self._observable_dimensions[observable], dtype=float)
                for observable in self.config.observables
            }
        elif n_walkers != self._n_walkers:
            raise ValueError("walker count changed inside transported FW stream")
        for observable in self.config.observables:
            if observable == "density":
                self._direct_reference_sum[observable] += weighted_density_profile(
                    event.positions,
                    bin_edges=self.config.density_bin_edges,
                    walker_weights=normalized,
                )
                for state in self._lag_states[observable].values():
                    state.step_density(
                        parent_indices=parent_indices,
                        parent_is_identity=parent_is_identity,
                        positions=event.positions,
                        bin_edges=self.config.density_bin_edges,
                        normalized_weights=normalized,
                    )
                continue
            values = event_contribution_matrix(
                event,
                observable=observable,
                center=self.config.center,
                observable_source=self.config.observable_source,
                density_bin_edges=self.config.density_bin_edges,
                pair_bin_edges=self.config.pair_bin_edges,
                structure_k_values=self.config.structure_k_values,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite observable value in transported FW event")
            self._direct_reference_sum[observable] += np.sum(
                normalized[:, np.newaxis] * values,
                axis=0,
            )
            for state in self._lag_states[observable].values():
                state.step(
                    parent_indices=parent_indices,
                    parent_is_identity=parent_is_identity,
                    observable_values=values,
                    normalized_weights=normalized,
                )
        self._production_event_count += 1

    def result(
        self,
        *,
        mixed_r2_reference: float | None = None,
        mixed_rms_radius_reference: float | None = None,
        mixed_observable_references: dict[str, LagValue] | None = None,
    ) -> PureWalkingResult:
        if self._n_walkers is None:
            return PureWalkingResult(
                status="PURE_WALKING_NO_BLOCKS_NO_GO",
                observable_results={},
                metadata=self._metadata(),
            )
        observable_results: dict[str, TransportedLagResult] = {}
        direct_references = self._direct_references()
        if mixed_observable_references is not None:
            direct_references.update(mixed_observable_references)
        if mixed_r2_reference is not None:
            direct_references["r2"] = float(mixed_r2_reference)
        for observable in self.config.observables:
            stats = {
                lag: state.block_stats()
                for lag, state in self._lag_states[observable].items()
            }
            if observable == "r2":
                observable_results[observable] = assemble_observable_result_from_stats(
                    config=self.config,
                    observable=observable,
                    block_mean_by_lag={lag: stat.mean for lag, stat in stats.items()},
                    block_stderr_by_lag={lag: stat.stderr for lag, stat in stats.items()},
                    block_count_by_lag={lag: stat.count for lag, stat in stats.items()},
                    weight_ess_min_by_lag={
                        lag: stat.weight_ess_min for lag, stat in stats.items()
                    },
                    weight_ess_mean_by_lag={
                        lag: stat.weight_ess_mean for lag, stat in stats.items()
                    },
                    variance_by_lag={
                        lag: stat.variance_inflation for lag, stat in stats.items()
                    },
                    mixed_observable_reference=direct_references.get(observable),
                    mixed_r2_reference=mixed_r2_reference,
                    mixed_rms_radius_reference=mixed_rms_radius_reference,
                )
            else:
                observable_results[observable] = assemble_observable_result_from_stats(
                    config=self.config,
                    observable=observable,
                    block_mean_by_lag={lag: stat.mean for lag, stat in stats.items()},
                    block_stderr_by_lag={lag: stat.stderr for lag, stat in stats.items()},
                    block_count_by_lag={lag: stat.count for lag, stat in stats.items()},
                    weight_ess_min_by_lag={
                        lag: stat.weight_ess_min for lag, stat in stats.items()
                    },
                    weight_ess_mean_by_lag={
                        lag: stat.weight_ess_mean for lag, stat in stats.items()
                    },
                    variance_by_lag={
                        lag: stat.variance_inflation for lag, stat in stats.items()
                    },
                    mixed_observable_reference=direct_references.get(observable),
                    mixed_r2_reference=None,
                    mixed_rms_radius_reference=None,
                )
        return PureWalkingResult(
            status=overall_status(tuple(observable_results.values())),
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
            "collection_stride_steps": self.config.collection_stride_steps,
            "observable_source": self.config.observable_source,
            "observables": list(self.config.observables),
            "transport_mode": self.config.transport_mode,
            "collection_mode": self.config.collection_mode,
            "block_storage_mode": "online_mean_variance_no_block_history",
            "plateau_sigma_threshold": self.config.plateau_sigma_threshold,
            "plateau_abs_tolerance": self.config.plateau_abs_tolerance,
            "density_plateau_relative_l2_tolerance": (
                self.config.density_plateau_relative_l2_tolerance
            ),
            "schema_atol": self.config.schema_atol,
            "schema_rtol": self.config.schema_rtol,
            "weight_convention": "post_step_normalized_log_weights",
            "parent_convention": "post_resample_parent_indices",
            "rms_semantics": "paper_rms_radius=sqrt(aggregated_pure_r2)",
            "transport_invariant_tests_passed": list(
                self.config.transport_invariant_tests_passed
            ),
            "event_count": self._event_count,
            "production_event_count": self._production_event_count,
            "schema_reference": "internal_same_event_weighted_mixed_reference",
        }

    def _direct_references(self) -> dict[str, LagValue]:
        if self._production_event_count <= 0:
            return {}
        references: dict[str, LagValue] = {}
        for observable, total in self._direct_reference_sum.items():
            value = total / self._production_event_count
            references[observable] = float(value[0]) if value.shape == (1,) else value.copy()
        return references

    def _validate_event_convention(self, event: TransportEventLike) -> None:
        convention = event.convention
        if convention.weight_convention != "post_step_normalized_log_weights":
            raise ValueError("unsupported event weight convention")
        if convention.parent_convention != "post_resample_parent_indices":
            raise ValueError("unsupported event parent convention")
        if convention.parent_map_scope != "single_dmc_step":
            raise ValueError("unsupported event parent map scope")

    def _new_lag_state(
        self,
        *,
        lag: int,
        n_walkers: int,
        dimension: int,
    ) -> LagState:
        state_cls = (
            SlidingLagState
            if self.config.collection_mode == "sliding_window" and lag > 0
            else LagState
        )
        return state_cls(
            lag_steps=lag,
            block_size_steps=self.config.block_size_steps,
            n_walkers=n_walkers,
            dimension=dimension,
            collection_stride_steps=self.config.collection_stride_steps,
        )

    def _observable_dimension(self, observable: str) -> int:
        if observable == "r2":
            return 1
        if observable == "density":
            edges = self.config.density_bin_edges
            if edges is None:
                raise ValueError("density_bin_edges must be provided for density")
            return int(np.asarray(edges, dtype=float).size - 1)
        if observable == "pair_distance_density":
            edges = self.config.pair_bin_edges
            if edges is None:
                raise ValueError("pair_bin_edges must be provided for pair_distance_density")
            return int(np.asarray(edges, dtype=float).size - 1)
        if observable == "structure_factor":
            k_values = self.config.structure_k_values
            if k_values is None:
                raise ValueError("structure_k_values must be provided for structure_factor")
            return int(np.asarray(k_values, dtype=float).size)
        raise ValueError(f"unsupported observable: {observable}")


def estimate_transported_auxiliary_forward_walking(
    events: Sequence[TransportEventLike],
    config: PureWalkingConfig,
    *,
    mixed_r2_reference: float | None = None,
    mixed_rms_radius_reference: float | None = None,
    mixed_observable_references: dict[str, LagValue] | None = None,
) -> PureWalkingResult:
    estimator = TransportedAuxiliaryForwardWalking(config)
    for event in events:
        estimator.record_transport_event(event)
    return estimator.result(
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_radius_reference,
        mixed_observable_references=mixed_observable_references,
    )
