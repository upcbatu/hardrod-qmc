from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hrdmc.estimators.pure.forward_walking.contributions import (
    add_density_profile_to_auxiliary,
    density_profile_matrix,
    weighted_density_profile,
)
from hrdmc.estimators.pure.forward_walking.protocols import FloatArray, IntArray


@dataclass
class LagBlockStats:
    count: int
    mean: FloatArray
    stderr: FloatArray
    weight_ess_min: float
    weight_ess_mean: float
    variance_inflation: float
    source_ancestor_ess_min: float
    source_ancestor_ess_mean: float
    unique_source_ancestor_min: int
    max_source_family_fraction: float


@dataclass
class LagState:
    lag_steps: int
    block_size_steps: int
    n_walkers: int
    dimension: int
    collection_stride_steps: int = 1
    auxiliary: FloatArray = field(init=False)
    weighted_collect_sum: FloatArray = field(init=False)
    collect_count: int = 0
    delay_count: int = 0
    min_weight_ess: float = float("inf")
    block_count: int = 0
    block_mean: FloatArray = field(init=False)
    block_m2: FloatArray = field(init=False)
    block_scalar_mean: float = 0.0
    block_scalar_m2: float = 0.0
    block_weight_ess_min: float = float("inf")
    block_weight_ess_sum: float = 0.0
    source_ancestors: IntArray = field(init=False)
    block_source_ancestor_ess_min: float = float("inf")
    block_source_ancestor_ess_sum: float = 0.0
    block_unique_source_ancestor_min: int = field(init=False)
    block_max_source_family_fraction: float = 0.0

    def __post_init__(self) -> None:
        self.auxiliary = np.zeros((self.n_walkers, self.dimension), dtype=float)
        self.weighted_collect_sum = np.zeros(self.dimension, dtype=float)
        self.block_mean = np.zeros(self.dimension, dtype=float)
        self.block_m2 = np.zeros(self.dimension, dtype=float)
        self.source_ancestors = np.arange(self.n_walkers, dtype=np.int64)
        self.block_unique_source_ancestor_min = self.n_walkers

    def step(
        self,
        *,
        parent_indices: IntArray,
        parent_is_identity: bool,
        observable_values: FloatArray,
        normalized_weights: FloatArray,
    ) -> None:
        weight_ess = float(1.0 / np.sum(normalized_weights * normalized_weights))
        self.min_weight_ess = min(self.min_weight_ess, weight_ess)
        if self.lag_steps == 0:
            self._collect_weighted_values(observable_values, normalized_weights)
            if self.collect_count == self.block_size_steps:
                self._record_block(
                    self.weighted_collect_sum / self.block_size_steps,
                    normalized_weights=normalized_weights,
                    source_ancestors=np.arange(self.n_walkers, dtype=np.int64),
                )
                self._reset_window()
            return

        if self.collect_count == 0 and self.delay_count == 0:
            self.source_ancestors = np.arange(self.n_walkers, dtype=np.int64)
        elif not parent_is_identity:
            self.auxiliary = self.auxiliary[parent_indices]
            self.source_ancestors = self.source_ancestors[parent_indices]
        if self.collect_count < self.block_size_steps:
            self.auxiliary += observable_values
            self._collect_weighted_values(observable_values, normalized_weights)
        elif self.delay_count < self.lag_steps:
            self.delay_count += 1
        if self.collect_count == self.block_size_steps and self.delay_count == self.lag_steps:
            block_value = (
                np.sum(normalized_weights[:, np.newaxis] * self.auxiliary, axis=0)
                / self.block_size_steps
            )
            self._record_block(
                block_value,
                normalized_weights=normalized_weights,
                source_ancestors=self.source_ancestors,
            )
            self._reset_window()

    def step_density(
        self,
        *,
        parent_indices: IntArray,
        parent_is_identity: bool,
        positions: FloatArray,
        bin_edges: FloatArray | None,
        normalized_weights: FloatArray,
    ) -> None:
        weight_ess = float(1.0 / np.sum(normalized_weights * normalized_weights))
        self.min_weight_ess = min(self.min_weight_ess, weight_ess)
        if self.lag_steps == 0:
            self.weighted_collect_sum += weighted_density_profile(
                positions,
                bin_edges=bin_edges,
                walker_weights=normalized_weights,
            )
            self.collect_count += 1
            if self.collect_count == self.block_size_steps:
                self._record_block(
                    self.weighted_collect_sum / self.block_size_steps,
                    normalized_weights=normalized_weights,
                    source_ancestors=np.arange(self.n_walkers, dtype=np.int64),
                )
                self._reset_window()
            return

        if self.collect_count == 0 and self.delay_count == 0:
            self.source_ancestors = np.arange(self.n_walkers, dtype=np.int64)
        elif not parent_is_identity:
            self.auxiliary = self.auxiliary[parent_indices]
            self.source_ancestors = self.source_ancestors[parent_indices]
        if self.collect_count < self.block_size_steps:
            add_density_profile_to_auxiliary(
                self.auxiliary,
                positions,
                bin_edges=bin_edges,
            )
            self.collect_count += 1
        elif self.delay_count < self.lag_steps:
            self.delay_count += 1
        if self.collect_count == self.block_size_steps and self.delay_count == self.lag_steps:
            block_value = (
                np.sum(normalized_weights[:, np.newaxis] * self.auxiliary, axis=0)
                / self.block_size_steps
            )
            self._record_block(
                block_value,
                normalized_weights=normalized_weights,
                source_ancestors=self.source_ancestors,
            )
            self._reset_window()

    def block_stats(self) -> LagBlockStats:
        if self.block_count < 2:
            stderr = np.full(self.dimension, np.nan, dtype=float)
            variance = float("inf")
        else:
            stderr = np.sqrt(self.block_m2 / (self.block_count - 1)) / np.sqrt(self.block_count)
            scalar_variance = self.block_scalar_m2 / (self.block_count - 1)
            variance = _variance_inflation_from_scalar(
                self.block_scalar_mean,
                scalar_variance,
                self.block_count,
            )
        ess_min = self.block_weight_ess_min if self.block_count else 0.0
        ess_mean = self.block_weight_ess_sum / self.block_count if self.block_count else 0.0
        return LagBlockStats(
            count=self.block_count,
            mean=self.block_mean.copy(),
            stderr=stderr,
            weight_ess_min=float(ess_min),
            weight_ess_mean=float(ess_mean),
            variance_inflation=float(variance),
            source_ancestor_ess_min=(
                float(self.block_source_ancestor_ess_min) if self.block_count else 0.0
            ),
            source_ancestor_ess_mean=(
                float(self.block_source_ancestor_ess_sum / self.block_count)
                if self.block_count
                else 0.0
            ),
            unique_source_ancestor_min=(
                int(self.block_unique_source_ancestor_min) if self.block_count else 0
            ),
            max_source_family_fraction=(
                float(self.block_max_source_family_fraction) if self.block_count else 1.0
            ),
        )

    def _collect_weighted_values(
        self,
        observable_values: FloatArray,
        normalized_weights: FloatArray,
    ) -> None:
        self.weighted_collect_sum += np.sum(
            normalized_weights[:, np.newaxis] * observable_values,
            axis=0,
        )
        self.collect_count += 1

    def _record_block(
        self,
        block_value: FloatArray,
        *,
        normalized_weights: FloatArray,
        source_ancestors: IntArray,
    ) -> None:
        self._record_block_with_weight_ess(
            block_value,
            self.min_weight_ess,
            normalized_weights=normalized_weights,
            source_ancestors=source_ancestors,
        )

    def _record_block_with_weight_ess(
        self,
        block_value: FloatArray,
        weight_ess_min: float,
        *,
        normalized_weights: FloatArray,
        source_ancestors: IntArray,
    ) -> None:
        self.block_count += 1
        delta = block_value - self.block_mean
        self.block_mean += delta / self.block_count
        self.block_m2 += delta * (block_value - self.block_mean)
        scalar = float(np.mean(block_value))
        scalar_delta = scalar - self.block_scalar_mean
        self.block_scalar_mean += scalar_delta / self.block_count
        self.block_scalar_m2 += scalar_delta * (scalar - self.block_scalar_mean)
        self.block_weight_ess_min = min(self.block_weight_ess_min, weight_ess_min)
        self.block_weight_ess_sum += weight_ess_min
        ancestor_ess, unique_ancestors, max_family = _source_family_diagnostics(
            source_ancestors,
            normalized_weights,
            n_walkers=self.n_walkers,
        )
        self.block_source_ancestor_ess_min = min(
            self.block_source_ancestor_ess_min,
            ancestor_ess,
        )
        self.block_source_ancestor_ess_sum += ancestor_ess
        self.block_unique_source_ancestor_min = min(
            self.block_unique_source_ancestor_min,
            unique_ancestors,
        )
        self.block_max_source_family_fraction = max(
            self.block_max_source_family_fraction,
            max_family,
        )

    def _reset_window(self) -> None:
        self.auxiliary.fill(0.0)
        self.weighted_collect_sum.fill(0.0)
        self.collect_count = 0
        self.delay_count = 0
        self.min_weight_ess = float("inf")
        self.source_ancestors = np.arange(self.n_walkers, dtype=np.int64)


def _variance_inflation_from_scalar(mean: float, variance: float, count: int) -> float:
    if count < 2:
        return float("inf")
    if mean == 0.0:
        return 1.0
    return max(1.0, variance / (mean * mean))


@dataclass
class SlidingLagState(LagState):
    active_auxiliaries: list[FloatArray] = field(default_factory=list)
    active_ages: list[int] = field(default_factory=list)
    active_weight_ess_min: list[float] = field(default_factory=list)
    active_source_ancestors: list[IntArray] = field(default_factory=list)
    event_index: int = 0

    def step(
        self,
        *,
        parent_indices: IntArray,
        parent_is_identity: bool,
        observable_values: FloatArray,
        normalized_weights: FloatArray,
    ) -> None:
        weight_ess = float(1.0 / np.sum(normalized_weights * normalized_weights))
        should_collect = self._should_collect()
        if self.lag_steps == 0:
            block_value = np.sum(
                normalized_weights[:, np.newaxis] * observable_values,
                axis=0,
            )
            self._record_block_with_weight_ess(
                block_value,
                weight_ess,
                normalized_weights=normalized_weights,
                source_ancestors=np.arange(self.n_walkers, dtype=np.int64),
            )
            self.event_index += 1
            return
        self._advance_active(
            parent_indices=parent_indices,
            parent_is_identity=parent_is_identity,
            normalized_weights=normalized_weights,
            weight_ess=weight_ess,
        )
        if should_collect:
            self._append_active(observable_values, weight_ess)
        self.event_index += 1

    def step_density(
        self,
        *,
        parent_indices: IntArray,
        parent_is_identity: bool,
        positions: FloatArray,
        bin_edges: FloatArray | None,
        normalized_weights: FloatArray,
    ) -> None:
        weight_ess = float(1.0 / np.sum(normalized_weights * normalized_weights))
        should_collect = self._should_collect()
        if self.lag_steps == 0:
            block_value = weighted_density_profile(
                positions,
                bin_edges=bin_edges,
                walker_weights=normalized_weights,
            )
            self._record_block_with_weight_ess(
                block_value,
                weight_ess,
                normalized_weights=normalized_weights,
                source_ancestors=np.arange(self.n_walkers, dtype=np.int64),
            )
            self.event_index += 1
            return
        self._advance_active(
            parent_indices=parent_indices,
            parent_is_identity=parent_is_identity,
            normalized_weights=normalized_weights,
            weight_ess=weight_ess,
        )
        if should_collect:
            self._append_active(
                density_profile_matrix(positions, bin_edges=bin_edges),
                weight_ess,
            )
        self.event_index += 1

    def _advance_active(
        self,
        *,
        parent_indices: IntArray,
        parent_is_identity: bool,
        normalized_weights: FloatArray,
        weight_ess: float,
    ) -> None:
        if not self.active_auxiliaries:
            return
        next_auxiliaries: list[FloatArray] = []
        next_ages: list[int] = []
        next_weight_ess: list[float] = []
        next_source_ancestors: list[IntArray] = []
        for auxiliary, age, ess_min, source_ancestors in zip(
            self.active_auxiliaries,
            self.active_ages,
            self.active_weight_ess_min,
            self.active_source_ancestors,
            strict=True,
        ):
            transported = auxiliary if parent_is_identity else auxiliary[parent_indices]
            transported_ancestors = (
                source_ancestors if parent_is_identity else source_ancestors[parent_indices]
            )
            new_age = age + 1
            new_ess_min = min(ess_min, weight_ess)
            if new_age >= self.lag_steps:
                block_value = np.sum(
                    normalized_weights[:, np.newaxis] * transported,
                    axis=0,
                )
                self._record_block_with_weight_ess(
                    block_value,
                    new_ess_min,
                    normalized_weights=normalized_weights,
                    source_ancestors=transported_ancestors,
                )
            else:
                next_auxiliaries.append(transported)
                next_ages.append(new_age)
                next_weight_ess.append(new_ess_min)
                next_source_ancestors.append(transported_ancestors)
        self.active_auxiliaries = next_auxiliaries
        self.active_ages = next_ages
        self.active_weight_ess_min = next_weight_ess
        self.active_source_ancestors = next_source_ancestors

    def _append_active(self, values: FloatArray, weight_ess: float) -> None:
        self.active_auxiliaries.append(np.asarray(values, dtype=float).copy())
        self.active_ages.append(0)
        self.active_weight_ess_min.append(weight_ess)
        self.active_source_ancestors.append(np.arange(self.n_walkers, dtype=np.int64))

    def _should_collect(self) -> bool:
        return self.event_index % self.collection_stride_steps == 0


def _source_family_diagnostics(
    source_ancestors: IntArray,
    normalized_weights: FloatArray,
    *,
    n_walkers: int,
) -> tuple[float, int, float]:
    ancestors = np.asarray(source_ancestors, dtype=np.int64)
    weights = np.asarray(normalized_weights, dtype=float)
    if ancestors.shape != (n_walkers,) or weights.shape != (n_walkers,):
        raise ValueError("source ancestry diagnostics require one label and weight per walker")
    family_weights = np.bincount(
        ancestors,
        weights=weights,
        minlength=n_walkers,
    ).astype(float)
    positive = family_weights > 0.0
    unique = int(np.count_nonzero(positive))
    if unique == 0:
        return 0.0, 0, 1.0
    ess = float(1.0 / np.sum(family_weights * family_weights))
    return ess, unique, float(np.max(family_weights))
