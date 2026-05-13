from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hrdmc.estimators.pure.forward_walking.contributions import (
    add_density_profile_to_auxiliary,
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


@dataclass
class LagState:
    lag_steps: int
    block_size_steps: int
    n_walkers: int
    dimension: int
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

    def __post_init__(self) -> None:
        self.auxiliary = np.zeros((self.n_walkers, self.dimension), dtype=float)
        self.weighted_collect_sum = np.zeros(self.dimension, dtype=float)
        self.block_mean = np.zeros(self.dimension, dtype=float)
        self.block_m2 = np.zeros(self.dimension, dtype=float)

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
                self._record_block(self.weighted_collect_sum / self.block_size_steps)
                self._reset_window()
            return

        if not parent_is_identity:
            self.auxiliary = self.auxiliary[parent_indices]
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
            self._record_block(block_value)
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
                self._record_block(self.weighted_collect_sum / self.block_size_steps)
                self._reset_window()
            return

        if not parent_is_identity:
            self.auxiliary = self.auxiliary[parent_indices]
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
            self._record_block(block_value)
            self._reset_window()

    def block_stats(self) -> LagBlockStats:
        if self.block_count < 2:
            stderr = np.full(self.dimension, np.nan, dtype=float)
            variance = float("inf")
        else:
            stderr = np.sqrt(self.block_m2 / (self.block_count - 1)) / np.sqrt(
                self.block_count
            )
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

    def _record_block(self, block_value: FloatArray) -> None:
        self.block_count += 1
        delta = block_value - self.block_mean
        self.block_mean += delta / self.block_count
        self.block_m2 += delta * (block_value - self.block_mean)
        scalar = float(np.mean(block_value))
        scalar_delta = scalar - self.block_scalar_mean
        self.block_scalar_mean += scalar_delta / self.block_count
        self.block_scalar_m2 += scalar_delta * (scalar - self.block_scalar_mean)
        self.block_weight_ess_min = min(self.block_weight_ess_min, self.min_weight_ess)
        self.block_weight_ess_sum += self.min_weight_ess

    def _reset_window(self) -> None:
        self.auxiliary.fill(0.0)
        self.weighted_collect_sum.fill(0.0)
        self.collect_count = 0
        self.delay_count = 0
        self.min_weight_ess = float("inf")


def _variance_inflation_from_scalar(mean: float, variance: float, count: int) -> float:
    if count < 2:
        return float("inf")
    if mean == 0.0:
        return 1.0
    return max(1.0, variance / (mean * mean))
