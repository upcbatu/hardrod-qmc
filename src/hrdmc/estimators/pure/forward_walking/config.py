from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PureWalkingConfig:
    """Transported auxiliary forward-walking configuration."""

    lag_steps: tuple[int, ...]
    lag_unit: str = "dmc_steps"
    observables: tuple[str, ...] = ("r2",)
    observable_source: str = "raw_r2"
    density_bin_edges: FloatArray | None = None
    pair_bin_edges: FloatArray | None = None
    structure_k_values: FloatArray | None = None
    min_block_count: int = 30
    min_walker_weight_ess: float = 30.0
    block_size_steps: int = 20
    transport_mode: str = "post_resample_auxiliary"
    collection_mode: str = "single_point"
    center: float = 0.0
    plateau_sigma_threshold: float = 1.0
    plateau_abs_tolerance: float = 0.0
    schema_atol: float = 1.0e-12
    schema_rtol: float = 1.0e-12
    transport_invariant_tests_passed: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.lag_unit != "dmc_steps":
            raise ValueError("transported auxiliary FW requires lag_unit='dmc_steps'")
        if not self.lag_steps:
            raise ValueError("lag_steps must be non-empty")
        if tuple(sorted(set(self.lag_steps))) != self.lag_steps:
            raise ValueError("lag_steps must be sorted and unique")
        if any(lag < 0 for lag in self.lag_steps):
            raise ValueError("lag_steps must be non-negative")
        if 0 not in self.lag_steps:
            raise ValueError("lag_steps must include 0 for schema identity checks")
        supported = {"r2", "density", "pair_distance_density", "structure_factor"}
        unsupported = set(self.observables) - supported
        if unsupported:
            raise ValueError(f"unsupported pure-walking observables: {sorted(unsupported)}")
        if self.observable_source not in {"raw_r2", "r2_rb"}:
            raise ValueError("observable_source must be 'raw_r2' or 'r2_rb'")
        if "density" in self.observables:
            _validate_edges(self.density_bin_edges, "density_bin_edges")
        if "pair_distance_density" in self.observables:
            _validate_edges(self.pair_bin_edges, "pair_bin_edges")
        if "structure_factor" in self.observables:
            if self.structure_k_values is None:
                raise ValueError("structure_k_values must be provided for structure_factor")
            k_values = np.asarray(self.structure_k_values, dtype=float)
            if k_values.ndim != 1 or k_values.size == 0 or not np.all(np.isfinite(k_values)):
                raise ValueError("structure_k_values must be a finite one-dimensional array")
        if self.min_block_count <= 0:
            raise ValueError("min_block_count must be positive")
        if self.min_walker_weight_ess <= 0.0:
            raise ValueError("min_walker_weight_ess must be positive")
        if self.block_size_steps <= 0:
            raise ValueError("block_size_steps must be positive")
        if self.transport_mode != "post_resample_auxiliary":
            raise ValueError("unsupported transport_mode")
        if self.collection_mode != "single_point":
            raise ValueError(
                "transported auxiliary FW currently supports "
                "collection_mode='single_point'"
            )
        if any(lag > 0 for lag in self.lag_steps) and self.block_size_steps != 1:
            raise ValueError(
                "single_point lagged FW requires block_size_steps=1; "
                "use sliding_window before collecting multi-step blocks"
            )


def _validate_edges(edges: FloatArray | None, name: str) -> None:
    if edges is None:
        raise ValueError(f"{name} must be provided")
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional edge array")
    widths = np.diff(values)
    if not np.all(np.isfinite(values)) or np.any(widths <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
