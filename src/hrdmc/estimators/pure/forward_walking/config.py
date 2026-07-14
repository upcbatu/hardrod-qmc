from __future__ import annotations

from dataclasses import dataclass, replace

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
    r2_rb_com_variance: float | None = None
    density_source: str = "raw_density"
    density_com_variance: float | None = None
    density_parity_average: bool = False
    density_expected_particles: float | None = None
    density_accounting_abs_tolerance: float = 5.0e-3
    density_bin_edges: FloatArray | None = None
    pair_bin_edges: FloatArray | None = None
    structure_k_values: FloatArray | None = None
    min_block_count: int = 30
    min_walker_weight_ess: float = 30.0
    min_source_ancestor_ess: float = 1.0
    max_source_family_fraction: float = 1.0
    block_size_steps: int = 1
    collection_stride_steps: int = 1
    transport_mode: str = "post_resample_auxiliary"
    collection_mode: str = "sliding_window"
    center: float = 0.0
    plateau_sigma_threshold: float = 1.0
    plateau_abs_tolerance: float = 0.0
    plateau_window_lag_count: int = 4
    density_lag_steps: tuple[int, ...] | None = None
    density_collection_stride_steps: int | None = None
    density_plateau_window_lag_count: int | None = None
    density_plateau_relative_l2_tolerance: float = 0.03
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
        if self.observable_source == "r2_rb" and self.r2_rb_com_variance is None:
            raise ValueError("r2_rb requires r2_rb_com_variance")
        if self.r2_rb_com_variance is not None and (
            not np.isfinite(self.r2_rb_com_variance) or self.r2_rb_com_variance < 0.0
        ):
            raise ValueError("r2_rb_com_variance must be finite and non-negative")
        if self.density_source not in {"raw_density", "com_rao_blackwell"}:
            raise ValueError("density_source must be 'raw_density' or 'com_rao_blackwell'")
        if self.density_source == "com_rao_blackwell" and self.density_com_variance is None:
            raise ValueError("com_rao_blackwell density requires density_com_variance")
        if self.density_com_variance is not None and (
            not np.isfinite(self.density_com_variance) or self.density_com_variance <= 0.0
        ):
            raise ValueError("density_com_variance must be finite and positive")
        if self.density_expected_particles is not None and (
            not np.isfinite(self.density_expected_particles)
            or self.density_expected_particles <= 0.0
        ):
            raise ValueError("density_expected_particles must be finite and positive")
        if (
            not np.isfinite(self.density_accounting_abs_tolerance)
            or self.density_accounting_abs_tolerance < 0.0
        ):
            raise ValueError("density_accounting_abs_tolerance must be finite and non-negative")
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
        if self.min_source_ancestor_ess <= 0.0:
            raise ValueError("min_source_ancestor_ess must be positive")
        if not 0.0 < self.max_source_family_fraction <= 1.0:
            raise ValueError("max_source_family_fraction must lie in (0, 1]")
        if self.block_size_steps <= 0:
            raise ValueError("block_size_steps must be positive")
        if self.collection_stride_steps <= 0:
            raise ValueError("collection_stride_steps must be positive")
        if self.density_plateau_relative_l2_tolerance < 0.0:
            raise ValueError("density_plateau_relative_l2_tolerance must be non-negative")
        if self.plateau_window_lag_count < 2:
            raise ValueError("plateau_window_lag_count must be at least 2")
        if self.density_lag_steps is not None:
            if tuple(sorted(set(self.density_lag_steps))) != self.density_lag_steps:
                raise ValueError("density_lag_steps must be sorted and unique")
            if any(lag < 0 for lag in self.density_lag_steps):
                raise ValueError("density_lag_steps must be non-negative")
            if 0 not in self.density_lag_steps:
                raise ValueError("density_lag_steps must include 0")
        if (
            self.density_collection_stride_steps is not None
            and self.density_collection_stride_steps <= 0
        ):
            raise ValueError("density_collection_stride_steps must be positive")
        if (
            self.density_plateau_window_lag_count is not None
            and self.density_plateau_window_lag_count < 2
        ):
            raise ValueError("density_plateau_window_lag_count must be at least 2")
        if self.transport_mode != "post_resample_auxiliary":
            raise ValueError("unsupported transport_mode")
        if self.collection_mode not in {"single_point", "sliding_window"}:
            raise ValueError("unsupported transported FW collection_mode")
        if (
            "density" in self.observables
            and (self.density_source != "raw_density" or self.density_parity_average)
            and self.collection_mode != "sliding_window"
        ):
            raise ValueError(
                "COM-integrated or parity-averaged density requires sliding_window collection"
            )
        if (
            self.collection_mode == "sliding_window"
            and any(lag > 0 for lag in self.lag_steps)
            and self.block_size_steps != 1
        ):
            raise ValueError("lagged sliding_window FW requires block_size_steps=1")
        if (
            self.collection_mode == "single_point"
            and any(lag > 0 for lag in self.lag_steps)
            and self.block_size_steps != 1
        ):
            raise ValueError(
                "single_point lagged FW requires block_size_steps=1; "
                "use sliding_window before collecting multi-step blocks"
            )

    def for_observable(self, observable: str) -> PureWalkingConfig:
        """Return the transport settings used by one observable stream.

        Density snapshots are much larger than scalar R2 payloads.  A separate
        density ladder can therefore retain the same parent-map transport while
        using a coarser snapshot cadence.
        """

        if observable != "density" or self.density_lag_steps is None:
            return self
        return replace(
            self,
            lag_steps=self.density_lag_steps,
            collection_stride_steps=(
                self.collection_stride_steps
                if self.density_collection_stride_steps is None
                else self.density_collection_stride_steps
            ),
            plateau_window_lag_count=(
                self.plateau_window_lag_count
                if self.density_plateau_window_lag_count is None
                else self.density_plateau_window_lag_count
            ),
            density_lag_steps=None,
            density_collection_stride_steps=None,
            density_plateau_window_lag_count=None,
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
