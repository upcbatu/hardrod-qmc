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
    rms_plateau_relative_tolerance: float = 0.0
    plateau_equivalence_confidence_level: float = 0.95
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
        _validate_lags(self.lag_steps, "lag_steps", require_nonempty=True)
        _validate_sources(self)
        _validate_thresholds(self)
        _validate_collection(self)

    def for_observable(self, observable: str) -> PureWalkingConfig:
        """Return the transport settings used by one observable stream."""
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


def _validate_lags(lags: tuple[int, ...], name: str, *, require_nonempty: bool) -> None:
    if require_nonempty and not lags:
        raise ValueError(f"{name} must be non-empty")
    if tuple(sorted(set(lags))) != lags:
        raise ValueError(f"{name} must be sorted and unique")
    if any(lag < 0 for lag in lags):
        raise ValueError(f"{name} must be non-negative")
    if 0 not in lags:
        suffix = " for schema identity checks" if name == "lag_steps" else ""
        raise ValueError(f"{name} must include 0{suffix}")


def _validate_sources(config: PureWalkingConfig) -> None:
    unsupported = set(config.observables) - {"r2", "density"}
    if unsupported:
        raise ValueError(f"unsupported pure-walking observables: {sorted(unsupported)}")
    if config.observable_source not in {"raw_r2", "r2_rb"}:
        raise ValueError("observable_source must be 'raw_r2' or 'r2_rb'")
    if config.observable_source == "r2_rb" and config.r2_rb_com_variance is None:
        raise ValueError("r2_rb requires r2_rb_com_variance")
    _validate_optional_finite(config.r2_rb_com_variance, "r2_rb_com_variance", allow_zero=True)
    if config.density_source not in {"raw_density", "com_rao_blackwell"}:
        raise ValueError("density_source must be 'raw_density' or 'com_rao_blackwell'")
    if config.density_source == "com_rao_blackwell" and config.density_com_variance is None:
        raise ValueError("com_rao_blackwell density requires density_com_variance")
    _validate_optional_finite(config.density_com_variance, "density_com_variance")
    _validate_optional_finite(config.density_expected_particles, "density_expected_particles")
    if "density" in config.observables:
        _validate_edges(config.density_bin_edges, "density_bin_edges")


def _validate_optional_finite(value: float | None, name: str, *, allow_zero: bool = False) -> None:
    if value is None:
        return
    valid_bound = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _validate_thresholds(config: PureWalkingConfig) -> None:
    positive = {
        "min_block_count": config.min_block_count,
        "min_walker_weight_ess": config.min_walker_weight_ess,
        "min_source_ancestor_ess": config.min_source_ancestor_ess,
        "block_size_steps": config.block_size_steps,
        "collection_stride_steps": config.collection_stride_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < config.max_source_family_fraction <= 1.0:
        raise ValueError("max_source_family_fraction must lie in (0, 1]")
    for name, value in (
        ("density_accounting_abs_tolerance", config.density_accounting_abs_tolerance),
        ("rms_plateau_relative_tolerance", config.rms_plateau_relative_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.density_plateau_relative_l2_tolerance < 0.0:
        raise ValueError("density_plateau_relative_l2_tolerance must be non-negative")
    if not 0.0 < config.plateau_equivalence_confidence_level < 1.0:
        raise ValueError(
            "plateau_equivalence_confidence_level must lie strictly between zero and one"
        )
    if config.plateau_window_lag_count < 2:
        raise ValueError("plateau_window_lag_count must be at least 2")


def _validate_collection(config: PureWalkingConfig) -> None:
    if config.density_lag_steps is not None:
        _validate_lags(config.density_lag_steps, "density_lag_steps", require_nonempty=False)
    if (
        config.density_collection_stride_steps is not None
        and config.density_collection_stride_steps <= 0
    ):
        raise ValueError("density_collection_stride_steps must be positive")
    if (
        config.density_plateau_window_lag_count is not None
        and config.density_plateau_window_lag_count < 2
    ):
        raise ValueError("density_plateau_window_lag_count must be at least 2")
    if config.transport_mode != "post_resample_auxiliary":
        raise ValueError("unsupported transport_mode")
    if config.collection_mode not in {"single_point", "sliding_window"}:
        raise ValueError("unsupported transported FW collection_mode")
    advanced_density = config.density_source != "raw_density" or config.density_parity_average
    if (
        "density" in config.observables
        and advanced_density
        and config.collection_mode != "sliding_window"
    ):
        raise ValueError(
            "COM-integrated or parity-averaged density requires sliding_window collection"
        )
    if any(lag > 0 for lag in config.lag_steps) and config.block_size_steps != 1:
        if config.collection_mode == "sliding_window":
            raise ValueError("lagged sliding_window FW requires block_size_steps=1")
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
