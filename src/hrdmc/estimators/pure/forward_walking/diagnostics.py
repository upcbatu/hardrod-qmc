from __future__ import annotations

import numpy as np

from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.results import LagValue


def standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def rms_delta_stderr(r2_value: float | None, r2_stderr: float | None) -> float:
    if r2_value is None or r2_stderr is None:
        return float("nan")
    if not np.isfinite(r2_value) or not np.isfinite(r2_stderr) or r2_value <= 0.0:
        return float("nan")
    return float(0.5 * r2_stderr / np.sqrt(r2_value))


def variance_inflation(values: np.ndarray) -> float:
    if values.size < 2:
        return float("inf")
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    if mean == 0.0:
        return 1.0
    return max(1.0, variance / (mean * mean))


def schema_status(
    *,
    config: PureWalkingConfig,
    values_by_lag: dict[int, LagValue],
    mixed_r2_reference: float | None,
    mixed_rms_radius_reference: float | None,
) -> str:
    for value in values_by_lag.values():
        arr = np.asarray(value, dtype=float)
        if np.any(np.isfinite(arr) & (arr < 0.0)):
            return "PURE_WALKING_SCHEMA_NO_GO"
    lag0 = values_by_lag.get(0)
    if lag0 is None or not np.all(np.isfinite(np.asarray(lag0, dtype=float))):
        return "PURE_WALKING_SCHEMA_NO_GO"
    if mixed_r2_reference is not None and not np.isclose(
        float(np.asarray(lag0, dtype=float).reshape(-1)[0]),
        mixed_r2_reference,
        rtol=config.schema_rtol,
        atol=config.schema_atol,
    ):
        return "PURE_WALKING_SCHEMA_NO_GO"
    if mixed_rms_radius_reference is not None and not np.isclose(
        np.sqrt(float(np.asarray(lag0, dtype=float).reshape(-1)[0])),
        mixed_rms_radius_reference,
        rtol=config.schema_rtol,
        atol=config.schema_atol,
    ):
        return "PURE_WALKING_SCHEMA_NO_GO"
    return "SCHEMA_GO"


def plateau_summary(
    *,
    config: PureWalkingConfig,
    values_by_lag: dict[int, LagValue],
    stderr_by_lag: dict[int, LagValue],
    block_count_by_lag: dict[int, int],
    weight_ess_min_by_lag: dict[int, float],
) -> tuple[LagValue | None, LagValue | None, tuple[LagValue, LagValue] | None, str]:
    value_lags = [
        lag for lag in config.lag_steps if _all_finite(values_by_lag.get(lag))
    ]
    if not value_lags:
        return None, None, None, "NO_BLOCKS"
    lag_max = value_lags[-1]
    if block_count_by_lag[lag_max] < config.min_block_count:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "INSUFFICIENT_BLOCKS"
    if weight_ess_min_by_lag[lag_max] < config.min_walker_weight_ess:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "INSUFFICIENT_WEIGHT_ESS"
    finite_lags = [
        lag for lag in value_lags if _all_finite(stderr_by_lag.get(lag))
    ]
    if not finite_lags:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "INSUFFICIENT_BLOCKS"
    if len(finite_lags) == 1:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "NO_LAG_PLATEAU"
    lag_prev = finite_lags[-2]
    bracket = (values_by_lag[lag_prev], values_by_lag[lag_max])
    if block_count_by_lag[lag_prev] < config.min_block_count:
        return (
            values_by_lag[lag_max],
            stderr_by_lag[lag_max],
            bracket,
            "INSUFFICIENT_BLOCKS",
        )
    if weight_ess_min_by_lag[lag_prev] < config.min_walker_weight_ess:
        return (
            values_by_lag[lag_max],
            stderr_by_lag[lag_max],
            bracket,
            "INSUFFICIENT_WEIGHT_ESS",
        )
    delta = float(
        np.max(
            np.abs(
                np.asarray(values_by_lag[lag_max], dtype=float)
                - np.asarray(values_by_lag[lag_prev], dtype=float)
            )
        )
    )
    combined = float(
        np.max(
            np.hypot(
                np.asarray(stderr_by_lag[lag_max], dtype=float),
                np.asarray(stderr_by_lag[lag_prev], dtype=float),
            )
        )
    )
    threshold = config.plateau_abs_tolerance + config.plateau_sigma_threshold * combined
    status = "PLATEAU_FOUND" if delta <= threshold else "NO_LAG_PLATEAU"
    stderr = np.maximum(
        np.asarray(stderr_by_lag[lag_max], dtype=float),
        np.asarray(stderr_by_lag[lag_prev], dtype=float),
    )
    return (
        values_by_lag[lag_max],
        float(stderr[0]) if stderr.shape == (1,) else stderr,
        bracket,
        status,
    )


def _all_finite(value: object) -> bool:
    if value is None:
        return False
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
