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
    observable: str,
    values_by_lag: dict[int, LagValue],
    mixed_observable_reference: LagValue | None,
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
    if (
        "lag0_identity" in config.transport_invariant_tests_passed
        and mixed_observable_reference is None
        and mixed_r2_reference is None
    ):
        return "PURE_WALKING_SCHEMA_NO_GO"
    if mixed_observable_reference is not None and not _lag0_matches_reference(
        config=config,
        observable=observable,
        lag0=lag0,
        reference=mixed_observable_reference,
    ):
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
    observable: str,
    values_by_lag: dict[int, LagValue],
    stderr_by_lag: dict[int, LagValue],
    block_count_by_lag: dict[int, int],
    weight_ess_min_by_lag: dict[int, float],
) -> tuple[
    LagValue | None,
    LagValue | None,
    tuple[LagValue, LagValue] | None,
    str,
    dict[str, object],
]:
    value_lags = [
        lag for lag in config.lag_steps if _all_finite(values_by_lag.get(lag))
    ]
    if not value_lags:
        return None, None, None, "NO_BLOCKS", {"reason": "no_finite_lag_values"}
    lag_max = value_lags[-1]
    if block_count_by_lag[lag_max] < config.min_block_count:
        value = values_by_lag[lag_max]
        return (
            value,
            stderr_by_lag[lag_max],
            (value, value),
            "INSUFFICIENT_BLOCKS",
            {
                "lag_max": lag_max,
                "block_count": block_count_by_lag[lag_max],
                "min_block_count": config.min_block_count,
            },
        )
    if weight_ess_min_by_lag[lag_max] < config.min_walker_weight_ess:
        value = values_by_lag[lag_max]
        return (
            value,
            stderr_by_lag[lag_max],
            (value, value),
            "INSUFFICIENT_WEIGHT_ESS",
            {
                "lag_max": lag_max,
                "weight_ess_min": weight_ess_min_by_lag[lag_max],
                "min_walker_weight_ess": config.min_walker_weight_ess,
            },
        )
    finite_lags = [
        lag for lag in value_lags if _all_finite(stderr_by_lag.get(lag))
    ]
    if not finite_lags:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "INSUFFICIENT_BLOCKS", {
            "reason": "no_finite_lag_stderr",
            "lag_max": lag_max,
        }
    if len(finite_lags) == 1:
        value = values_by_lag[lag_max]
        return value, stderr_by_lag[lag_max], (value, value), "NO_LAG_PLATEAU", {
            "reason": "only_one_finite_lag",
            "lag_max": lag_max,
        }
    lag_prev = finite_lags[-2]
    bracket = (values_by_lag[lag_prev], values_by_lag[lag_max])
    if block_count_by_lag[lag_prev] < config.min_block_count:
        return (
            values_by_lag[lag_max],
            stderr_by_lag[lag_max],
            bracket,
            "INSUFFICIENT_BLOCKS",
            {
                "lag_previous": lag_prev,
                "lag_max": lag_max,
                "previous_block_count": block_count_by_lag[lag_prev],
                "min_block_count": config.min_block_count,
            },
        )
    if weight_ess_min_by_lag[lag_prev] < config.min_walker_weight_ess:
        return (
            values_by_lag[lag_max],
            stderr_by_lag[lag_max],
            bracket,
            "INSUFFICIENT_WEIGHT_ESS",
            {
                "lag_previous": lag_prev,
                "lag_max": lag_max,
                "previous_weight_ess_min": weight_ess_min_by_lag[lag_prev],
                "min_walker_weight_ess": config.min_walker_weight_ess,
            },
        )
    delta, combined = _plateau_delta_and_stderr(
        config=config,
        observable=observable,
        value_prev=values_by_lag[lag_prev],
        value_max=values_by_lag[lag_max],
        stderr_prev=stderr_by_lag[lag_prev],
        stderr_max=stderr_by_lag[lag_max],
    )
    if observable == "density":
        norm = "relative_l2"
        base_tolerance = config.density_plateau_relative_l2_tolerance
        threshold = (
            base_tolerance
            + config.plateau_sigma_threshold * combined
        )
    else:
        norm = "max_abs"
        base_tolerance = config.plateau_abs_tolerance
        threshold = base_tolerance + config.plateau_sigma_threshold * combined
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
        {
            "observable": observable,
            "norm": norm,
            "lag_previous": lag_prev,
            "lag_max": lag_max,
            "delta": delta,
            "combined_stderr": combined,
            "base_tolerance": base_tolerance,
            "sigma_threshold": config.plateau_sigma_threshold,
            "threshold": threshold,
            "decision": status,
            "previous_block_count": block_count_by_lag[lag_prev],
            "max_block_count": block_count_by_lag[lag_max],
            "previous_weight_ess_min": weight_ess_min_by_lag[lag_prev],
            "max_weight_ess_min": weight_ess_min_by_lag[lag_max],
        },
    )


def _lag0_matches_reference(
    *,
    config: PureWalkingConfig,
    observable: str,
    lag0: LagValue,
    reference: LagValue,
) -> bool:
    lag0_arr = np.asarray(lag0, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if lag0_arr.shape != ref_arr.shape:
        return False
    if not np.all(np.isfinite(lag0_arr)) or not np.all(np.isfinite(ref_arr)):
        return False
    if observable == "density":
        edges = np.asarray(config.density_bin_edges, dtype=float)
        widths = np.diff(edges)
        if widths.shape != lag0_arr.shape:
            return False
        denominator = float(np.sqrt(np.sum(ref_arr * ref_arr * widths)))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return False
        relative_l2 = float(
            np.sqrt(np.sum((lag0_arr - ref_arr) ** 2 * widths)) / denominator
        )
        return relative_l2 <= max(config.schema_atol, config.schema_rtol)
    return bool(
        np.allclose(lag0_arr, ref_arr, rtol=config.schema_rtol, atol=config.schema_atol)
    )


def _plateau_delta_and_stderr(
    *,
    config: PureWalkingConfig,
    observable: str,
    value_prev: LagValue,
    value_max: LagValue,
    stderr_prev: LagValue,
    stderr_max: LagValue,
) -> tuple[float, float]:
    previous = np.asarray(value_prev, dtype=float)
    current = np.asarray(value_max, dtype=float)
    previous_stderr = np.asarray(stderr_prev, dtype=float)
    current_stderr = np.asarray(stderr_max, dtype=float)
    if observable == "density":
        edges = np.asarray(config.density_bin_edges, dtype=float)
        widths = np.diff(edges)
        if widths.shape != current.shape:
            return float("inf"), float("inf")
        scale = float(np.sqrt(np.sum(current * current * widths)))
        if scale <= 0.0 or not np.isfinite(scale):
            return float("inf"), float("inf")
        delta = float(np.sqrt(np.sum((current - previous) ** 2 * widths)) / scale)
        combined_by_bin = np.hypot(current_stderr, previous_stderr)
        combined = float(np.sqrt(np.sum(combined_by_bin * combined_by_bin * widths)) / scale)
        return delta, combined
    delta = float(np.max(np.abs(current - previous)))
    combined = float(np.max(np.hypot(current_stderr, previous_stderr)))
    return delta, combined


def _all_finite(value: object) -> bool:
    if value is None:
        return False
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
