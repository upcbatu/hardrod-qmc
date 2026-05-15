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
    window_lags = finite_lags[-min(config.plateau_window_lag_count, len(finite_lags)) :]
    for lag in window_lags:
        if block_count_by_lag[lag] < config.min_block_count:
            bracket = (values_by_lag[window_lags[0]], values_by_lag[lag_max])
            return (
                values_by_lag[lag_max],
                stderr_by_lag[lag_max],
                bracket,
                "INSUFFICIENT_BLOCKS",
                {
                    "method": "late_window",
                    "window_lags": list(window_lags),
                    "lag": lag,
                    "lag_max": lag_max,
                    "block_count": block_count_by_lag[lag],
                    "min_block_count": config.min_block_count,
                },
            )
        if weight_ess_min_by_lag[lag] < config.min_walker_weight_ess:
            bracket = (values_by_lag[window_lags[0]], values_by_lag[lag_max])
            return (
                values_by_lag[lag_max],
                stderr_by_lag[lag_max],
                bracket,
                "INSUFFICIENT_WEIGHT_ESS",
                {
                    "method": "late_window",
                    "window_lags": list(window_lags),
                    "lag": lag,
                    "lag_max": lag_max,
                    "weight_ess_min": weight_ess_min_by_lag[lag],
                    "min_walker_weight_ess": config.min_walker_weight_ess,
                },
            )
    if len(window_lags) == 2:
        return _last_two_plateau_summary(
            config=config,
            observable=observable,
            lag_prev=window_lags[0],
            lag_max=window_lags[1],
            values_by_lag=values_by_lag,
            stderr_by_lag=stderr_by_lag,
            block_count_by_lag=block_count_by_lag,
            weight_ess_min_by_lag=weight_ess_min_by_lag,
        )
    return _late_window_plateau_summary(
        config=config,
        observable=observable,
        window_lags=tuple(window_lags),
        values_by_lag=values_by_lag,
        stderr_by_lag=stderr_by_lag,
        block_count_by_lag=block_count_by_lag,
        weight_ess_min_by_lag=weight_ess_min_by_lag,
    )


def _last_two_plateau_summary(
    *,
    config: PureWalkingConfig,
    observable: str,
    lag_prev: int,
    lag_max: int,
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
    else:
        norm = "max_abs"
        base_tolerance = config.plateau_abs_tolerance
    threshold = _plateau_threshold(
        config=config,
        observable=observable,
        combined_stderr=combined,
    )
    status = "PLATEAU_FOUND" if delta <= threshold else "NO_LAG_PLATEAU"
    stderr = np.maximum(
        _lag_array(stderr_by_lag[lag_max]),
        _lag_array(stderr_by_lag[lag_prev]),
    )
    return (
        values_by_lag[lag_max],
        _lag_value(stderr),
        bracket,
        status,
        {
            "method": "last_two",
            "observable": observable,
            "norm": norm,
            "lag_previous": lag_prev,
            "lag_max": lag_max,
            "window_lags": [lag_prev, lag_max],
            "plateau_window_lag_count": config.plateau_window_lag_count,
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


def _late_window_plateau_summary(
    *,
    config: PureWalkingConfig,
    observable: str,
    window_lags: tuple[int, ...],
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
    values = np.stack([_lag_array(values_by_lag[lag]) for lag in window_lags], axis=0)
    stderrs = np.stack([_lag_array(stderr_by_lag[lag]) for lag in window_lags], axis=0)
    plateau = np.mean(values, axis=0)
    mean_stderr = np.sqrt(np.sum(stderrs * stderrs, axis=0)) / float(len(window_lags))
    sample_stderr = np.std(values, axis=0, ddof=1) / np.sqrt(float(len(window_lags)))
    plateau_stderr = np.maximum(np.max(stderrs, axis=0), sample_stderr)
    norm = "relative_l2" if observable == "density" else "max_abs"
    base_tolerance = (
        config.density_plateau_relative_l2_tolerance
        if observable == "density"
        else config.plateau_abs_tolerance
    )
    plateau_value = _lag_value(plateau)
    plateau_stderr_value = _lag_value(plateau_stderr)
    bracket = (values_by_lag[window_lags[0]], values_by_lag[window_lags[-1]])

    window_deltas: dict[int, float] = {}
    window_combined_stderr: dict[int, float] = {}
    window_thresholds: dict[int, float] = {}
    spread_pass = True
    for lag in window_lags:
        delta, combined = _plateau_delta_and_stderr(
            config=config,
            observable=observable,
            value_prev=values_by_lag[lag],
            value_max=plateau_value,
            stderr_prev=stderr_by_lag[lag],
            stderr_max=_lag_value(mean_stderr),
        )
        threshold = _plateau_threshold(
            config=config,
            observable=observable,
            combined_stderr=combined,
        )
        window_deltas[lag] = delta
        window_combined_stderr[lag] = combined
        window_thresholds[lag] = threshold
        spread_pass = spread_pass and delta <= threshold

    slope_delta, slope_combined = _window_slope_delta_and_stderr(
        config=config,
        observable=observable,
        window_lags=window_lags,
        values=values,
        stderrs=stderrs,
        plateau=plateau,
    )
    slope_threshold = _plateau_threshold(
        config=config,
        observable=observable,
        combined_stderr=slope_combined,
    )
    slope_pass = slope_delta <= slope_threshold
    status = "PLATEAU_FOUND" if spread_pass and slope_pass else "NO_LAG_PLATEAU"
    return (
        plateau_value,
        plateau_stderr_value,
        bracket,
        status,
        {
            "method": "late_window",
            "observable": observable,
            "norm": norm,
            "window_lags": list(window_lags),
            "lag_max": window_lags[-1],
            "plateau_window_lag_count": config.plateau_window_lag_count,
            "window_delta_by_lag": window_deltas,
            "window_combined_stderr_by_lag": window_combined_stderr,
            "window_threshold_by_lag": window_thresholds,
            "max_window_delta": max(window_deltas.values()),
            "min_window_threshold": min(window_thresholds.values()),
            "window_spread_pass": spread_pass,
            "slope_delta": slope_delta,
            "slope_combined_stderr": slope_combined,
            "slope_threshold": slope_threshold,
            "slope_pass": slope_pass,
            "base_tolerance": base_tolerance,
            "sigma_threshold": config.plateau_sigma_threshold,
            "decision": status,
            "block_count_by_window_lag": {
                lag: block_count_by_lag[lag] for lag in window_lags
            },
            "weight_ess_min_by_window_lag": {
                lag: weight_ess_min_by_lag[lag] for lag in window_lags
            },
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
    previous = _lag_array(value_prev)
    current = _lag_array(value_max)
    previous_stderr = _lag_array(stderr_prev)
    current_stderr = _lag_array(stderr_max)
    if previous.shape != current.shape or previous_stderr.shape != current_stderr.shape:
        return float("inf"), float("inf")
    delta = _delta_norm(
        config=config,
        observable=observable,
        difference=current - previous,
        scale=current,
    )
    combined = _stderr_norm(
        config=config,
        observable=observable,
        stderr=np.hypot(current_stderr, previous_stderr),
        scale=current,
    )
    return delta, combined


def _delta_norm(
    *,
    config: PureWalkingConfig,
    observable: str,
    difference: np.ndarray,
    scale: np.ndarray,
) -> float:
    if observable == "density":
        edges = np.asarray(config.density_bin_edges, dtype=float)
        widths = np.diff(edges)
        if widths.shape != scale.shape or difference.shape != scale.shape:
            return float("inf")
        denominator = float(np.sqrt(np.sum(scale * scale * widths)))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("inf")
        return float(np.sqrt(np.sum(difference * difference * widths)) / denominator)
    return float(np.max(np.abs(difference)))


def _stderr_norm(
    *,
    config: PureWalkingConfig,
    observable: str,
    stderr: np.ndarray,
    scale: np.ndarray,
) -> float:
    if observable == "density":
        edges = np.asarray(config.density_bin_edges, dtype=float)
        widths = np.diff(edges)
        if widths.shape != scale.shape or stderr.shape != scale.shape:
            return float("inf")
        denominator = float(np.sqrt(np.sum(scale * scale * widths)))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("inf")
        return float(np.sqrt(np.sum(stderr * stderr * widths)) / denominator)
    return float(np.max(np.abs(stderr)))


def _window_slope_delta_and_stderr(
    *,
    config: PureWalkingConfig,
    observable: str,
    window_lags: tuple[int, ...],
    values: np.ndarray,
    stderrs: np.ndarray,
    plateau: np.ndarray,
) -> tuple[float, float]:
    lags = np.asarray(window_lags, dtype=float)
    centered = lags - float(np.mean(lags))
    denominator = float(np.sum(centered * centered))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("inf"), float("inf")
    span = float(lags[-1] - lags[0])
    slope = np.sum(centered[:, None] * values, axis=0) / denominator
    span_change = slope * span
    slope_stderr = (
        np.sqrt(np.sum((centered[:, None] * stderrs) ** 2, axis=0))
        * span
        / denominator
    )
    delta = _delta_norm(
        config=config,
        observable=observable,
        difference=span_change,
        scale=plateau,
    )
    combined = _stderr_norm(
        config=config,
        observable=observable,
        stderr=slope_stderr,
        scale=plateau,
    )
    return delta, combined


def _plateau_threshold(
    *,
    config: PureWalkingConfig,
    observable: str,
    combined_stderr: float,
) -> float:
    base_tolerance = (
        config.density_plateau_relative_l2_tolerance
        if observable == "density"
        else config.plateau_abs_tolerance
    )
    return base_tolerance + config.plateau_sigma_threshold * combined_stderr


def _lag_array(value: LagValue) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _lag_value(value: np.ndarray) -> LagValue:
    if value.shape == (1,):
        return float(value[0])
    return value


def _all_finite(value: object) -> bool:
    if value is None:
        return False
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
