from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.results import (
    GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    GENEALOGY_NOT_EVALUATED,
    GENEALOGY_SOURCE_FAMILY_DOMINANCE,
    GENEALOGY_SUPPORT_ACCEPTED,
    PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    PLATEAU_INSUFFICIENT_BLOCKS,
    PLATEAU_NO_BLOCKS,
    PLATEAU_RESOLVED,
    PLATEAU_UNRESOLVED,
    SCHEMA_INVALID,
    SCHEMA_VALID,
    LagValue,
)

MIN_AGGREGATE_PLATEAU_LAGS = 3


def aggregate_fw_observable_summary(
    seed_payloads: list[dict[str, Any]],
    *,
    observable: str,
    config: PureWalkingConfig,
) -> dict[str, Any]:
    """Aggregate independent-seed FW ladders on a genealogy-supported window."""

    observable_config = config.for_observable(observable)
    preflight = _preflight(seed_payloads, observable=observable)
    if preflight is not None:
        return preflight
    semantic_failure = _semantic_preflight(
        seed_payloads,
        observable=observable,
        config=observable_config,
    )
    if semantic_failure is not None:
        return semantic_failure
    seed_ids = [int(payload["seed"]) for payload in seed_payloads]
    seed_results = [
        payload["pure_walking"]["observable_results"][observable] for payload in seed_payloads
    ]
    invalid_schema_seeds = [
        {
            "seed": seed,
            "schema_status": result.get("schema_status"),
            "density_accounting_status": result.get("metadata", {}).get(
                "density_accounting_status"
            ),
        }
        for seed, result in zip(seed_ids, seed_results, strict=True)
        if result.get("schema_status") != SCHEMA_VALID
    ]
    if invalid_schema_seeds:
        return _failed_summary(
            plateau_status=PLATEAU_NO_BLOCKS,
            schema_status=SCHEMA_INVALID,
            genealogy_status=GENEALOGY_NOT_EVALUATED,
            diagnostics={
                "reason": "seed_schema_invalid",
                "seed_ids": seed_ids,
                "invalid_schema_seeds": invalid_schema_seeds,
            },
        )

    lag_values: dict[int, np.ndarray] = {}
    lag_support: dict[int, dict[str, Any]] = {}
    for lag in (value for value in observable_config.lag_steps if value > 0):
        values, support = _lag_values_and_support(
            seed_results,
            lag=lag,
            observable=observable,
            config=observable_config,
        )
        lag_support[lag] = support
        if values is not None:
            lag_values[lag] = values

    lag_zero_values = _complete_seed_values(seed_results, lag=0)
    if lag_zero_values is not None:
        lag_values[0] = lag_zero_values
    aggregate_values_by_lag = {
        lag: _lag_value(np.mean(values, axis=0), scalar=observable == "r2")
        for lag, values in lag_values.items()
    }
    aggregate_stderr_by_lag = {
        lag: _lag_value(_seed_sem(values), scalar=observable == "r2")
        for lag, values in lag_values.items()
    }

    supported_runs = _supported_prefix(
        tuple(lag for lag in observable_config.lag_steps if lag > 0),
        lag_support,
    )
    eligible_runs = [run for run in supported_runs if len(run) >= MIN_AGGREGATE_PLATEAU_LAGS]
    support_diagnostics = {
        "decision_level": "independent_seed_aggregate",
        "seed_ids": seed_ids,
        "seed_count": len(seed_ids),
        "lag_support": lag_support,
        "supported_runs": [list(run) for run in supported_runs],
        "minimum_aggregate_plateau_lag_count": MIN_AGGREGATE_PLATEAU_LAGS,
        "values_by_lag": aggregate_values_by_lag,
        "stderr_by_lag": aggregate_stderr_by_lag,
    }
    if not eligible_runs:
        plateau_status, genealogy_status = _support_failure_status(
            lag_values=lag_values,
            lag_support=lag_support,
        )
        return _failed_summary(
            plateau_status=plateau_status,
            schema_status=SCHEMA_VALID,
            genealogy_status=genealogy_status,
            diagnostics={
                **support_diagnostics,
                "reason": "insufficient_supported_prefix_for_plateau",
            },
        )

    selected_run = eligible_runs[-1]
    window_count = min(
        len(selected_run),
        max(MIN_AGGREGATE_PLATEAU_LAGS, observable_config.plateau_window_lag_count),
    )
    selected_lags = selected_run[-window_count:]
    excluded_later_lags = [lag for lag in observable_config.lag_steps if lag > selected_lags[-1]]
    excluded_unsupported_lags = [
        lag for lag in excluded_later_lags if not bool(lag_support[lag]["supported"])
    ]
    selected_values = np.stack([lag_values[lag] for lag in selected_lags], axis=1)
    plateau = _paired_plateau_summary(
        selected_values,
        selected_lags=selected_lags,
        observable=observable,
        config=observable_config,
    )
    selected_support = [lag_support[lag] for lag in selected_lags]
    diagnostics = {
        **support_diagnostics,
        **plateau["plateau_diagnostics"],
        "selected_run_lags": list(selected_run),
        "selected_window_lags": list(selected_lags),
        "excluded_later_lags": excluded_later_lags,
        "excluded_unsupported_lags": excluded_unsupported_lags,
        "selected_window_pooled_ancestor_ess_lower_min": min(
            float(item["pooled_ancestor_ess_lower_bound"]) for item in selected_support
        ),
        "selected_window_pooled_family_fraction_upper_max": max(
            float(item["pooled_family_fraction_upper_bound"]) for item in selected_support
        ),
    }
    return {
        **plateau,
        "schema_status": SCHEMA_VALID,
        "genealogy_status": GENEALOGY_SUPPORT_ACCEPTED,
        "plateau_diagnostics": diagnostics,
    }


def _preflight(
    seed_payloads: list[dict[str, Any]],
    *,
    observable: str,
) -> dict[str, Any] | None:
    if not seed_payloads:
        return _failed_summary(
            plateau_status=PLATEAU_NO_BLOCKS,
            schema_status=SCHEMA_INVALID,
            genealogy_status=GENEALOGY_NOT_EVALUATED,
            diagnostics={"reason": "no_seed_payloads"},
        )
    seed_ids = [payload.get("seed") for payload in seed_payloads]
    if any(not isinstance(seed, int) for seed in seed_ids) or len(set(seed_ids)) != len(seed_ids):
        return _failed_summary(
            plateau_status=PLATEAU_NO_BLOCKS,
            schema_status=SCHEMA_INVALID,
            genealogy_status=GENEALOGY_NOT_EVALUATED,
            diagnostics={"reason": "missing_or_duplicate_seed_ids", "seed_ids": seed_ids},
        )
    if len(seed_payloads) < 2:
        return _failed_summary(
            plateau_status=PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
            schema_status=SCHEMA_VALID,
            genealogy_status=GENEALOGY_NOT_EVALUATED,
            diagnostics={"reason": "fewer_than_two_independent_seeds", "seed_ids": seed_ids},
        )
    if any(
        observable not in payload.get("pure_walking", {}).get("observable_results", {})
        for payload in seed_payloads
    ):
        return _failed_summary(
            plateau_status=PLATEAU_NO_BLOCKS,
            schema_status=SCHEMA_INVALID,
            genealogy_status=GENEALOGY_NOT_EVALUATED,
            diagnostics={"reason": "missing_observable_seed_result", "seed_ids": seed_ids},
        )
    return None


def _semantic_preflight(
    seed_payloads: list[dict[str, Any]],
    *,
    observable: str,
    config: PureWalkingConfig,
) -> dict[str, Any] | None:
    mismatches: list[dict[str, Any]] = []
    for payload in seed_payloads:
        seed = int(payload["seed"])
        pure = payload["pure_walking"]
        pure_metadata = pure.get("metadata", {})
        result = pure["observable_results"][observable]
        result_metadata = result.get("metadata", {})
        reasons: list[str] = []
        result_lags = tuple(int(lag) for lag in result.get("lag_steps", ()))
        if result_lags != config.lag_steps:
            reasons.append("lag_steps_mismatch")
        if result.get("lag_unit") != config.lag_unit:
            reasons.append("lag_unit_mismatch")
        if pure_metadata.get("collection_mode", config.collection_mode) != config.collection_mode:
            reasons.append("collection_mode_mismatch")
        if observable == "density":
            density_stride = pure_metadata.get("density_collection_stride_steps")
            actual_stride = (
                pure_metadata.get("collection_stride_steps", config.collection_stride_steps)
                if density_stride is None
                else density_stride
            )
        else:
            actual_stride = pure_metadata.get(
                "collection_stride_steps",
                config.collection_stride_steps,
            )
        if actual_stride != config.collection_stride_steps:
            reasons.append("collection_stride_mismatch")
        if observable == "r2":
            source = result_metadata.get(
                "observable_source",
                pure_metadata.get("observable_source", "raw_r2"),
            )
            if source != config.observable_source:
                reasons.append("r2_source_mismatch")
            if config.observable_source == "r2_rb" and not _optional_float_matches(
                result_metadata.get(
                    "r2_rb_com_variance",
                    pure_metadata.get("r2_rb_com_variance"),
                ),
                config.r2_rb_com_variance,
            ):
                reasons.append("r2_com_variance_mismatch")
        elif observable == "density":
            source = result_metadata.get(
                "density_source",
                pure_metadata.get("density_source", "raw_density"),
            )
            if source != config.density_source:
                reasons.append("density_source_mismatch")
            parity = result_metadata.get(
                "density_parity_average",
                pure_metadata.get("density_parity_average", False),
            )
            if parity is not config.density_parity_average:
                reasons.append("density_parity_mismatch")
            if config.density_source == "com_rao_blackwell" and not _optional_float_matches(
                result_metadata.get(
                    "density_com_variance",
                    pure_metadata.get("density_com_variance"),
                ),
                config.density_com_variance,
            ):
                reasons.append("density_com_variance_mismatch")
            if not _array_matches(result_metadata.get("bin_edges"), config.density_bin_edges):
                reasons.append("density_bin_edges_mismatch")
            if config.density_expected_particles is not None and not _optional_float_matches(
                result_metadata.get("density_expected_particles"),
                config.density_expected_particles,
            ):
                reasons.append("density_expected_particles_mismatch")
            if config.density_expected_particles is not None and not _optional_float_matches(
                result_metadata.get("density_accounting_abs_tolerance"),
                config.density_accounting_abs_tolerance,
            ):
                reasons.append("density_accounting_tolerance_mismatch")
        elif observable == "pair_distance_density":
            if not _array_matches(result_metadata.get("bin_edges"), config.pair_bin_edges):
                reasons.append("pair_bin_edges_mismatch")
        elif observable == "structure_factor" and not _array_matches(
            result_metadata.get("k_values"),
            config.structure_k_values,
        ):
            reasons.append("structure_k_values_mismatch")
        if reasons:
            mismatches.append({"seed": seed, "reasons": reasons})
    if not mismatches:
        return None
    return _failed_summary(
        plateau_status=PLATEAU_NO_BLOCKS,
        schema_status=SCHEMA_INVALID,
        genealogy_status=GENEALOGY_NOT_EVALUATED,
        diagnostics={"reason": "incompatible_seed_estimator_semantics", "mismatches": mismatches},
    )


def _lag_values_and_support(
    seed_results: list[dict[str, Any]],
    *,
    lag: int,
    observable: str,
    config: PureWalkingConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    reasons: list[str] = []
    values: list[np.ndarray] = []
    block_counts: list[int] = []
    weight_ess_values: list[float] = []
    ancestor_ess_values: list[float] = []
    family_fraction_values: list[float] = []
    expected_shape: tuple[int, ...] | None = None
    for result in seed_results:
        value = _lag_dict_value(result.get("values_by_lag", {}), lag)
        if value is None or not np.all(np.isfinite(value)):
            reasons.append("missing_or_nonfinite_value")
        elif expected_shape is not None and value.shape != expected_shape:
            reasons.append("inconsistent_value_shape")
        else:
            expected_shape = value.shape
            values.append(value)
        block_count = _lag_dict_int(result.get("block_count_by_lag", {}), lag)
        if block_count is None:
            reasons.append("missing_block_count")
        else:
            block_counts.append(block_count)
        weight_ess_values.append(
            _lag_dict_float(result.get("block_weight_ess_min_by_lag", {}), lag)
        )
        ancestor_ess_values.append(
            _lag_dict_float(result.get("block_source_ancestor_ess_min_by_lag", {}), lag)
        )
        family_fraction_values.append(
            _lag_dict_float(result.get("block_max_source_family_fraction_by_lag", {}), lag)
        )
    seed_count = len(seed_results)
    if len(values) != seed_count:
        reasons.append("incomplete_seed_values")
    density_integrals: list[float] = []
    density_accounting_errors: list[float] = []
    if observable == "density" and config.density_expected_particles is not None:
        widths = np.diff(np.asarray(config.density_bin_edges, dtype=float))
        for value in values:
            if value.shape != widths.shape:
                reasons.append("density_value_shape_mismatch")
                continue
            integral = float(np.sum(value * widths))
            error = abs(integral - config.density_expected_particles)
            density_integrals.append(integral)
            density_accounting_errors.append(error)
            if not np.isfinite(error) or error > config.density_accounting_abs_tolerance:
                reasons.append("density_particle_count_mismatch")
    if len(block_counts) != seed_count or min(block_counts, default=0) < config.min_block_count:
        reasons.append("block_count_below_minimum")
    if not _all_finite_positive(weight_ess_values) or min(weight_ess_values, default=0.0) < (
        config.min_walker_weight_ess
    ):
        reasons.append("walker_weight_ess_below_minimum")
    if not _all_finite_positive(ancestor_ess_values):
        reasons.append("invalid_ancestor_ess")
        pooled_ancestor_ess = 0.0
    else:
        pooled_ancestor_ess = float(
            seed_count * seed_count / np.sum(1.0 / np.asarray(ancestor_ess_values))
        )
        if pooled_ancestor_ess < config.min_source_ancestor_ess:
            reasons.append("pooled_ancestor_ess_below_minimum")
    if not _all_finite_fraction(family_fraction_values):
        reasons.append("invalid_family_fraction")
        pooled_family_fraction = 1.0
    else:
        pooled_family_fraction = float(max(family_fraction_values) / seed_count)
        if pooled_family_fraction > config.max_source_family_fraction:
            reasons.append("pooled_family_fraction_above_maximum")
    diagnostics = {
        "supported": not reasons,
        "reasons": sorted(set(reasons)),
        "seed_count": seed_count,
        "min_block_count": min(block_counts, default=0),
        "min_walker_weight_ess": min(weight_ess_values, default=0.0),
        "per_seed_ancestor_ess_min": ancestor_ess_values,
        "pooled_ancestor_ess_lower_bound": pooled_ancestor_ess,
        "required_pooled_ancestor_ess": config.min_source_ancestor_ess,
        "per_seed_family_fraction_max": family_fraction_values,
        "pooled_family_fraction_upper_bound": pooled_family_fraction,
        "maximum_pooled_family_fraction": config.max_source_family_fraction,
    }
    if observable == "density" and config.density_expected_particles is not None:
        diagnostics.update(
            {
                "density_expected_particles": config.density_expected_particles,
                "density_accounting_abs_tolerance": (config.density_accounting_abs_tolerance),
                "density_integrals_by_seed": density_integrals,
                "density_accounting_abs_errors_by_seed": density_accounting_errors,
                "density_accounting_abs_error_max": max(
                    density_accounting_errors,
                    default=float("inf"),
                ),
            }
        )
    stacked_values = np.stack(values, axis=0) if len(values) == seed_count else None
    return stacked_values, diagnostics


def _complete_seed_values(
    seed_results: list[dict[str, Any]],
    *,
    lag: int,
) -> np.ndarray | None:
    values = [_lag_dict_value(result.get("values_by_lag", {}), lag) for result in seed_results]
    if any(value is None or not np.all(np.isfinite(value)) for value in values):
        return None
    arrays = [np.asarray(value, dtype=float) for value in values]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        return None
    return np.stack(arrays, axis=0)


def _supported_prefix(
    positive_lags: tuple[int, ...],
    support: dict[int, dict[str, Any]],
) -> list[tuple[int, ...]]:
    active: list[int] = []
    for lag in positive_lags:
        if bool(support.get(lag, {}).get("supported")):
            active.append(lag)
        else:
            break
    return [tuple(active)] if active else []


def _paired_plateau_summary(
    values: np.ndarray,
    *,
    selected_lags: tuple[int, ...],
    observable: str,
    config: PureWalkingConfig,
) -> dict[str, Any]:
    if len(selected_lags) < MIN_AGGREGATE_PLATEAU_LAGS:
        raise ValueError("aggregate FW plateau requires at least three supported lags")

    seed_plateaus = np.mean(values, axis=1)
    plateau = np.mean(seed_plateaus, axis=0)
    plateau_stderr = _seed_sem(seed_plateaus)
    deltas: dict[int, float] = {}
    uncertainties: dict[int, float] = {}
    thresholds: dict[int, float] = {}
    spread_pass = True
    for index, lag in enumerate(selected_lags):
        residuals = values[:, index, :] - seed_plateaus
        delta, uncertainty = _paired_norms(
            residuals,
            scale=plateau,
            observable=observable,
            config=config,
        )
        threshold = _threshold(config, observable=observable, uncertainty=uncertainty)
        deltas[lag] = delta
        uncertainties[lag] = uncertainty
        thresholds[lag] = threshold
        spread_pass = spread_pass and _finite_stability_pass(delta, uncertainty, threshold)
    slope_changes = _seed_slope_changes(values, selected_lags=selected_lags)
    slope_delta, slope_uncertainty = _paired_norms(
        slope_changes,
        scale=plateau,
        observable=observable,
        config=config,
    )
    slope_threshold = _threshold(
        config,
        observable=observable,
        uncertainty=slope_uncertainty,
    )
    slope_pass = _finite_stability_pass(slope_delta, slope_uncertainty, slope_threshold)
    status = PLATEAU_RESOLVED if spread_pass and slope_pass else PLATEAU_UNRESOLVED
    return {
        "plateau_status": status,
        "plateau_value": _lag_value(plateau, scalar=observable == "r2"),
        "plateau_stderr": _lag_value(plateau_stderr, scalar=observable == "r2"),
        "bias_bracket": (
            _lag_value(
                np.mean(values[:, 0, :], axis=0),
                scalar=observable == "r2",
            ),
            _lag_value(
                np.mean(values[:, -1, :], axis=0),
                scalar=observable == "r2",
            ),
        ),
        "plateau_diagnostics": {
            "method": "paired_independent_seed_window",
            "window_lags": list(selected_lags),
            "paired_delta_by_lag": deltas,
            "paired_stderr_by_lag": uncertainties,
            "paired_threshold_by_lag": thresholds,
            "max_window_delta": max(deltas.values()),
            "min_window_threshold": min(thresholds.values()),
            "window_spread_pass": spread_pass,
            "slope_delta": slope_delta,
            "slope_paired_stderr": slope_uncertainty,
            "slope_threshold": slope_threshold,
            "slope_pass": slope_pass,
            "decision": status,
        },
    }


def _seed_slope_changes(values: np.ndarray, *, selected_lags: tuple[int, ...]) -> np.ndarray:
    lags = np.asarray(selected_lags, dtype=float)
    centered = lags - float(np.mean(lags))
    denominator = float(np.sum(centered * centered))
    slopes = np.sum(values * centered[np.newaxis, :, np.newaxis], axis=1) / denominator
    return slopes * float(lags[-1] - lags[0])


def _paired_norms(
    seed_differences: np.ndarray,
    *,
    scale: np.ndarray,
    observable: str,
    config: PureWalkingConfig,
) -> tuple[float, float]:
    mean_difference = np.mean(seed_differences, axis=0)
    sem = _seed_sem(seed_differences)
    return (
        _observable_norm(mean_difference, scale=scale, observable=observable, config=config),
        _observable_norm(sem, scale=scale, observable=observable, config=config),
    )


def _observable_norm(
    value: np.ndarray,
    *,
    scale: np.ndarray,
    observable: str,
    config: PureWalkingConfig,
) -> float:
    if observable != "density":
        result = float(np.max(np.abs(value)))
        return result if np.isfinite(result) else float("inf")
    edges = np.asarray(config.density_bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size != scale.size + 1 or not np.all(np.isfinite(edges)):
        return float("inf")
    widths = np.diff(edges)
    denominator = float(np.sqrt(np.sum(scale * scale * widths)))
    if (
        value.shape != scale.shape
        or widths.shape != scale.shape
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(widths))
        or np.any(widths <= 0.0)
        or denominator <= 0.0
        or not np.isfinite(denominator)
    ):
        return float("inf")
    return float(np.sqrt(np.sum(value * value * widths)) / denominator)


def _threshold(config: PureWalkingConfig, *, observable: str, uncertainty: float) -> float:
    base = (
        config.density_plateau_relative_l2_tolerance
        if observable == "density"
        else config.plateau_abs_tolerance
    )
    return float(base + config.plateau_sigma_threshold * uncertainty)


def _finite_stability_pass(delta: float, uncertainty: float, threshold: float) -> bool:
    return bool(
        np.isfinite(delta)
        and np.isfinite(uncertainty)
        and np.isfinite(threshold)
        and delta <= threshold
    )


def _seed_sem(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _support_failure_status(
    *,
    lag_values: dict[int, np.ndarray],
    lag_support: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    positive_values = {lag for lag in lag_values if lag > 0}
    if not positive_values:
        return PLATEAU_NO_BLOCKS, GENEALOGY_NOT_EVALUATED
    reason_sets = [set(item.get("reasons", [])) for item in lag_support.values()]
    nonempty_reasons = [reasons for reasons in reason_sets if reasons]
    if any("density_particle_count_mismatch" in reasons for reasons in nonempty_reasons):
        return PLATEAU_UNRESOLVED, GENEALOGY_NOT_EVALUATED
    if nonempty_reasons and all(
        reasons <= {"block_count_below_minimum"} for reasons in nonempty_reasons
    ):
        return PLATEAU_INSUFFICIENT_BLOCKS, GENEALOGY_NOT_EVALUATED
    genealogy_reasons = {
        "invalid_ancestor_ess",
        "pooled_ancestor_ess_below_minimum",
        "invalid_family_fraction",
        "pooled_family_fraction_above_maximum",
    }
    genealogy_only = [
        reasons for reasons in nonempty_reasons if reasons and reasons <= genealogy_reasons
    ]
    if genealogy_only:
        if any("pooled_family_fraction_above_maximum" in reasons for reasons in genealogy_only):
            genealogy_status = GENEALOGY_SOURCE_FAMILY_DOMINANCE
        else:
            genealogy_status = GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM
    else:
        genealogy_status = GENEALOGY_NOT_EVALUATED
    return PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM, genealogy_status


def _optional_float_matches(actual: object, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, (int, float, np.integer, np.floating)):
        return False
    value = float(actual)
    return bool(np.isfinite(value) and np.isclose(value, expected, rtol=0.0, atol=1.0e-12))


def _array_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    try:
        actual_array = np.asarray(actual, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        actual_array.shape == expected_array.shape
        and np.all(np.isfinite(actual_array))
        and np.all(np.isfinite(expected_array))
        and np.allclose(actual_array, expected_array, rtol=0.0, atol=1.0e-12)
    )


def _failed_summary(
    *,
    plateau_status: str,
    schema_status: str,
    genealogy_status: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "plateau_status": plateau_status,
        "plateau_value": None,
        "plateau_stderr": None,
        "bias_bracket": None,
        "schema_status": schema_status,
        "genealogy_status": genealogy_status,
        "plateau_diagnostics": diagnostics,
    }


def _lag_dict_value(values: Any, lag: int) -> np.ndarray | None:
    if not isinstance(values, dict):
        return None
    value = values.get(lag, values.get(str(lag)))
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _lag_dict_float(values: Any, lag: int) -> float:
    value = _lag_dict_value(values, lag)
    if value is None or value.size != 1:
        return float("nan")
    return float(value[0])


def _lag_dict_int(values: Any, lag: int) -> int | None:
    value = _lag_dict_float(values, lag)
    if not np.isfinite(value) or not float(value).is_integer():
        return None
    return int(value)


def _all_finite_positive(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(array.size > 0 and np.all(np.isfinite(array)) and np.all(array > 0.0))


def _all_finite_fraction(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(
        array.size > 0
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
        and np.all(array <= 1.0)
    )


def _lag_value(value: np.ndarray, *, scalar: bool) -> LagValue:
    return float(value[0]) if scalar and value.shape == (1,) else value.copy()
