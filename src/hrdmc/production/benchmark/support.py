from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.forward_walking.results import (
    GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    GENEALOGY_NOT_EVALUATED,
    GENEALOGY_SOURCE_FAMILY_DOMINANCE,
    PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    PLATEAU_INSUFFICIENT_BLOCKS,
    PLATEAU_NO_BLOCKS,
    PLATEAU_UNRESOLVED,
    SCHEMA_INVALID,
    SCHEMA_VALID,
)


@dataclass
class _LagInputs:
    reasons: list[str]
    values: list[np.ndarray]
    block_counts: list[int]
    weight_ess: list[float]
    ancestor_ess: list[float]
    family_fraction: list[float]


def preflight(seed_payloads: list[dict[str, Any]], *, observable: str) -> dict[str, Any] | None:
    if not seed_payloads:
        return failed_summary(
            PLATEAU_NO_BLOCKS,
            SCHEMA_INVALID,
            GENEALOGY_NOT_EVALUATED,
            {"reason": "no_seed_payloads"},
        )
    seed_ids = [payload.get("seed") for payload in seed_payloads]
    if any(not isinstance(seed, int) for seed in seed_ids) or len(set(seed_ids)) != len(seed_ids):
        return failed_summary(
            PLATEAU_NO_BLOCKS,
            SCHEMA_INVALID,
            GENEALOGY_NOT_EVALUATED,
            {"reason": "missing_or_duplicate_seed_ids", "seed_ids": seed_ids},
        )
    if len(seed_payloads) < 2:
        return failed_summary(
            PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
            SCHEMA_VALID,
            GENEALOGY_NOT_EVALUATED,
            {"reason": "fewer_than_two_independent_seeds", "seed_ids": seed_ids},
        )
    missing = any(
        observable not in payload.get("pure_walking", {}).get("observable_results", {})
        for payload in seed_payloads
    )
    if missing:
        return failed_summary(
            PLATEAU_NO_BLOCKS,
            SCHEMA_INVALID,
            GENEALOGY_NOT_EVALUATED,
            {"reason": "missing_observable_seed_result", "seed_ids": seed_ids},
        )
    return None


def semantic_preflight(
    seed_payloads: list[dict[str, Any]], *, observable: str, config: PureWalkingConfig
) -> dict[str, Any] | None:
    mismatches = [
        {"seed": int(payload["seed"]), "reasons": reasons}
        for payload in seed_payloads
        if (reasons := _semantic_reasons(payload, observable, config))
    ]
    if not mismatches:
        return None
    return failed_summary(
        PLATEAU_NO_BLOCKS,
        SCHEMA_INVALID,
        GENEALOGY_NOT_EVALUATED,
        {"reason": "incompatible_seed_estimator_semantics", "mismatches": mismatches},
    )


def lag_values_and_support(
    seed_results: list[dict[str, Any]],
    *,
    lag: int,
    observable: str,
    config: PureWalkingConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    inputs = _collect_lag_inputs(seed_results, lag)
    seed_count = len(seed_results)
    if len(inputs.values) != seed_count:
        inputs.reasons.append("incomplete_seed_values")
    density = _density_accounting(inputs.values, observable, config)
    inputs.reasons.extend(density.pop("reasons"))
    support = _support_metrics(inputs, seed_count, config)
    diagnostics = {
        "supported": not inputs.reasons,
        "reasons": sorted(set(inputs.reasons)),
        "seed_count": seed_count,
        "min_block_count": min(inputs.block_counts, default=0),
        "min_walker_weight_ess": min(inputs.weight_ess, default=0.0),
        "per_seed_ancestor_ess_min": inputs.ancestor_ess,
        "per_seed_family_fraction_max": inputs.family_fraction,
        "required_pooled_ancestor_ess": config.min_source_ancestor_ess,
        "maximum_pooled_family_fraction": config.max_source_family_fraction,
        **support,
        **density,
    }
    stacked = np.stack(inputs.values, axis=0) if len(inputs.values) == seed_count else None
    return stacked, diagnostics


def support_failure_status(
    *, lag_values: dict[int, np.ndarray], lag_support: dict[int, dict[str, Any]]
) -> tuple[str, str]:
    if not {lag for lag in lag_values if lag > 0}:
        return PLATEAU_NO_BLOCKS, GENEALOGY_NOT_EVALUATED
    reason_sets = [set(item.get("reasons", [])) for item in lag_support.values()]
    nonempty = [reasons for reasons in reason_sets if reasons]
    if any("density_particle_count_mismatch" in reasons for reasons in nonempty):
        return PLATEAU_UNRESOLVED, GENEALOGY_NOT_EVALUATED
    if nonempty and all(reasons <= {"block_count_below_minimum"} for reasons in nonempty):
        return PLATEAU_INSUFFICIENT_BLOCKS, GENEALOGY_NOT_EVALUATED
    genealogy = {
        "invalid_ancestor_ess",
        "pooled_ancestor_ess_below_minimum",
        "invalid_family_fraction",
        "pooled_family_fraction_above_maximum",
    }
    genealogy_only = [reasons for reasons in nonempty if reasons and reasons <= genealogy]
    status = GENEALOGY_NOT_EVALUATED
    if genealogy_only:
        status = (
            GENEALOGY_SOURCE_FAMILY_DOMINANCE
            if any("pooled_family_fraction_above_maximum" in row for row in genealogy_only)
            else GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM
        )
    return PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM, status


def failed_summary(
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


def lag_dict_value(values: Any, lag: int) -> np.ndarray | None:
    if not isinstance(values, dict):
        return None
    value = values.get(lag, values.get(str(lag)))
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _semantic_reasons(
    payload: dict[str, Any], observable: str, config: PureWalkingConfig
) -> list[str]:
    pure = payload["pure_walking"]
    pure_metadata = pure.get("metadata", {})
    result = pure["observable_results"][observable]
    result_metadata = result.get("metadata", {})
    checks = (
        (
            tuple(int(lag) for lag in result.get("lag_steps", ())) == config.lag_steps,
            "lag_steps_mismatch",
        ),
        (result.get("lag_unit") == config.lag_unit, "lag_unit_mismatch"),
        (
            pure_metadata.get("collection_mode", config.collection_mode) == config.collection_mode,
            "collection_mode_mismatch",
        ),
        (
            _collection_stride(pure_metadata, observable, config) == config.collection_stride_steps,
            "collection_stride_mismatch",
        ),
    )
    reasons = [message for accepted, message in checks if not accepted]
    reasons.extend(
        _r2_semantic_reasons(pure_metadata, result_metadata, config)
        if observable == "r2"
        else _density_semantic_reasons(pure_metadata, result_metadata, config)
    )
    return reasons


def _collection_stride(metadata: dict[str, Any], observable: str, config: PureWalkingConfig) -> Any:
    density_stride = metadata.get("density_collection_stride_steps")
    if observable == "density" and density_stride is not None:
        return density_stride
    return metadata.get("collection_stride_steps", config.collection_stride_steps)


def _r2_semantic_reasons(
    pure: dict[str, Any], result: dict[str, Any], config: PureWalkingConfig
) -> list[str]:
    source = result.get("observable_source", pure.get("observable_source", "raw_r2"))
    reasons = [] if source == config.observable_source else ["r2_source_mismatch"]
    variance = result.get("r2_rb_com_variance", pure.get("r2_rb_com_variance"))
    if config.observable_source == "r2_rb" and not _optional_float_matches(
        variance, config.r2_rb_com_variance
    ):
        reasons.append("r2_com_variance_mismatch")
    return reasons


def _density_semantic_reasons(
    pure: dict[str, Any], result: dict[str, Any], config: PureWalkingConfig
) -> list[str]:
    checks = [
        (
            result.get("density_source", pure.get("density_source", "raw_density"))
            == config.density_source,
            "density_source_mismatch",
        ),
        (
            result.get("density_parity_average", pure.get("density_parity_average", False))
            is config.density_parity_average,
            "density_parity_mismatch",
        ),
        (
            _array_matches(result.get("bin_edges"), config.density_bin_edges),
            "density_bin_edges_mismatch",
        ),
    ]
    variance = result.get("density_com_variance", pure.get("density_com_variance"))
    if config.density_source == "com_rao_blackwell":
        checks.append(
            (
                _optional_float_matches(variance, config.density_com_variance),
                "density_com_variance_mismatch",
            )
        )
    if config.density_expected_particles is not None:
        checks.extend(
            (
                (
                    _optional_float_matches(
                        result.get("density_expected_particles"), config.density_expected_particles
                    ),
                    "density_expected_particles_mismatch",
                ),
                (
                    _optional_float_matches(
                        result.get("density_accounting_abs_tolerance"),
                        config.density_accounting_abs_tolerance,
                    ),
                    "density_accounting_tolerance_mismatch",
                ),
            )
        )
    return [message for accepted, message in checks if not accepted]


def _collect_lag_inputs(seed_results: list[dict[str, Any]], lag: int) -> _LagInputs:
    result = _LagInputs([], [], [], [], [], [])
    expected_shape = None
    for seed in seed_results:
        value = lag_dict_value(seed.get("values_by_lag", {}), lag)
        if value is None or not np.all(np.isfinite(value)):
            result.reasons.append("missing_or_nonfinite_value")
        elif expected_shape is not None and value.shape != expected_shape:
            result.reasons.append("inconsistent_value_shape")
        else:
            expected_shape = value.shape
            result.values.append(value)
        block_count = _lag_dict_int(seed.get("block_count_by_lag", {}), lag)
        if block_count is None:
            result.reasons.append("missing_block_count")
        else:
            result.block_counts.append(block_count)
        result.weight_ess.append(_lag_dict_float(seed.get("block_weight_ess_min_by_lag", {}), lag))
        result.ancestor_ess.append(
            _lag_dict_float(seed.get("block_source_ancestor_ess_min_by_lag", {}), lag)
        )
        result.family_fraction.append(
            _lag_dict_float(seed.get("block_max_source_family_fraction_by_lag", {}), lag)
        )
    return result


def _density_accounting(
    values: list[np.ndarray], observable: str, config: PureWalkingConfig
) -> dict[str, Any]:
    if observable != "density" or config.density_expected_particles is None:
        return {"reasons": []}
    widths = np.diff(np.asarray(config.density_bin_edges, dtype=float))
    reasons, integrals, errors = [], [], []
    for value in values:
        if value.shape != widths.shape:
            reasons.append("density_value_shape_mismatch")
            continue
        integral = float(np.sum(value * widths))
        error = abs(integral - config.density_expected_particles)
        integrals.append(integral)
        errors.append(error)
        if not np.isfinite(error) or error > config.density_accounting_abs_tolerance:
            reasons.append("density_particle_count_mismatch")
    return {
        "reasons": reasons,
        "density_expected_particles": config.density_expected_particles,
        "density_accounting_abs_tolerance": config.density_accounting_abs_tolerance,
        "density_integrals_by_seed": integrals,
        "density_accounting_abs_errors_by_seed": errors,
        "density_accounting_abs_error_max": max(errors, default=float("inf")),
    }


def _support_metrics(
    inputs: _LagInputs, seed_count: int, config: PureWalkingConfig
) -> dict[str, float]:
    if (
        len(inputs.block_counts) != seed_count
        or min(inputs.block_counts, default=0) < config.min_block_count
    ):
        inputs.reasons.append("block_count_below_minimum")
    if (
        not _all_finite_positive(inputs.weight_ess)
        or min(inputs.weight_ess, default=0.0) < config.min_walker_weight_ess
    ):
        inputs.reasons.append("walker_weight_ess_below_minimum")
    ancestor = _pooled_ancestor_ess(inputs, seed_count, config)
    family = _pooled_family_fraction(inputs, seed_count, config)
    return {
        "pooled_ancestor_ess_lower_bound": ancestor,
        "pooled_family_fraction_upper_bound": family,
    }


def _pooled_ancestor_ess(inputs: _LagInputs, seed_count: int, config: PureWalkingConfig) -> float:
    if not _all_finite_positive(inputs.ancestor_ess):
        inputs.reasons.append("invalid_ancestor_ess")
        return 0.0
    value = float(seed_count * seed_count / np.sum(1.0 / np.asarray(inputs.ancestor_ess)))
    if value < config.min_source_ancestor_ess:
        inputs.reasons.append("pooled_ancestor_ess_below_minimum")
    return value


def _pooled_family_fraction(
    inputs: _LagInputs, seed_count: int, config: PureWalkingConfig
) -> float:
    if not _all_finite_fraction(inputs.family_fraction):
        inputs.reasons.append("invalid_family_fraction")
        return 1.0
    value = float(max(inputs.family_fraction) / seed_count)
    if value > config.max_source_family_fraction:
        inputs.reasons.append("pooled_family_fraction_above_maximum")
    return value


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


def _lag_dict_float(values: Any, lag: int) -> float:
    value = lag_dict_value(values, lag)
    return float("nan") if value is None or value.size != 1 else float(value[0])


def _lag_dict_int(values: Any, lag: int) -> int | None:
    value = _lag_dict_float(values, lag)
    return int(value) if np.isfinite(value) and float(value).is_integer() else None


def _all_finite_positive(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(array.size and np.all(np.isfinite(array)) and np.all(array > 0.0))


def _all_finite_fraction(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(
        array.size and np.all(np.isfinite(array)) and np.all(array >= 0.0) and np.all(array <= 1.0)
    )
