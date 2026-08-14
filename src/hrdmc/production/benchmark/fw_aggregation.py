from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.forward_walking.results import (
    GENEALOGY_NOT_EVALUATED,
    GENEALOGY_SUPPORT_ACCEPTED,
    PLATEAU_NO_BLOCKS,
    PLATEAU_RESOLVED,
    PLATEAU_UNRESOLVED,
    SCHEMA_INVALID,
    SCHEMA_VALID,
    LagValue,
)
from hrdmc.production.benchmark.support import (
    failed_summary as _failed_summary,
)
from hrdmc.production.benchmark.support import (
    lag_dict_value as _lag_dict_value,
)
from hrdmc.production.benchmark.support import (
    lag_values_and_support as _lag_values_and_support,
)
from hrdmc.production.benchmark.support import (
    preflight as _preflight,
)
from hrdmc.production.benchmark.support import (
    semantic_preflight as _semantic_preflight,
)
from hrdmc.production.benchmark.support import (
    support_failure_status as _support_failure_status,
)
from hrdmc.statistics.equivalence import (
    simultaneous_pairwise_equivalence,
    simultaneous_pairwise_norm_equivalence,
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
    if observable == "r2":
        plateau = _paired_rms_equivalence_summary(
            selected_values,
            selected_lags=selected_lags,
            config=observable_config,
        )
    elif observable == "density":
        plateau = _paired_density_equivalence_summary(
            selected_values,
            selected_lags=selected_lags,
            config=observable_config,
        )
    else:  # PureWalkingConfig rejects every other observable.
        raise ValueError(f"unsupported forward-walking observable: {observable}")
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


def _paired_rms_equivalence_summary(
    values: np.ndarray,
    *,
    selected_lags: tuple[int, ...],
    config: PureWalkingConfig,
) -> dict[str, Any]:
    if len(selected_lags) < MIN_AGGREGATE_PLATEAU_LAGS:
        raise ValueError("aggregate FW plateau requires at least three supported lags")
    if values.ndim != 3 or values.shape[2] != 1:
        raise ValueError("R2 aggregate values must contain one scalar per seed and lag")
    scalar_r2 = values[:, :, 0]
    seed_plateaus = np.mean(values, axis=1)
    plateau = np.mean(seed_plateaus, axis=0)
    plateau_stderr = _seed_sem(seed_plateaus)
    if not np.all(np.isfinite(scalar_r2)) or np.any(scalar_r2 <= 0.0):
        return {
            "plateau_status": PLATEAU_UNRESOLVED,
            "plateau_value": _lag_value(plateau, scalar=True),
            "plateau_stderr": _lag_value(plateau_stderr, scalar=True),
            "bias_bracket": (
                _lag_value(np.mean(values[:, 0, :], axis=0), scalar=True),
                _lag_value(np.mean(values[:, -1, :], axis=0), scalar=True),
            ),
            "plateau_diagnostics": {
                "method": "paired_seed_simultaneous_rms_equivalence",
                "window_lags": list(selected_lags),
                "reason": "nonpositive_or_nonfinite_r2",
                "decision": PLATEAU_UNRESOLVED,
            },
        }
    rms_values = np.sqrt(scalar_r2)
    rms_scale = float(np.sqrt(np.mean(scalar_r2)))
    relative_margin = float(config.rms_plateau_relative_tolerance)
    absolute_margin = float(relative_margin * rms_scale)
    equivalence = simultaneous_pairwise_equivalence(
        rms_values,
        equivalence_margin=absolute_margin,
        confidence_level=config.plateau_equivalence_confidence_level,
    )
    relative_upper_bound = float(equivalence.simultaneous_upper_bound / rms_scale)
    relative_observed_difference = float(equivalence.observed_max_difference / rms_scale)
    status = PLATEAU_RESOLVED if equivalence.equivalent else PLATEAU_UNRESOLVED
    return {
        "plateau_status": status,
        "plateau_value": _lag_value(plateau, scalar=True),
        "plateau_stderr": _lag_value(plateau_stderr, scalar=True),
        "bias_bracket": (
            _lag_value(np.mean(values[:, 0, :], axis=0), scalar=True),
            _lag_value(np.mean(values[:, -1, :], axis=0), scalar=True),
        ),
        "plateau_diagnostics": {
            "method": "paired_seed_simultaneous_rms_equivalence",
            "familywise_method": "bonferroni_paired_student_t",
            "window_lags": list(selected_lags),
            "rms_scale": rms_scale,
            "rms_relative_equivalence_margin": relative_margin,
            "rms_absolute_equivalence_margin": absolute_margin,
            "confidence_level": equivalence.confidence_level,
            "pair_count": equivalence.pair_count,
            "critical_value": equivalence.critical_value,
            "observed_max_rms_pairwise_difference": equivalence.observed_max_difference,
            "observed_max_rms_pairwise_relative_difference": (relative_observed_difference),
            "simultaneous_rms_upper_bound": equivalence.simultaneous_upper_bound,
            "simultaneous_rms_relative_upper_bound": relative_upper_bound,
            "equivalence_pass": equivalence.equivalent,
            "pairwise_bounds": [
                {
                    **bound.to_dict(),
                    "first_lag": selected_lags[bound.first_index],
                    "second_lag": selected_lags[bound.second_index],
                }
                for bound in equivalence.pairwise_bounds
            ],
            "decision": status,
        },
    }


def _paired_density_equivalence_summary(
    values: np.ndarray,
    *,
    selected_lags: tuple[int, ...],
    config: PureWalkingConfig,
) -> dict[str, Any]:
    if len(selected_lags) < MIN_AGGREGATE_PLATEAU_LAGS:
        raise ValueError("aggregate FW plateau requires at least three supported lags")
    if values.ndim != 3:
        raise ValueError("density aggregate values must contain one vector per seed and lag")
    edges = np.asarray(config.density_bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size != values.shape[2] + 1:
        raise ValueError("density equivalence requires matching bin edges")
    widths = np.diff(edges)
    seed_plateaus = np.mean(values, axis=1)
    plateau = np.mean(seed_plateaus, axis=0)
    plateau_stderr = _seed_sem(seed_plateaus)
    equivalence = simultaneous_pairwise_norm_equivalence(
        values,
        feature_weights=widths,
        scale_vector=plateau,
        equivalence_margin=config.density_plateau_relative_l2_tolerance,
        confidence_level=config.plateau_equivalence_confidence_level,
    )
    status = PLATEAU_RESOLVED if equivalence.equivalent else PLATEAU_UNRESOLVED
    return {
        "plateau_status": status,
        "plateau_value": _lag_value(plateau, scalar=False),
        "plateau_stderr": _lag_value(plateau_stderr, scalar=False),
        "bias_bracket": (
            _lag_value(np.mean(values[:, 0, :], axis=0), scalar=False),
            _lag_value(np.mean(values[:, -1, :], axis=0), scalar=False),
        ),
        "plateau_diagnostics": {
            "method": "paired_seed_simultaneous_density_l2_equivalence",
            "familywise_method": "bonferroni_paired_student_t",
            "window_lags": list(selected_lags),
            "density_relative_l2_equivalence_margin": (
                config.density_plateau_relative_l2_tolerance
            ),
            "confidence_level": equivalence.confidence_level,
            "pair_count": equivalence.pair_count,
            "critical_value": equivalence.critical_value,
            "observed_max_density_relative_l2": (equivalence.observed_max_relative_norm),
            "simultaneous_density_relative_l2_upper_bound": (equivalence.simultaneous_upper_bound),
            "equivalence_pass": equivalence.equivalent,
            "pairwise_bounds": [
                {
                    **bound.to_dict(),
                    "first_lag": selected_lags[bound.first_index],
                    "second_lag": selected_lags[bound.second_index],
                }
                for bound in equivalence.pairwise_bounds
            ],
            "decision": status,
        },
    }


def _seed_sem(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _lag_value(value: np.ndarray, *, scalar: bool) -> LagValue:
    return float(value[0]) if scalar and value.shape == (1,) else value.copy()
