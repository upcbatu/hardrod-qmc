from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import norm

ENERGY_DIRECTIONAL_STATISTICS = (
    "slope_z_autocorr_adjusted",
    "first_last_quarter_z",
    "late_cumulative_z",
)


def assess_matrix_energy_stationarity(
    stationarity_by_case: Mapping[str, Mapping[str, Any]],
    *,
    confidence_level: float = 0.95,
    rhat_limit: float = 1.01,
    min_effective_samples: float = 400.0,
) -> dict[str, Any]:
    """Apply one simultaneous energy-stationarity screen to a case matrix.

    The stored first/last-block statistic is not counted separately because the
    current four-block construction makes it identical to the first/last-quarter
    statistic. Raw source classifications remain part of each case result.
    """

    _validate_controls(
        stationarity_by_case,
        confidence_level=confidence_level,
        rhat_limit=rhat_limit,
        min_effective_samples=min_effective_samples,
    )
    directional_values: list[float] = []
    diagnostics_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id, stationarity in stationarity_by_case.items():
        chain_rows = _energy_chain_rows(case_id, stationarity)
        diagnostics_by_case[case_id] = chain_rows
        for row in chain_rows:
            directional_values.extend(float(row[name]) for name in ENERGY_DIRECTIONAL_STATISTICS)

    test_count = len(directional_values)
    alpha = 1.0 - confidence_level
    critical_value = float(norm.ppf(1.0 - alpha / (2.0 * test_count)))
    cases: dict[str, dict[str, Any]] = {}
    for case_id, stationarity in stationarity_by_case.items():
        chain_rows = diagnostics_by_case[case_id]
        maximum_directional_z = max(
            float(row[name]) for row in chain_rows for name in ENERGY_DIRECTIONAL_STATISTICS
        )
        rhat = _finite_float(stationarity.get("rhat_energy"))
        effective_samples = _finite_float(stationarity.get("neff_energy"))
        invariant_checks = {
            "base_numerics_valid": bool(stationarity.get("base_numerics_valid", False)),
            "valid_finite_clean": bool(stationarity.get("valid_finite_clean", False)),
            "density_accounting_clean": bool(stationarity.get("density_accounting_clean", False)),
            "population_weights_controlled": (
                stationarity.get("population_weight_status") == "accepted"
            ),
            "rhat_below_limit": rhat is not None and rhat < rhat_limit,
            "effective_samples_above_minimum": (
                effective_samples is not None and effective_samples >= min_effective_samples
            ),
            "directional_statistics_below_simultaneous_limit": (
                maximum_directional_z <= critical_value
            ),
        }
        failures = [name for name, passed in invariant_checks.items() if not passed]
        cases[case_id] = {
            "status": "accepted" if not failures else "review",
            "source_stationarity_status": stationarity.get("stationarity_energy"),
            "source_stationarity_reason": stationarity.get("stationarity_reason_energy"),
            "source_failing_seeds": _seed_list(
                stationarity.get("stationarity_failing_seeds_energy")
            ),
            "rhat": rhat,
            "effective_samples_min": effective_samples,
            "maximum_directional_z": maximum_directional_z,
            "invariant_checks": invariant_checks,
            "failures": failures,
            "spread_warning_count": int(stationarity.get("spread_warning_count", 0)),
        }

    return {
        "status": (
            "accepted" if all(case["status"] == "accepted" for case in cases.values()) else "review"
        ),
        "method": "bonferroni_simultaneous_directional_screen",
        "confidence_level": confidence_level,
        "directional_statistics": list(ENERGY_DIRECTIONAL_STATISTICS),
        "first_last_block_statistic": "retained_in_source_but_not_double_counted",
        "case_count": len(stationarity_by_case),
        "chain_count": sum(len(rows) for rows in diagnostics_by_case.values()),
        "directional_test_count": test_count,
        "simultaneous_two_sided_critical_z": critical_value,
        "rhat_kind": "stored_split_rhat_without_rank_normalization",
        "rhat_limit": rhat_limit,
        "effective_samples_kind": "minimum_per_seed_autocorrelation_effective_samples",
        "min_effective_samples": min_effective_samples,
        "cases": cases,
    }


def _energy_chain_rows(
    case_id: str,
    stationarity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = stationarity.get("diagnostics")
    energy = diagnostics.get("energy") if isinstance(diagnostics, dict) else None
    rows = energy.get("chain_diagnostics") if isinstance(energy, dict) else None
    seeds = stationarity.get("seeds")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{case_id}: stored energy chain diagnostics are unavailable")
    if not isinstance(seeds, list) or len(seeds) != len(rows):
        raise ValueError(f"{case_id}: seed and energy-chain counts disagree")
    checked: list[dict[str, Any]] = []
    for seed, row in zip(seeds, rows, strict=True):
        if not isinstance(row, dict):
            raise ValueError(f"{case_id}: energy chain diagnostic is not a mapping")
        values = {name: _finite_float(row.get(name)) for name in ENERGY_DIRECTIONAL_STATISTICS}
        if any(value is None or value < 0.0 for value in values.values()):
            raise ValueError(f"{case_id}: energy directional statistic is invalid")
        quarter_value = values["first_last_quarter_z"]
        if quarter_value is None:
            raise ValueError(f"{case_id}: first/last-quarter statistic is invalid")
        quarter_z = float(quarter_value)
        block_z = _finite_float(row.get("first_last_blocking_z"))
        if block_z is None or not np.isclose(block_z, quarter_z, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{case_id}: first/last block statistic is not the stored duplicate")
        checked.append({"seed": int(seed), **values})
    return checked


def _validate_controls(
    stationarity_by_case: Mapping[str, Mapping[str, Any]],
    *,
    confidence_level: float,
    rhat_limit: float,
    min_effective_samples: float,
) -> None:
    if not stationarity_by_case:
        raise ValueError("matrix energy assessment requires at least one case")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not np.isfinite(rhat_limit) or rhat_limit <= 1.0:
        raise ValueError("rhat_limit must be finite and greater than one")
    if not np.isfinite(min_effective_samples) or min_effective_samples <= 0.0:
        raise ValueError("min_effective_samples must be finite and positive")


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _seed_list(value: object) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []
