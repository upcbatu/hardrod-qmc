from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

from hrdmc.artifacts import (
    build_run_provenance,
    ensure_dir,
    write_json,
    write_run_manifest,
)
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import DMCRunControls, controls_to_dict


def write_benchmark_packet_artifacts(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    case_id: str,
    seeds: list[int],
    controls: DMCRunControls,
    parallel_workers: int | None,
    initialization: InitializationControls,
    collective_rn: CollectiveRNControls | None,
    guide_family: str,
    guide_parameter_source: str,
    guide_parameter_source_sha256: str | None,
    guide_parameter_source_manifest_sha256: str | None,
    guide_parameter_source_identity_fingerprint: str | None,
    plot_paths: list[str],
    plot_formats: tuple[str, ...],
    command: list[str] | None,
) -> dict[str, Path]:
    """Persist one benchmark packet and bind every output in its run manifest."""

    summary_path = ensure_dir(output_dir) / "summary.json"
    write_json(summary_path, payload)
    artifacts = {
        "summary": summary_path,
        "seed_table": write_benchmark_packet_seed_table(
            output_dir,
            payload["seed_results"],
        ),
        "packet_table": write_benchmark_packet_table(output_dir, payload),
        "fw_plateau_table": write_benchmark_packet_fw_plateau_table(output_dir, payload),
        "energy_stationarity_table": write_benchmark_packet_energy_stationarity_table(
            output_dir,
            payload,
        ),
        "density_fw_table": write_benchmark_packet_density_fw_table(output_dir, payload),
    }
    manifest_artifacts = [*artifacts.values()]
    manifest_artifacts.extend(output_dir / path for path in plot_paths)
    artifacts["run_manifest"] = write_run_manifest(
        output_dir,
        run_name="dmc_benchmark_packet",
        config={
            "case": case_id,
            "seeds": seeds,
            "controls": controls_to_dict(controls),
            "parallel_workers": parallel_workers,
            "collective_rn": (None if collective_rn is None else collective_rn.to_metadata()),
            "initialization_mode": initialization.mode,
            "init_width_log_sigma": initialization.init_width_log_sigma,
            "relative_alpha": controls.relative_alpha,
            "breathing_preburn_steps": initialization.breathing_preburn_steps,
            "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
            "guide_family": guide_family,
            "guide_parameter_source": guide_parameter_source,
            "guide_parameter_source_sha256": guide_parameter_source_sha256,
            "guide_parameter_source_manifest_sha256": guide_parameter_source_manifest_sha256,
            "guide_parameter_source_identity_fingerprint": (
                guide_parameter_source_identity_fingerprint
            ),
            "pure_config": payload["pure_config"],
            "plot_formats": list(plot_formats),
        },
        artifacts=manifest_artifacts,
        schema_version=str(payload["schema_version"]),
        provenance=build_run_provenance(command),
        status=str(payload["status"]),
    )
    return artifacts


def write_benchmark_packet_seed_table(
    output_dir: Path,
    seed_payloads: list[dict[str, Any]],
) -> Path:
    fields = [
        "seed",
        "status",
        "dmc_mixed_energy",
        "dmc_r2_radius",
        "dmc_rms_radius",
        "r2_schema_status",
        "r2_plateau_status",
        "r2_plateau_value",
        "rms_radius",
        "local_step_method",
        "drift_limiter",
        "local_acceptance_fraction_mean",
        "invalid_proposal_fraction_max",
        "metropolis_rejection_fraction_max",
        "local_energy_median_mean",
        "local_energy_mad_mean",
        "local_energy_p001_min",
        "local_energy_p01_min",
        "local_energy_p99_max",
        "local_energy_p999_max",
        "drift_norm_max",
        "configuration_esjd_mean",
        "r2_esjd_mean",
        "weighted_free_gap_esjd_mean",
        "free_gap_min",
        "free_gap_p01_min",
        "lag_max_block_count",
        "lag_max_weight_ess_min",
        "r2_genealogy_status",
        "lag_max_source_ancestor_ess_min",
        "lag_max_unique_source_ancestor_min",
        "lag_max_source_family_fraction_max",
    ]
    path = ensure_dir(output_dir) / "seed_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in seed_payloads:
            writer.writerow(_seed_table_row(payload))
    return path


def write_benchmark_packet_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    path = ensure_dir(output_dir) / "packet_table.csv"
    estimates = payload["estimates"]
    pure = payload.get("pure_walking", {})
    aggregate_r2 = pure.get("r2_aggregate_plateau_diagnostics", {})
    if not isinstance(aggregate_r2, dict):
        aggregate_r2 = {}
    row = {
        "case_id": payload["case_id"],
        "status": payload["status"],
        "energy_status": payload["energy_validation_status"],
        "pure_fw_status": payload["pure_fw_validation_status"],
        "collective_rn_enabled": payload.get("collective_rn_controls") is not None,
        "guide_family": payload.get("guide_family", ""),
        "energy": estimates["energy"]["value"],
        "energy_stderr": estimates["energy"]["stderr"],
        "energy_delta_vs_lda": estimates["energy"]["delta_vs_lda"],
        "pure_r2": estimates["r2"]["value"],
        "pure_r2_stderr": estimates["r2"]["stderr"],
        "pure_r2_delta_vs_lda": estimates["r2"]["delta_vs_lda"],
        "rms_radius": estimates["rms"]["value"],
        "rms_radius_stderr": estimates["rms"]["stderr"],
        "rms_delta_vs_lda": estimates["rms"]["delta_vs_lda"],
        "density_status": estimates["density"]["status"],
        "pair_distance_density_status": estimates["pair_distance_density"]["status"],
        "structure_factor_status": estimates["structure_factor"]["status"],
        "mixed_density_l2_diagnostic": estimates["density"]["mixed_diagnostic_density_l2"],
        "r2_aggregate_plateau_status": pure.get("r2_aggregate_plateau_status", ""),
        "r2_aggregate_slope_sigma_ratio": _slope_sigma_ratio(aggregate_r2),
        "r2_aggregate_window_sigma_ratio": _window_sigma_ratio(aggregate_r2),
        "r2_seed_plateau_resolved_count": pure.get("r2_seed_plateau_resolved_count", ""),
        "r2_seed_plateau_unresolved_count": pure.get("r2_seed_plateau_unresolved_count", ""),
    }
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def write_benchmark_packet_fw_plateau_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    fields = [
        "row_type",
        "seed",
        "r2_plateau_status",
        "r2_plateau_value",
        "r2_plateau_stderr",
        "slope_sigma_ratio",
        "window_sigma_ratio",
        "slope_delta",
        "slope_threshold",
        "max_window_delta",
        "min_window_threshold",
        "slope_pass",
        "window_spread_pass",
        "window_lags",
        "lag_max_block_count",
        "lag_max_weight_ess_min",
        "genealogy_status",
        "lag_max_source_ancestor_ess_min",
        "lag_max_unique_source_ancestor_min",
        "lag_max_source_family_fraction_max",
    ]
    path = ensure_dir(output_dir) / "fw_plateau_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        aggregate = _aggregate_r2_row(payload)
        if aggregate:
            writer.writerow(aggregate)
        for seed_payload in payload.get("seed_results", []):
            writer.writerow(_seed_r2_plateau_row(seed_payload))
    return path


def write_benchmark_packet_energy_stationarity_table(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    fields = [
        "row_type",
        "seed",
        "energy_status",
        "failures",
        "slope_z",
        "first_last_quarter_z",
        "late_cumulative_z",
        "first_last_blocking_z",
        "spread_blocking_z",
        "effective_independent_samples",
        "tau_int_samples",
        "point_count",
        "block_count",
        "stationarity_clean",
        "spread_warning",
        "spread_veto",
        "rhat",
        "neff_min",
    ]
    path = ensure_dir(output_dir) / "energy_stationarity_table.csv"
    stationarity = payload.get("stationarity", {})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_aggregate_energy_stationarity_row(stationarity))
        for row in _seed_energy_stationarity_rows(stationarity):
            writer.writerow(row)
    return path


def write_benchmark_packet_density_fw_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    fields = [
        "row_type",
        "seed",
        "density_plateau_status",
        "density_integral",
        "mixed_density_l2_diagnostic",
        "slope_sigma_ratio",
        "window_sigma_ratio",
        "slope_delta",
        "slope_threshold",
        "max_window_delta",
        "min_window_threshold",
        "slope_pass",
        "window_spread_pass",
        "window_lags",
        "lag_max_block_count",
        "lag_max_weight_ess_min",
        "genealogy_status",
        "lag_max_source_ancestor_ess_min",
        "lag_max_unique_source_ancestor_min",
        "lag_max_source_family_fraction_max",
    ]
    path = ensure_dir(output_dir) / "density_fw_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_aggregate_density_row(payload))
        for seed_payload in payload.get("seed_results", []):
            writer.writerow(_seed_density_fw_row(seed_payload))
    return path


def _seed_table_row(payload: dict[str, Any]) -> dict[str, Any]:
    r2 = payload["pure_walking"]["observable_results"].get("r2", {})
    dmc = payload["dmc_summary"]
    lag_steps = r2.get("lag_steps", [])
    lag_max = lag_steps[-1] if lag_steps else ""
    return {
        "seed": payload["seed"],
        "status": payload["status"],
        "dmc_mixed_energy": dmc["mixed_energy"],
        "dmc_r2_radius": dmc["r2_radius"],
        "dmc_rms_radius": dmc["rms_radius"],
        "r2_schema_status": r2.get("schema_status", ""),
        "r2_plateau_status": r2.get("plateau_status", ""),
        "r2_plateau_value": r2.get("plateau_value", ""),
        "rms_radius": r2.get("rms_radius", ""),
        "local_step_method": dmc["metadata"].get("local_step_method", ""),
        "drift_limiter": dmc["metadata"].get("drift_limiter", ""),
        "local_acceptance_fraction_mean": dmc["metadata"].get(
            "local_acceptance_fraction_mean",
            "",
        ),
        "invalid_proposal_fraction_max": dmc["metadata"].get(
            "invalid_proposal_fraction_max",
            "",
        ),
        "metropolis_rejection_fraction_max": dmc["metadata"].get(
            "metropolis_rejection_fraction_max",
            "",
        ),
        "local_energy_median_mean": dmc["metadata"].get("local_energy_median_mean", ""),
        "local_energy_mad_mean": dmc["metadata"].get("local_energy_mad_mean", ""),
        "local_energy_p001_min": dmc["metadata"].get("local_energy_p001_min", ""),
        "local_energy_p01_min": dmc["metadata"].get("local_energy_p01_min", ""),
        "local_energy_p99_max": dmc["metadata"].get("local_energy_p99_max", ""),
        "local_energy_p999_max": dmc["metadata"].get("local_energy_p999_max", ""),
        "drift_norm_max": dmc["metadata"].get("drift_norm_max", ""),
        "configuration_esjd_mean": dmc["metadata"].get("configuration_esjd_mean", ""),
        "r2_esjd_mean": dmc["metadata"].get("r2_esjd_mean", ""),
        "weighted_free_gap_esjd_mean": dmc["metadata"].get(
            "weighted_free_gap_esjd_mean",
            "",
        ),
        "free_gap_min": dmc["metadata"].get("free_gap_min", ""),
        "free_gap_p01_min": dmc["metadata"].get("free_gap_p01_min", ""),
        "lag_max_block_count": _lag_dict_get(r2.get("block_count_by_lag", {}), lag_max),
        "lag_max_weight_ess_min": _lag_dict_get(
            r2.get("block_weight_ess_min_by_lag", {}),
            lag_max,
        ),
        "r2_genealogy_status": r2.get("genealogy_status", ""),
        "lag_max_source_ancestor_ess_min": _lag_dict_get(
            r2.get("block_source_ancestor_ess_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_unique_source_ancestor_min": _lag_dict_get(
            r2.get("block_unique_source_ancestor_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_source_family_fraction_max": _lag_dict_get(
            r2.get("block_max_source_family_fraction_by_lag", {}),
            lag_max,
        ),
    }


def _lag_dict_get(values: object, lag: object) -> object:
    if not isinstance(values, dict):
        return ""
    return values.get(lag, values.get(str(lag), ""))


def _aggregate_r2_row(payload: dict[str, Any]) -> dict[str, Any]:
    pure = payload.get("pure_walking", {})
    diagnostics = pure.get("r2_aggregate_plateau_diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics:
        return {}
    return {
        **_plateau_diagnostic_row(diagnostics),
        "row_type": "aggregate",
        "seed": "all",
        "r2_plateau_status": pure.get("r2_aggregate_plateau_status", ""),
        "r2_plateau_value": pure.get("r2_aggregate_plateau_value", ""),
        "r2_plateau_stderr": pure.get("r2_aggregate_plateau_stderr", ""),
    }


def _seed_r2_plateau_row(payload: dict[str, Any]) -> dict[str, Any]:
    r2 = payload.get("pure_walking", {}).get("observable_results", {}).get("r2", {})
    diagnostics = r2.get("plateau_diagnostics", {})
    lag_steps = r2.get("lag_steps", [])
    lag_max = lag_steps[-1] if lag_steps else ""
    return {
        **_plateau_diagnostic_row(diagnostics if isinstance(diagnostics, dict) else {}),
        "row_type": "seed",
        "seed": payload.get("seed", ""),
        "r2_plateau_status": r2.get("plateau_status", ""),
        "r2_plateau_value": r2.get("plateau_value", ""),
        "r2_plateau_stderr": r2.get("plateau_stderr", ""),
        "lag_max_block_count": _lag_dict_get(r2.get("block_count_by_lag", {}), lag_max),
        "lag_max_weight_ess_min": _lag_dict_get(
            r2.get("block_weight_ess_min_by_lag", {}),
            lag_max,
        ),
        "genealogy_status": r2.get("genealogy_status", ""),
        "lag_max_source_ancestor_ess_min": _lag_dict_get(
            r2.get("block_source_ancestor_ess_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_unique_source_ancestor_min": _lag_dict_get(
            r2.get("block_unique_source_ancestor_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_source_family_fraction_max": _lag_dict_get(
            r2.get("block_max_source_family_fraction_by_lag", {}),
            lag_max,
        ),
    }


def _aggregate_energy_stationarity_row(stationarity: dict[str, Any]) -> dict[str, Any]:
    audit = stationarity.get("stationarity_failure_audit", {}).get("energy", {})
    if not isinstance(audit, dict):
        audit = {}
    return {
        "row_type": "aggregate",
        "seed": "all",
        "energy_status": stationarity.get("stationarity_energy", ""),
        "failures": stationarity.get("stationarity_reason_energy", ""),
        "slope_z": stationarity.get("stationarity_slope_z_max_energy", ""),
        "first_last_quarter_z": stationarity.get("stationarity_quarter_z_max_energy", ""),
        "late_cumulative_z": stationarity.get("stationarity_late_z_max_energy", ""),
        "first_last_blocking_z": stationarity.get("stationarity_block_z_max_energy", ""),
        "spread_blocking_z": audit.get("spread_blocking_z_max", ""),
        "effective_independent_samples": stationarity.get("neff_energy", ""),
        "tau_int_samples": "",
        "point_count": "",
        "block_count": "",
        "stationarity_clean": stationarity.get("stationarity_energy", "")
        in {"accepted", "spread_warning"},
        "spread_warning": stationarity.get("stationarity_reason_energy", "") == "spread_warning",
        "spread_veto": "",
        "rhat": stationarity.get("rhat_energy", ""),
        "neff_min": stationarity.get("neff_energy", ""),
    }


def _seed_energy_stationarity_rows(stationarity: dict[str, Any]) -> list[dict[str, Any]]:
    audit = stationarity.get("stationarity_failure_audit", {}).get("energy", {})
    if not isinstance(audit, dict):
        audit = {}
    audit_rows = {
        row.get("seed"): row
        for row in audit.get("per_seed", [])
        if isinstance(row, dict) and "seed" in row
    }
    seeds = stationarity.get("seeds", [])
    chain_rows = stationarity.get("diagnostics", {}).get("energy", {}).get("chain_diagnostics", [])
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds if isinstance(seeds, list) else []):
        audit_row = audit_rows.get(seed, {})
        chain_row = (
            chain_rows[index]
            if index < len(chain_rows) and isinstance(chain_rows[index], dict)
            else {}
        )
        failures = audit_row.get("failures", [])
        rows.append(
            {
                "row_type": "seed",
                "seed": seed,
                "energy_status": "stationary"
                if bool(chain_row.get("stationarity_clean", False))
                else "flagged",
                "failures": ",".join(str(item) for item in failures)
                if isinstance(failures, list)
                else failures,
                "slope_z": audit_row.get(
                    "slope_z_autocorr_adjusted",
                    chain_row.get("slope_z_autocorr_adjusted", ""),
                ),
                "first_last_quarter_z": audit_row.get(
                    "first_last_quarter_z",
                    chain_row.get("first_last_quarter_z", ""),
                ),
                "late_cumulative_z": audit_row.get(
                    "late_cumulative_z",
                    chain_row.get("late_cumulative_z", ""),
                ),
                "first_last_blocking_z": audit_row.get(
                    "first_last_blocking_z",
                    chain_row.get("first_last_blocking_z", ""),
                ),
                "spread_blocking_z": audit_row.get(
                    "spread_blocking_z",
                    chain_row.get("spread_blocking_z", ""),
                ),
                "effective_independent_samples": chain_row.get(
                    "effective_independent_samples",
                    "",
                ),
                "tau_int_samples": chain_row.get("tau_int_samples", ""),
                "point_count": chain_row.get("point_count", ""),
                "block_count": chain_row.get("block_count", ""),
                "stationarity_clean": chain_row.get("stationarity_clean", ""),
                "spread_warning": chain_row.get("spread_warning", ""),
                "spread_veto": chain_row.get("spread_veto", ""),
                "rhat": "",
                "neff_min": "",
            }
        )
    return rows


def _aggregate_density_row(payload: dict[str, Any]) -> dict[str, Any]:
    density = payload.get("estimates", {}).get("density", {})
    return {
        "row_type": "aggregate",
        "seed": "all",
        "density_plateau_status": density.get("status", ""),
        "density_integral": density.get("integral", ""),
        "mixed_density_l2_diagnostic": density.get("mixed_diagnostic_density_l2", ""),
        "slope_sigma_ratio": "",
        "window_sigma_ratio": "",
        "slope_delta": "",
        "slope_threshold": "",
        "max_window_delta": "",
        "min_window_threshold": "",
        "slope_pass": "",
        "window_spread_pass": "",
        "window_lags": "",
        "lag_max_block_count": "",
        "lag_max_weight_ess_min": "",
        "genealogy_status": "",
        "lag_max_source_ancestor_ess_min": "",
        "lag_max_unique_source_ancestor_min": "",
        "lag_max_source_family_fraction_max": "",
    }


def _seed_density_fw_row(payload: dict[str, Any]) -> dict[str, Any]:
    density = payload.get("pure_walking", {}).get("observable_results", {}).get("density", {})
    diagnostics = density.get("plateau_diagnostics", {})
    lag_steps = density.get("lag_steps", [])
    lag_max = lag_steps[-1] if lag_steps else ""
    return {
        **_plateau_diagnostic_row(diagnostics if isinstance(diagnostics, dict) else {}),
        "row_type": "seed",
        "seed": payload.get("seed", ""),
        "density_plateau_status": density.get("plateau_status", ""),
        "density_integral": "",
        "mixed_density_l2_diagnostic": "",
        "lag_max_block_count": _lag_dict_get(density.get("block_count_by_lag", {}), lag_max),
        "lag_max_weight_ess_min": _lag_dict_get(
            density.get("block_weight_ess_min_by_lag", {}),
            lag_max,
        ),
        "genealogy_status": density.get("genealogy_status", ""),
        "lag_max_source_ancestor_ess_min": _lag_dict_get(
            density.get("block_source_ancestor_ess_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_unique_source_ancestor_min": _lag_dict_get(
            density.get("block_unique_source_ancestor_min_by_lag", {}),
            lag_max,
        ),
        "lag_max_source_family_fraction_max": _lag_dict_get(
            density.get("block_max_source_family_fraction_by_lag", {}),
            lag_max,
        ),
    }


def _plateau_diagnostic_row(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "slope_sigma_ratio": _slope_sigma_ratio(diagnostics),
        "window_sigma_ratio": _window_sigma_ratio(diagnostics),
        "slope_delta": diagnostics.get("slope_delta", ""),
        "slope_threshold": diagnostics.get("slope_threshold", ""),
        "max_window_delta": diagnostics.get(
            "max_window_delta",
            diagnostics.get("delta", ""),
        ),
        "min_window_threshold": diagnostics.get(
            "min_window_threshold",
            diagnostics.get("threshold", ""),
        ),
        "slope_pass": diagnostics.get("slope_pass", ""),
        "window_spread_pass": diagnostics.get("window_spread_pass", ""),
        "window_lags": _join_lags(diagnostics.get("window_lags", "")),
        "lag_max_block_count": "",
        "lag_max_weight_ess_min": "",
    }


def _slope_sigma_ratio(diagnostics: dict[str, Any]) -> object:
    if not isinstance(diagnostics, dict):
        return ""
    return _safe_ratio(diagnostics.get("slope_delta"), diagnostics.get("slope_threshold"))


def _window_sigma_ratio(diagnostics: dict[str, Any]) -> object:
    if not isinstance(diagnostics, dict):
        return ""
    deltas = diagnostics.get("window_delta_by_lag")
    thresholds = diagnostics.get("window_threshold_by_lag")
    if isinstance(deltas, dict) and isinstance(thresholds, dict):
        ratios: list[float] = []
        for lag, delta in deltas.items():
            threshold = thresholds.get(lag, thresholds.get(str(lag)))
            ratio = _safe_ratio(delta, threshold)
            if isinstance(ratio, float):
                ratios.append(ratio)
        if ratios:
            return max(ratios)
    return _safe_ratio(diagnostics.get("delta"), diagnostics.get("threshold"))


def _safe_ratio(numerator: object, denominator: object) -> object:
    try:
        num = abs(float(cast(Any, numerator)))
        den = float(cast(Any, denominator))
    except (TypeError, ValueError):
        return ""
    if den <= 0.0:
        return ""
    return num / den


def _join_lags(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value) if value else ""
