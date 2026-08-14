from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import ensure_dir, write_csv, write_json, write_run_manifest
from hrdmc.sampling.initial_conditions import InitializationControls
from hrdmc.system.settings import DMCRunControls, controls_to_dict


def write_benchmark_packet_artifacts(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    case_id: str,
    seeds: list[int],
    controls: DMCRunControls,
    parallel_workers: int | None,
    initialization: InitializationControls,
    guide_family: str,
    guide_parameter_source: str,
    plot_paths: list[str],
    plot_formats: tuple[str, ...],
    command: list[str] | None,
) -> dict[str, Path]:
    root = ensure_dir(output_dir)
    summary = root / "summary.json"
    write_json(summary, payload)
    artifacts = {
        "summary": summary,
        "seed_table": write_benchmark_packet_seed_table(root, payload["seed_results"]),
        "packet_table": _write_benchmark_packet_table(root, payload),
        "fw_plateau_table": write_benchmark_packet_fw_plateau_table(root, payload),
        "energy_stationarity_table": _write_benchmark_packet_energy_stationarity_table(
            root, payload
        ),
        "density_fw_table": write_benchmark_packet_density_fw_table(root, payload),
    }
    artifacts["run_manifest"] = write_run_manifest(
        root,
        run_name="dmc_benchmark_packet",
        config={
            "case": case_id,
            "seeds": seeds,
            "controls": controls_to_dict(controls),
            "parallel_workers": parallel_workers,
            "initialization_mode": initialization.mode,
            "init_width_log_sigma": initialization.init_width_log_sigma,
            "relative_alpha": controls.relative_alpha,
            "breathing_preburn_steps": initialization.breathing_preburn_steps,
            "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
            "guide_family": guide_family,
            "guide_parameter_source": guide_parameter_source,
            "pure_config": payload["pure_config"],
            "plot_formats": list(plot_formats),
            "command": command,
        },
        artifacts=[*artifacts.values(), *(root / path for path in plot_paths)],
        status=str(payload["status"]),
    )
    return artifacts
def write_benchmark_packet_seed_table(
    output_dir: Path,
    seed_payloads: list[dict[str, Any]],
) -> Path:
    rows = (_seed_row(row) for row in seed_payloads)
    return write_csv(output_dir / "seed_table.csv", rows)
def _write_benchmark_packet_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    estimates = _mapping(payload.get("estimates"))
    pure = _mapping(payload.get("pure_walking"))
    r2_diag = _mapping(pure.get("r2_aggregate_plateau_diagnostics"))
    density_diag = _mapping(
        _mapping(_mapping(pure.get("observables")).get("density")).get(
            "aggregate_plateau_diagnostics"
        )
    )
    row = {
        "case_id": payload.get("case_id"),
        "status": payload.get("status"),
        "energy_status": payload.get("energy_validation_status"),
        "pure_fw_status": payload.get("pure_fw_validation_status"),
        "guide_family": payload.get("guide_family"),
        **_estimate_fields("energy", _mapping(estimates.get("energy"))),
        **_estimate_fields("r2", _mapping(estimates.get("r2"))),
        **_estimate_fields("rms", _mapping(estimates.get("rms"))),
        **_estimate_fields("density", _mapping(estimates.get("density"))),
        **_diagnostic_fields("r2", r2_diag),
        **_diagnostic_fields("density", density_diag),
    }
    return write_csv(output_dir / "packet_table.csv", [row])
def write_benchmark_packet_fw_plateau_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    return write_csv(
        output_dir / "fw_plateau_table.csv",
        _observable_rows(payload, "r2"),
    )
def _write_benchmark_packet_energy_stationarity_table(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    stationarity = _mapping(payload.get("stationarity"))
    rows = [
        {
            "row_type": "aggregate",
            "seed": "all",
            "energy_status": stationarity.get("stationarity_energy"),
            "failures": stationarity.get("stationarity_reason_energy"),
            "rhat": stationarity.get("rhat_energy"),
            "neff_min": stationarity.get("neff_energy"),
            "slope_z": stationarity.get("stationarity_slope_z_max_energy"),
            "first_last_quarter_z": stationarity.get("stationarity_quarter_z_max_energy"),
            "late_cumulative_z": stationarity.get("stationarity_late_z_max_energy"),
            "first_last_blocking_z": stationarity.get("stationarity_block_z_max_energy"),
        }
    ]
    audit = _mapping(_mapping(stationarity.get("stationarity_failure_audit")).get("energy"))
    diagnostics = _mapping(_mapping(stationarity.get("diagnostics")).get("energy"))
    chain_rows = diagnostics.get("chain_diagnostics")
    chain_rows = chain_rows if isinstance(chain_rows, list) else []
    audit_rows = audit.get("per_seed")
    audit_rows = audit_rows if isinstance(audit_rows, list) else []
    seeds = stationarity.get("seeds")
    for seed, chain, audit_row in zip(
        seeds if isinstance(seeds, list) else [], chain_rows, audit_rows, strict=False
    ):
        rows.append(
            {
                "row_type": "seed",
                "seed": seed,
                **_mapping(chain),
                "failures": ",".join(_mapping(audit_row).get("failures", [])),
            }
        )
    return write_csv(output_dir / "energy_stationarity_table.csv", rows)
def write_benchmark_packet_density_fw_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    return write_csv(
        output_dir / "density_fw_table.csv",
        _observable_rows(payload, "density"),
    )
def _seed_row(payload: dict[str, Any]) -> dict[str, Any]:
    dmc = _mapping(payload.get("dmc_summary"))
    metadata = _mapping(dmc.get("metadata"))
    pure = _mapping(_mapping(payload.get("pure_walking")).get("observable_results"))
    r2 = _mapping(pure.get("r2"))
    fields = [
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
    ]
    return {
        "seed": payload.get("seed"),
        "status": payload.get("status"),
        "dmc_mixed_energy": dmc.get("mixed_energy"),
        "dmc_r2_radius": dmc.get("r2_radius"),
        "dmc_rms_radius": dmc.get("rms_radius"),
        "r2_schema_status": r2.get("schema_status"),
        "r2_plateau_status": r2.get("plateau_status"),
        "r2_plateau_value": r2.get("plateau_value"),
        "rms_radius": r2.get("rms_radius"),
        **{
            "r2_genealogy_status" if key == "genealogy_status" else key: value
            for key, value in _lag_fields(r2).items()
        },
        **{field: metadata.get(field) for field in fields},
    }
def _observable_rows(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    pure = _mapping(payload.get("pure_walking"))
    observable = _mapping(_mapping(pure.get("observables")).get(name))
    diagnostics = _mapping(observable.get("aggregate_plateau_diagnostics"))
    rows = [
        {
            "row_type": "aggregate",
            "seed": "all",
            "status": observable.get("status"),
            "plateau_status": observable.get("aggregate_plateau_status"),
            "genealogy_status": observable.get("aggregate_genealogy_status"),
            **_diagnostic_fields(name, diagnostics),
        }
    ]
    for seed in payload.get("seed_results", []):
        result = _mapping(_mapping(_mapping(seed).get("pure_walking")).get("observable_results"))
        value = _mapping(result.get(name))
        rows.append(
            {
                "row_type": "seed",
                "seed": _mapping(seed).get("seed"),
                "plateau_status": value.get("plateau_status"),
                "plateau_value": value.get("plateau_value"),
                "plateau_stderr": value.get("plateau_stderr"),
                **_lag_fields(value),
                **_diagnostic_fields(name, _mapping(value.get("plateau_diagnostics"))),
            }
        )
    return rows
def _estimate_fields(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": item for key, item in value.items() if not isinstance(item, (list, dict))
    }
def _diagnostic_fields(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "confidence_level",
        "slope_delta",
        "slope_threshold",
        "max_window_delta",
        "min_window_threshold",
        "equivalence_pass",
        "simultaneous_rms_relative_upper_bound",
        "simultaneous_density_relative_l2_upper_bound",
        "selected_window_pooled_ancestor_ess_lower_min",
        "selected_window_pooled_family_fraction_upper_max",
    ]
    return {f"{prefix}_{key}": value.get(key) for key in keys}
def _lag_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    lags = value.get("lag_steps")
    lag = max(lags) if isinstance(lags, list) and lags else None
    def at_max(name: str) -> Any:
        values = value.get(name)
        if not isinstance(values, dict) or lag is None:
            return None
        return values.get(lag, values.get(str(lag)))
    return {
        "genealogy_status": value.get("genealogy_status"),
        "lag_max_source_ancestor_ess_min": at_max("block_source_ancestor_ess_min_by_lag"),
        "lag_max_unique_source_ancestor_min": at_max("block_unique_source_ancestor_min_by_lag"),
    }
def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
