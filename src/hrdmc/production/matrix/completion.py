from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hrdmc.artifacts.manifest import config_fingerprint, verify_run_manifest
from hrdmc.production.matrix.method import RowMethod
from hrdmc.sampling.dmc.run import parse_seeds
from hrdmc.system.settings import DMCRunControls, controls_to_dict

if TYPE_CHECKING:
    from hrdmc.production.matrix.final_matrix import FinalMatrixConfig


def verified_completed_row(
    config: FinalMatrixConfig,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> tuple[bool, list[str]]:
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False, ["missing summary.json or run_manifest.json"]
    try:
        verified, errors = verify_run_manifest(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, [f"invalid benchmark packet: {exc}"]
    if not verified:
        return False, errors
    config_payload = manifest.get("config")
    if not isinstance(config_payload, dict):
        return False, ["run manifest has no configuration payload"]
    errors = _identity_errors(manifest, summary, case_id, config_payload)
    expected = _expected_manifest_fields(config, case_id, grid_plan, method)
    mismatches = [key for key, value in expected.items() if _nested(config_payload, key) != value]
    if mismatches:
        errors.append(f"configuration mismatch: {', '.join(mismatches)}")
    required = {
        "summary.json",
        "seed_table.csv",
        "packet_table.csv",
        "fw_plateau_table.csv",
        "energy_stationarity_table.csv",
        "density_fw_table.csv",
    }
    recorded = {
        str(entry.get("path")) for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if missing := sorted(required - recorded):
        errors.append(f"manifest missing required artifacts: {', '.join(missing)}")
    return not errors, errors


def _identity_errors(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    case_id: str,
    config: dict[str, Any],
) -> list[str]:
    checks = (
        (
            manifest.get("run_name") == "dmc_benchmark_packet",
            "run manifest has the wrong artifact owner",
        ),
        (
            manifest.get("status") == "accepted",
            f"run manifest scientific status is {manifest.get('status')!r}",
        ),
        (
            summary.get("case_id") == case_id,
            "benchmark summary case does not match the planned case",
        ),
        (
            summary.get("status") == "accepted",
            f"benchmark summary scientific status is {summary.get('status')!r}",
        ),
        (
            summary.get("status") == manifest.get("status"),
            "benchmark summary and run manifest statuses disagree",
        ),
        (
            manifest.get("config_fingerprint") == config_fingerprint(config),
            "run manifest configuration fingerprint mismatch",
        ),
    )
    return [message for accepted, message in checks if not accepted]


def _expected_manifest_fields(
    config: FinalMatrixConfig,
    case_id: str,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> dict[str, Any]:
    controls = DMCRunControls(
        dt=method.dt,
        walkers=method.walkers,
        burn_tau=config.burn_tau,
        production_tau=config.production_tau,
        store_every=method.store_every,
        grid_extent=float(grid_plan["grid_extent"]),
        n_bins=int(grid_plan["n_bins"]),
        ess_resample_fraction=config.ess_resample_fraction,
        drift_limiter=method.drift_limiter,
        relative_alpha=method.relative_alpha,
    )
    return {
        "case": case_id,
        "seeds": parse_seeds(config.seeds),
        "controls": controls_to_dict(controls),
        "parallel_workers": config.parallel_workers,
        "initialization_mode": method.initialization_mode,
        "init_width_log_sigma": method.init_width_log_sigma,
        "breathing_preburn_steps": method.breathing_preburn_steps,
        "breathing_preburn_log_step": method.breathing_preburn_log_step,
        "guide_family": method.guide_family,
        "guide_parameter_source": method.guide_parameter_source,
        "pure_config.lag_steps": list(method.pure_fw_lags),
        "pure_config.density_lag_steps": list(method.pure_fw_density_lags),
        "pure_config.observables": ["r2", "density"],
        "pure_config.observable_source": "r2_rb",
        "pure_config.density_source": "com_rao_blackwell",
        "pure_config.density_parity_average": True,
        "pure_config.block_size_steps": config.pure_fw_block_size_steps,
        "pure_config.collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_config.density_collection_stride_steps": (
            method.pure_fw_density_collection_stride_steps
        ),
        "pure_config.min_block_count": config.pure_fw_min_block_count,
        "pure_config.min_walker_weight_ess": config.pure_fw_min_walker_weight_ess,
        "pure_config.min_source_ancestor_ess": config.pure_fw_min_source_ancestor_ess,
        "pure_config.max_source_family_fraction": config.pure_fw_max_source_family_fraction,
        "pure_config.rms_plateau_relative_tolerance": config.pure_fw_rms_plateau_relative_tolerance,
        "pure_config.plateau_equivalence_confidence_level": (
            config.pure_fw_plateau_equivalence_confidence_level
        ),
        "pure_config.plateau_window_lag_count": 4,
        "pure_config.density_plateau_window_lag_count": (
            config.pure_fw_density_plateau_window_lag_count
        ),
        "plot_formats": [value.strip() for value in config.plot_formats.split(",")],
    }


def _nested(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
