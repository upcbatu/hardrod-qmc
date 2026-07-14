from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts import (
    build_run_provenance,
    file_sha256,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.plotting import write_benchmark_packet_plots
from hrdmc.workflows.dmc.benchmark_packet.case import (
    BENCHMARK_PACKET_SCHEMA_VERSION,
    estimates,
    summarize_pure_seed_payloads,
)
from hrdmc.workflows.dmc.benchmark_packet.outputs import (
    write_benchmark_packet_density_fw_table,
    write_benchmark_packet_energy_stationarity_table,
    write_benchmark_packet_fw_plateau_table,
    write_benchmark_packet_seed_table,
    write_benchmark_packet_table,
)
from hrdmc.workflows.dmc.benchmark_packet.selection import (
    benchmark_validation_status,
    energy_validation_status,
    pure_fw_validation_status,
)
from hrdmc.workflows.dmc.pure_walking.case import pure_config_metadata

BENCHMARK_PACKET_REANALYSIS_SCHEMA_VERSION = "dmc_benchmark_packet_reanalysis_v2"
BENCHMARK_PACKET_REANALYSIS_RUN_NAME = "dmc_benchmark_packet_reanalysis"
BENCHMARK_PACKET_REANALYSIS_MATRIX_SCHEMA_VERSION = "dmc_benchmark_packet_reanalysis_matrix_v2"
BENCHMARK_PACKET_REANALYSIS_MATRIX_RUN_NAME = "dmc_benchmark_packet_reanalysis_matrix"
SOURCE_BENCHMARK_PACKET_SCHEMA_VERSIONS = {
    "dmc_benchmark_packet_v2",
    BENCHMARK_PACKET_SCHEMA_VERSION,
}


def reanalyze_benchmark_packet(
    source_dir: Path,
    output_dir: Path,
    *,
    rms_relative_equivalence_margin: float,
    confidence_level: float,
    sensitivity_margins: tuple[float, ...],
    policy_timing: str,
    plot_formats: tuple[str, ...] = ("png", "pdf"),
    write_plots: bool = True,
    command: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Reclassify one immutable packet from its stored independent-seed FW ladder."""

    _validate_reanalysis_controls(
        rms_relative_equivalence_margin=rms_relative_equivalence_margin,
        confidence_level=confidence_level,
        sensitivity_margins=sensitivity_margins,
        policy_timing=policy_timing,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"reanalysis output directory is not empty: {output_dir}")

    source_summary_path = source_dir / "summary.json"
    source_manifest_path = source_dir / "run_manifest.json"
    source_manifest = _verified_source_manifest(source_manifest_path)
    source_payload = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_payload.get("schema_version") not in SOURCE_BENCHMARK_PACKET_SCHEMA_VERSIONS:
        raise ValueError("source summary has an unsupported benchmark-packet schema")
    _validate_source_identity(
        source_dir=source_dir,
        output_dir=output_dir,
        summary=source_payload,
        manifest=source_manifest,
    )

    config = _pure_config_from_payload(
        source_payload,
        rms_relative_equivalence_margin=rms_relative_equivalence_margin,
        confidence_level=confidence_level,
    )
    pure_summary = summarize_pure_seed_payloads(
        source_payload["seed_results"],
        config=config,
    )
    stationarity = source_payload["stationarity"]
    recomputed_energy_status = energy_validation_status(stationarity)
    energy_status = str(source_payload.get("energy_validation_status", ""))
    if recomputed_energy_status != energy_status:
        raise ValueError("source energy status is not reproducible from stored stationarity")
    pure_status = pure_fw_validation_status(pure_summary)
    status = benchmark_validation_status(energy_status=energy_status, pure_status=pure_status)
    calculation_name = (
        "DMC"
        if source_payload.get("collective_rn_controls") is None
        else "DMC with scheduled collective RN moves"
    )
    recomputed_estimates = estimates(
        stationarity,
        pure_summary,
        calculation_name=calculation_name,
    )
    _assert_unmodified_observables(
        source_payload=source_payload,
        pure_summary=pure_summary,
        recomputed_estimates=recomputed_estimates,
    )
    derived_estimates = copy.deepcopy(source_payload["estimates"])
    derived_estimates["r2"] = recomputed_estimates["r2"]
    derived_estimates["rms"] = recomputed_estimates["rms"]
    derived_estimates["density"] = recomputed_estimates["density"]

    payload = copy.deepcopy(source_payload)
    payload.update(
        {
            "schema_version": BENCHMARK_PACKET_REANALYSIS_SCHEMA_VERSION,
            "status": status,
            "energy_validation_status": energy_status,
            "pure_fw_validation_status": pure_status,
            "pure_config": pure_config_metadata(config),
            "pure_walking": pure_summary,
            "estimates": derived_estimates,
        }
    )
    r2_diagnostics = pure_summary.get("r2_aggregate_plateau_diagnostics", {})
    relative_upper_bound = _optional_finite_float(
        r2_diagnostics.get("simultaneous_rms_relative_upper_bound")
        if isinstance(r2_diagnostics, dict)
        else None
    )
    payload["reanalysis"] = {
        "method": "paired_seed_simultaneous_rms_equivalence",
        "scope": (
            "aggregate forward-walking R2 and density-lag equivalence classification "
            "plus derived density L2 reporting; stored seed samples, genealogy support, "
            "energy, stationarity and density arrays are unchanged"
        ),
        "policy_timing": policy_timing,
        "rms_relative_equivalence_margin": rms_relative_equivalence_margin,
        "confidence_level": confidence_level,
        "sensitivity": [
            {
                "rms_relative_equivalence_margin": margin,
                "equivalent": (relative_upper_bound is not None and relative_upper_bound <= margin),
            }
            for margin in sensitivity_margins
        ],
        "source": {
            "summary_path": str(source_summary_path.resolve()),
            "summary_sha256": file_sha256(source_summary_path),
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": file_sha256(source_manifest_path),
            "run_id": source_manifest["run_id"],
            "bundle_sha256": source_manifest["bundle_sha256"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        write_benchmark_packet_plots(output_dir, payload, formats=plot_formats)
        if write_plots
        else []
    )
    payload["plots"] = plot_paths
    artifacts = {
        "summary": output_dir / "summary.json",
        "seed_table": write_benchmark_packet_seed_table(output_dir, payload["seed_results"]),
        "packet_table": write_benchmark_packet_table(output_dir, payload),
        "fw_plateau_table": write_benchmark_packet_fw_plateau_table(output_dir, payload),
        "energy_stationarity_table": write_benchmark_packet_energy_stationarity_table(
            output_dir,
            payload,
        ),
        "density_fw_table": write_benchmark_packet_density_fw_table(output_dir, payload),
    }
    write_json(artifacts["summary"], payload)
    manifest_artifacts = [*artifacts.values(), *(output_dir / path for path in plot_paths)]
    artifacts["run_manifest"] = write_run_manifest(
        output_dir,
        run_name=BENCHMARK_PACKET_REANALYSIS_RUN_NAME,
        config={
            "case": payload["case_id"],
            "source_summary_sha256": payload["reanalysis"]["source"]["summary_sha256"],
            "source_manifest_sha256": payload["reanalysis"]["source"]["manifest_sha256"],
            "source_run_id": source_manifest["run_id"],
            "source_bundle_sha256": source_manifest["bundle_sha256"],
            "rms_relative_equivalence_margin": rms_relative_equivalence_margin,
            "confidence_level": confidence_level,
            "sensitivity_margins": list(sensitivity_margins),
            "policy_timing": policy_timing,
            "plot_formats": list(plot_formats),
            "write_plots": write_plots,
        },
        artifacts=manifest_artifacts,
        schema_version=BENCHMARK_PACKET_REANALYSIS_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=status,
    )
    return payload, artifacts


def write_benchmark_reanalysis_matrix(
    output_root: Path,
    *,
    source_root: Path,
    cases: tuple[str, ...],
    rows: list[dict[str, Any]],
    artifacts_by_case: dict[str, dict[str, Path]],
    rms_relative_equivalence_margin: float,
    confidence_level: float,
    sensitivity_margins: tuple[float, ...],
    policy_timing_by_case: dict[str, str],
    plot_formats: tuple[str, ...],
    write_plots: bool,
    command: list[str] | None,
) -> dict[str, Path]:
    """Bind per-case reanalyses into one recursively verifiable matrix artifact."""

    if tuple(row.get("case") for row in rows) != cases:
        raise ValueError("reanalysis matrix rows do not match the requested case order")
    if set(artifacts_by_case) != set(cases):
        raise ValueError("reanalysis matrix artifacts do not match the requested cases")
    if set(policy_timing_by_case) != set(cases):
        raise ValueError("reanalysis matrix policy timing does not match the requested cases")
    if any(
        policy_timing_by_case[case_id] not in {"prospective", "retrospective"} for case_id in cases
    ):
        raise ValueError("reanalysis matrix policy timing contains an invalid value")
    if any(row.get("policy_timing") != policy_timing_by_case[str(row.get("case"))] for row in rows):
        raise ValueError("reanalysis matrix row policy timing disagrees with its declaration")
    matrix_status = "accepted" if all(row.get("status") == "accepted" for row in rows) else "review"
    matrix_summary = output_root / "reanalysis_matrix.json"
    matrix_table = output_root / "reanalysis_matrix.csv"
    write_json(
        matrix_summary,
        {
            "schema_version": BENCHMARK_PACKET_REANALYSIS_MATRIX_SCHEMA_VERSION,
            "status": matrix_status,
            "source_root": str(source_root.resolve()),
            "output_root": str(output_root.resolve()),
            "rms_relative_equivalence_margin": rms_relative_equivalence_margin,
            "confidence_level": confidence_level,
            "sensitivity_margins": list(sensitivity_margins),
            "policy_timing_by_case": policy_timing_by_case,
            "rows": rows,
        },
    )
    _write_reanalysis_matrix_table(matrix_table, rows)
    matrix_manifest = write_run_manifest(
        output_root,
        run_name=BENCHMARK_PACKET_REANALYSIS_MATRIX_RUN_NAME,
        config={
            "source_root": str(source_root.resolve()),
            "cases": list(cases),
            "rms_relative_equivalence_margin": rms_relative_equivalence_margin,
            "confidence_level": confidence_level,
            "sensitivity_margins": list(sensitivity_margins),
            "policy_timing_by_case": policy_timing_by_case,
            "plot_formats": list(plot_formats),
            "write_plots": write_plots,
        },
        artifacts=[
            matrix_summary,
            matrix_table,
            *(artifacts_by_case[case_id]["run_manifest"] for case_id in cases),
        ],
        schema_version=BENCHMARK_PACKET_REANALYSIS_MATRIX_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=matrix_status,
    )
    verified, errors = verify_benchmark_reanalysis_matrix_manifest(matrix_manifest)
    if not verified:
        raise RuntimeError("written reanalysis matrix failed verification: " + "; ".join(errors))
    return {
        "matrix_summary": matrix_summary,
        "matrix_table": matrix_table,
        "matrix_manifest": matrix_manifest,
    }


def verify_benchmark_reanalysis_matrix_manifest(path: Path) -> tuple[bool, list[str]]:
    """Verify the parent manifest and every bound per-case child manifest."""

    verified, errors = verify_run_manifest(path)
    if not verified:
        return False, errors
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validation_errors: list[str] = []
    if manifest.get("run_name") != BENCHMARK_PACKET_REANALYSIS_MATRIX_RUN_NAME:
        validation_errors.append("matrix manifest has the wrong owner")
    if manifest.get("result_schema_version") != BENCHMARK_PACKET_REANALYSIS_MATRIX_SCHEMA_VERSION:
        validation_errors.append("matrix manifest has the wrong result schema")
    root = path.parent
    summary_path = root / "reanalysis_matrix.json"
    if not summary_path.is_file():
        validation_errors.append("matrix summary is missing")
        return False, validation_errors
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    cases = config.get("cases", []) if isinstance(config, dict) else []
    rows = summary.get("rows", [])
    row_cases = [row.get("case") for row in rows if isinstance(row, dict)]
    if summary.get("schema_version") != BENCHMARK_PACKET_REANALYSIS_MATRIX_SCHEMA_VERSION:
        validation_errors.append("matrix summary has the wrong schema")
    if summary.get("status") != manifest.get("status"):
        validation_errors.append("matrix summary and manifest statuses disagree")
    policy_timing_by_case = (
        config.get("policy_timing_by_case", {}) if isinstance(config, dict) else {}
    )
    if summary.get("policy_timing_by_case") != policy_timing_by_case:
        validation_errors.append("matrix policy timing declarations disagree")
    if not isinstance(cases, list) or row_cases != cases:
        validation_errors.append("matrix case identities disagree")
        return False, validation_errors
    artifact_paths = {
        str(entry.get("path")) for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    required_artifacts = {
        "reanalysis_matrix.json",
        "reanalysis_matrix.csv",
        *(f"{case_id}/run_manifest.json" for case_id in cases),
    }
    missing_artifacts = sorted(required_artifacts - artifact_paths)
    if missing_artifacts:
        validation_errors.append(
            "matrix manifest omits required artifacts: " + ", ".join(missing_artifacts)
        )
    row_status = {
        str(row["case"]): row.get("status")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("case"), str)
    }
    for case_id in cases:
        child_path = root / str(case_id) / "run_manifest.json"
        child_verified, child_errors = verify_run_manifest(child_path)
        if not child_verified:
            validation_errors.extend(f"{case_id}: {error}" for error in child_errors)
            continue
        child = json.loads(child_path.read_text(encoding="utf-8"))
        if child.get("run_name") != BENCHMARK_PACKET_REANALYSIS_RUN_NAME:
            validation_errors.append(f"{case_id}: child manifest has the wrong owner")
        if child.get("result_schema_version") != BENCHMARK_PACKET_REANALYSIS_SCHEMA_VERSION:
            validation_errors.append(f"{case_id}: child manifest has the wrong result schema")
        child_config = child.get("config", {})
        child_case = child_config.get("case") if isinstance(child_config, dict) else None
        if child_case != case_id:
            validation_errors.append(f"{case_id}: child manifest case identity disagrees")
        if child.get("status") != row_status.get(str(case_id)):
            validation_errors.append(f"{case_id}: child and matrix statuses disagree")
        child_timing = child_config.get("policy_timing") if isinstance(child_config, dict) else None
        if child_timing != policy_timing_by_case.get(case_id):
            validation_errors.append(f"{case_id}: child and matrix policy timing disagree")
    return not validation_errors, validation_errors


def _verified_source_manifest(path: Path) -> dict[str, Any]:
    verified, errors = verify_run_manifest(path)
    if not verified:
        raise ValueError("source run manifest verification failed: " + "; ".join(errors))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_name") != "dmc_benchmark_packet":
        raise ValueError("source manifest has the wrong artifact owner")
    if payload.get("result_schema_version") not in SOURCE_BENCHMARK_PACKET_SCHEMA_VERSIONS:
        raise ValueError("source manifest has the wrong result schema")
    if "summary.json" not in {
        entry.get("path") for entry in payload.get("artifacts", []) if isinstance(entry, dict)
    }:
        raise ValueError("source manifest does not bind summary.json")
    return payload


def _pure_config_from_payload(
    payload: dict[str, Any],
    *,
    rms_relative_equivalence_margin: float,
    confidence_level: float,
) -> PureWalkingConfig:
    metadata = payload["pure_config"]
    first_observables = payload["seed_results"][0]["pure_walking"]["observable_results"]

    def observable_metadata(name: str) -> dict[str, Any]:
        result = first_observables.get(name, {})
        value = result.get("metadata", {}) if isinstance(result, dict) else {}
        return value if isinstance(value, dict) else {}

    density_metadata = observable_metadata("density")
    pair_metadata = observable_metadata("pair_distance_density")
    structure_metadata = observable_metadata("structure_factor")
    config = PureWalkingConfig(
        lag_steps=tuple(int(value) for value in metadata["lag_steps"]),
        lag_unit=str(metadata.get("lag_unit", "dmc_steps")),
        observables=tuple(str(value) for value in metadata["observables"]),
        observable_source=str(metadata.get("observable_source", "raw_r2")),
        r2_rb_com_variance=_optional_finite_float(metadata.get("r2_rb_com_variance")),
        density_source=str(metadata.get("density_source", "raw_density")),
        density_com_variance=_optional_finite_float(metadata.get("density_com_variance")),
        density_parity_average=bool(metadata.get("density_parity_average", False)),
        density_expected_particles=_optional_finite_float(
            metadata.get("density_expected_particles")
        ),
        density_accounting_abs_tolerance=float(
            metadata.get("density_accounting_abs_tolerance", 5.0e-3)
        ),
        density_bin_edges=_optional_array(density_metadata.get("bin_edges")),
        pair_bin_edges=_optional_array(pair_metadata.get("bin_edges")),
        structure_k_values=_optional_array(structure_metadata.get("k_values")),
        min_block_count=int(metadata["min_block_count"]),
        min_walker_weight_ess=float(metadata["min_walker_weight_ess"]),
        min_source_ancestor_ess=float(metadata["min_source_ancestor_ess"]),
        max_source_family_fraction=float(metadata["max_source_family_fraction"]),
        block_size_steps=int(metadata["block_size_steps"]),
        collection_stride_steps=int(metadata["collection_stride_steps"]),
        transport_mode=str(metadata.get("transport_mode", "post_resample_auxiliary")),
        collection_mode=str(metadata.get("collection_mode", "sliding_window")),
        center=float(metadata.get("center", 0.0)),
        plateau_sigma_threshold=float(metadata.get("plateau_sigma_threshold", 1.0)),
        plateau_abs_tolerance=float(metadata.get("plateau_abs_tolerance", 0.0)),
        rms_plateau_relative_tolerance=rms_relative_equivalence_margin,
        plateau_equivalence_confidence_level=confidence_level,
        plateau_window_lag_count=int(metadata.get("plateau_window_lag_count", 4)),
        density_lag_steps=(
            None
            if metadata.get("density_lag_steps") is None
            else tuple(int(value) for value in metadata["density_lag_steps"])
        ),
        density_collection_stride_steps=(
            None
            if metadata.get("density_collection_stride_steps") is None
            else int(metadata["density_collection_stride_steps"])
        ),
        density_plateau_window_lag_count=(
            None
            if metadata.get("density_plateau_window_lag_count") is None
            else int(metadata["density_plateau_window_lag_count"])
        ),
        density_plateau_relative_l2_tolerance=float(
            metadata.get("density_plateau_relative_l2_tolerance", 0.03)
        ),
        transport_invariant_tests_passed=tuple(
            str(value) for value in metadata.get("transport_invariant_tests_passed", ())
        ),
    )
    config.validate()
    return config


def _validate_source_identity(
    *,
    source_dir: Path,
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    summary_case = summary.get("case_id")
    manifest_config = manifest.get("config", {})
    manifest_case = manifest_config.get("case") if isinstance(manifest_config, dict) else None
    if not isinstance(summary_case, str):
        raise ValueError("source summary has no case identity")
    if source_dir.name != summary_case:
        raise ValueError("source directory and summary case identities disagree")
    if output_dir.name != summary_case:
        raise ValueError("output directory and summary case identities disagree")
    if manifest_case != summary_case:
        raise ValueError("source manifest and summary case identities disagree")
    summary_seeds = summary.get("seeds")
    manifest_seeds = manifest_config.get("seeds") if isinstance(manifest_config, dict) else None
    if not isinstance(summary_seeds, list) or manifest_seeds != summary_seeds:
        raise ValueError("source manifest and summary seed identities disagree")


def _assert_unmodified_observables(
    *,
    source_payload: dict[str, Any],
    pure_summary: dict[str, Any],
    recomputed_estimates: dict[str, Any],
) -> None:
    source_estimates = source_payload.get("estimates", {})
    if not isinstance(source_estimates, dict):
        raise ValueError("source summary has no estimate payload")
    for name in ("energy", "pair_distance_density", "structure_factor"):
        if not _semantic_json_equal(recomputed_estimates.get(name), source_estimates.get(name)):
            raise ValueError(f"offline R2 reanalysis changed the stored {name} estimate")
    source_density = source_estimates.get("density")
    derived_density = recomputed_estimates.get("density")
    if not isinstance(source_density, dict) or not isinstance(derived_density, dict):
        raise ValueError("source or derived density estimate is invalid")
    added_density_fields = {
        "fw_relative_l2_vs_lda",
        "fw_relative_l2_vs_mixed",
        "fw_lag_systematic_relative_l2_upper_bound",
        "fw_lag_relative_l2_equivalence_margin",
        "fw_lag_equivalence_confidence_level",
        "comparison_semantics",
    }
    if not _semantic_json_equal(
        {key: value for key, value in derived_density.items() if key not in added_density_fields},
        {key: value for key, value in source_density.items() if key not in added_density_fields},
    ):
        raise ValueError("offline R2 reanalysis changed the stored density estimate")
    source_observables = source_payload.get("pure_walking", {}).get("observables", {})
    derived_observables = pure_summary.get("observables", {})
    if not isinstance(source_observables, dict) or not isinstance(derived_observables, dict):
        raise ValueError("source or derived pure-observable payload is invalid")
    for name in set(source_observables) - {"r2", "density"}:
        if not _semantic_json_equal(
            derived_observables.get(name),
            source_observables.get(name),
        ):
            raise ValueError(f"offline R2 reanalysis changed the stored {name} pure summary")
    source_density_observable = source_observables.get("density")
    derived_density_observable = derived_observables.get("density")
    if isinstance(source_density_observable, dict) and isinstance(
        derived_density_observable,
        dict,
    ):
        preserved_density_fields = {
            "value",
            "stderr",
            "metadata",
            "bin_edges",
            "integral",
            "x",
            "schema_statuses",
            "genealogy_statuses",
            "genealogy_diagnostics_by_seed",
        }
        if not _semantic_json_equal(
            {key: source_density_observable.get(key) for key in preserved_density_fields},
            {key: derived_density_observable.get(key) for key in preserved_density_fields},
        ):
            raise ValueError("offline equivalence analysis changed stored density values")


def _semantic_json_equal(first: object, second: object) -> bool:
    return json.loads(json.dumps(to_jsonable(first), sort_keys=True)) == json.loads(
        json.dumps(to_jsonable(second), sort_keys=True)
    )


def _write_reanalysis_matrix_table(
    path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    if not rows:
        raise ValueError("reanalysis matrix table requires at least one row")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("reanalysis matrix rows have inconsistent fields")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_reanalysis_controls(
    *,
    rms_relative_equivalence_margin: float,
    confidence_level: float,
    sensitivity_margins: tuple[float, ...],
    policy_timing: str,
) -> None:
    if not np.isfinite(rms_relative_equivalence_margin) or rms_relative_equivalence_margin < 0.0:
        raise ValueError("RMS relative equivalence margin must be finite and non-negative")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    if not sensitivity_margins or any(
        not np.isfinite(margin) or margin < 0.0 for margin in sensitivity_margins
    ):
        raise ValueError("sensitivity margins must be finite and non-negative")
    if policy_timing not in {"prospective", "retrospective"}:
        raise ValueError("policy_timing must be prospective or retrospective")


def _optional_finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if np.all(np.isfinite(array)) else None
