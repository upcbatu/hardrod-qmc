from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.analysis import assess_matrix_energy_stationarity
from hrdmc.artifacts import (
    build_run_provenance,
    file_sha256,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.workflows.dmc.final_matrix import DEFAULT_CASE_ORDER

FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION = "dmc_final_matrix_assembly_v1"
FINAL_MATRIX_ASSEMBLY_RUN_NAME = "dmc_final_matrix_assembly"
BENCHMARK_PACKET_SCHEMA_VERSIONS = {"dmc_benchmark_packet_v2", "dmc_benchmark_packet_v3"}
REQUIRED_CASE_ORDER = DEFAULT_CASE_ORDER
FINAL_MATRIX_INTERPRETATION = {
    "energy": "mixed DMC local-energy estimator",
    "r2": "transported auxiliary forward-walking pure estimator",
    "rms_radius": "square root of seed-aggregated pure R2",
    "density": "transported auxiliary forward-walking pure estimator",
    "lda_comparisons": "descriptive finite-system versus smooth-LDA differences",
}


def assemble_final_benchmark_matrix(
    source_root: Path,
    output_root: Path,
    *,
    cases: Sequence[str] = REQUIRED_CASE_ORDER,
    r2_supplements: Mapping[str, Path] | None = None,
    retrospective_energy_cases: Sequence[str] = (),
    energy_confidence_level: float = 0.95,
    energy_rhat_limit: float = 1.01,
    energy_min_effective_samples: float = 400.0,
    command: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Assemble verified benchmark packets without mutating their source artifacts."""

    requested_cases = tuple(cases)
    if requested_cases != REQUIRED_CASE_ORDER:
        raise ValueError("final benchmark assembly requires the canonical eight-case order")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"final matrix output directory is not empty: {output_root}")
    supplements = dict(r2_supplements or {})
    retrospective = set(retrospective_energy_cases)
    unknown = (set(supplements) | retrospective) - set(requested_cases)
    if unknown:
        raise ValueError("unknown case ids in assembly controls: " + ", ".join(sorted(unknown)))

    primary_sources = {
        case_id: _load_verified_packet(source_root / case_id) for case_id in requested_cases
    }
    supplement_sources = {
        case_id: _load_verified_packet(path) for case_id, path in supplements.items()
    }
    for case_id, supplement in supplement_sources.items():
        _validate_r2_supplement(primary_sources[case_id], supplement)

    energy_assessment = _assess_source_matrix_energy(
        primary_sources,
        confidence_level=energy_confidence_level,
        rhat_limit=energy_rhat_limit,
        min_effective_samples=energy_min_effective_samples,
    )

    rows = [
        _assemble_row(
            case_id,
            primary=primary_sources[case_id],
            r2_source=supplement_sources.get(case_id, primary_sources[case_id]),
            energy_assessment=energy_assessment,
            retrospective_energy_cases=retrospective,
            reference_root=output_root,
        )
        for case_id in requested_cases
    ]
    status = "accepted" if all(row["status"] == "accepted" for row in rows) else "review"
    payload = {
        "schema_version": FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION,
        "status": status,
        "case_order": list(requested_cases),
        "source_root": _relative_locator(source_root.resolve(), output_root.resolve()),
        "source_locator_base": "assembly_directory",
        "energy_stationarity_assessment": energy_assessment,
        "retrospective_energy_cases": sorted(retrospective),
        "rows": rows,
        "interpretation": dict(FINAL_MATRIX_INTERPRETATION),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "final_matrix_summary.json"
    table_path = output_root / "final_matrix_table.csv"
    write_json(summary_path, payload)
    _write_matrix_table(table_path, rows)
    source_config = {
        case_id: {
            "primary": _source_reference(primary_sources[case_id], reference_root=output_root),
            "r2": _source_reference(
                supplement_sources.get(case_id, primary_sources[case_id]),
                reference_root=output_root,
            ),
        }
        for case_id in requested_cases
    }
    manifest_path = write_run_manifest(
        output_root,
        run_name=FINAL_MATRIX_ASSEMBLY_RUN_NAME,
        config={
            "case_order": list(requested_cases),
            "source_locator_base": "assembly_directory",
            "sources": source_config,
            "retrospective_energy_cases": sorted(retrospective),
            "energy_confidence_level": energy_confidence_level,
            "energy_rhat_limit": energy_rhat_limit,
            "energy_min_effective_samples": energy_min_effective_samples,
        },
        artifacts=[summary_path, table_path],
        schema_version=FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=status,
    )
    verified, errors = verify_final_benchmark_matrix_manifest(manifest_path)
    if not verified:
        raise RuntimeError("written final matrix failed verification: " + "; ".join(errors))
    return payload, {
        "summary": summary_path,
        "table": table_path,
        "run_manifest": manifest_path,
    }


def _assess_source_matrix_energy(
    primary_sources: Mapping[str, dict[str, Any]],
    *,
    confidence_level: float,
    rhat_limit: float,
    min_effective_samples: float,
) -> dict[str, Any]:
    assessment = assess_matrix_energy_stationarity(
        {
            case_id: _mapping(source["summary"].get("stationarity"))
            for case_id, source in primary_sources.items()
        },
        confidence_level=confidence_level,
        rhat_limit=rhat_limit,
        min_effective_samples=min_effective_samples,
    )
    assessment["policy_timing"] = "retrospective"
    assessment["scope"] = "canonical_eight_case_final_matrix"
    return assessment


def verify_final_benchmark_matrix_manifest(path: Path) -> tuple[bool, list[str]]:
    try:
        verified, errors = verify_run_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"final matrix manifest is unreadable: {exc}"]
    if not verified:
        return False, errors
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    validation_errors: list[str] = []
    if manifest.get("run_name") != FINAL_MATRIX_ASSEMBLY_RUN_NAME:
        validation_errors.append("final matrix manifest has the wrong owner")
    if manifest.get("result_schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        validation_errors.append("final matrix manifest has the wrong result schema")
    artifact_paths = {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if artifact_paths != {"final_matrix_summary.json", "final_matrix_table.csv"}:
        validation_errors.append("final matrix manifest has the wrong artifact set")
    summary_path = root / "final_matrix_summary.json"
    table_path = root / "final_matrix_table.csv"
    if not summary_path.is_file():
        return False, ["final matrix summary is missing"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"final matrix summary is unreadable: {exc}"]
    if summary.get("schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        validation_errors.append("final matrix summary has the wrong result schema")
    config = _mapping(manifest.get("config"))
    case_order = config.get("case_order")
    rows = summary.get("rows")
    if case_order != list(REQUIRED_CASE_ORDER):
        validation_errors.append("final matrix manifest has the wrong case order")
    if summary.get("case_order") != case_order:
        validation_errors.append("final matrix summary has the wrong case order")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or [row.get("case") for row in rows] != case_order
    ):
        validation_errors.append("final matrix case identities disagree")
    if summary.get("status") != manifest.get("status"):
        validation_errors.append("final matrix summary and manifest statuses disagree")
    retrospective_cases = config.get("retrospective_energy_cases")
    if summary.get("retrospective_energy_cases") != retrospective_cases:
        validation_errors.append("final matrix retrospective energy declarations disagree")
    if (
        not isinstance(retrospective_cases, list)
        or len(retrospective_cases) != len(set(retrospective_cases))
        or not set(retrospective_cases).issubset(REQUIRED_CASE_ORDER)
    ):
        validation_errors.append("final matrix retrospective energy cases are invalid")
        retrospective_cases = []
    if config.get("source_locator_base") != "assembly_directory":
        validation_errors.append("final matrix manifest has the wrong source locator base")
    if summary.get("source_locator_base") != "assembly_directory":
        validation_errors.append("final matrix summary has the wrong source locator base")
    sources = config.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_CASE_ORDER):
        validation_errors.append("final matrix source identities are incomplete")
        return False, validation_errors
    primary_sources: dict[str, dict[str, Any]] = {}
    r2_sources: dict[str, dict[str, Any]] = {}
    for case_id in REQUIRED_CASE_ORDER:
        case_sources = sources.get(case_id)
        if not isinstance(case_sources, dict):
            validation_errors.append(f"{case_id}: source declaration is invalid")
            continue
        for observable_source in ("primary", "r2"):
            reference = case_sources.get(observable_source)
            reference_errors = _verify_source_reference(
                root,
                case_id,
                observable_source,
                reference,
            )
            validation_errors.extend(reference_errors)
            if reference_errors or not isinstance(reference, dict):
                continue
            manifest_path = _resolve_reference_path(root, reference.get("manifest_path"))
            try:
                loaded = _load_verified_packet(manifest_path.parent)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                validation_errors.append(f"{case_id} {observable_source} source: {exc}")
                continue
            if observable_source == "primary":
                primary_sources[case_id] = loaded
            else:
                r2_sources[case_id] = loaded

    if set(primary_sources) != set(REQUIRED_CASE_ORDER) or set(r2_sources) != set(
        REQUIRED_CASE_ORDER
    ):
        return False, validation_errors

    for case_id in REQUIRED_CASE_ORDER:
        try:
            _validate_r2_supplement(primary_sources[case_id], r2_sources[case_id])
        except ValueError as exc:
            validation_errors.append(str(exc))

    try:
        energy_assessment = _assess_source_matrix_energy(
            primary_sources,
            confidence_level=_required_config_float(config, "energy_confidence_level"),
            rhat_limit=_required_config_float(config, "energy_rhat_limit"),
            min_effective_samples=_required_config_float(
                config,
                "energy_min_effective_samples",
            ),
        )
    except (TypeError, ValueError) as exc:
        validation_errors.append(f"final matrix energy controls are invalid: {exc}")
        return False, validation_errors
    if not _semantic_equal(summary.get("energy_stationarity_assessment"), energy_assessment):
        validation_errors.append("final matrix energy assessment disagrees with its sources")

    expected_rows = [
        _assemble_row(
            case_id,
            primary=primary_sources[case_id],
            r2_source=r2_sources[case_id],
            energy_assessment=energy_assessment,
            retrospective_energy_cases=set(retrospective_cases),
            reference_root=root,
        )
        for case_id in REQUIRED_CASE_ORDER
    ]
    if not _semantic_equal(rows, expected_rows):
        validation_errors.append("final matrix case records disagree with their sources")
    expected_status = (
        "accepted"
        if energy_assessment.get("status") == "accepted"
        and all(row["status"] == "accepted" for row in expected_rows)
        else "review"
    )
    if summary.get("status") != expected_status:
        validation_errors.append("final matrix status disagrees with its case records")
    if summary.get("interpretation") != FINAL_MATRIX_INTERPRETATION:
        validation_errors.append("final matrix estimator interpretation disagrees")
    source_parents = {source["directory"].parent.resolve() for source in primary_sources.values()}
    if len(source_parents) != 1:
        validation_errors.append("final matrix primary sources do not share one source root")
    else:
        expected_source_root = _relative_locator(next(iter(source_parents)), root)
        if summary.get("source_root") != expected_source_root:
            validation_errors.append("final matrix source root disagrees with its sources")
    if not table_path.is_file():
        validation_errors.append("final matrix table is missing")
    else:
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            table_text = handle.read()
        if table_text != _matrix_table_text(expected_rows):
            validation_errors.append("final matrix CSV disagrees with its case records")
    return not validation_errors, validation_errors


def _assemble_row(
    case_id: str,
    *,
    primary: dict[str, Any],
    r2_source: dict[str, Any],
    energy_assessment: dict[str, Any],
    retrospective_energy_cases: set[str],
    reference_root: Path,
) -> dict[str, Any]:
    primary_summary = primary["summary"]
    r2_summary = r2_source["summary"]
    primary_estimates = _mapping(primary_summary.get("estimates"))
    r2_estimates = _mapping(r2_summary.get("estimates"))
    energy = _mapping(primary_estimates.get("energy"))
    r2 = _mapping(r2_estimates.get("r2"))
    rms = _mapping(r2_estimates.get("rms"))
    density = _mapping(primary_estimates.get("density"))
    source_energy_status = str(primary_summary.get("energy_validation_status", ""))
    matrix_energy = _mapping(_mapping(energy_assessment.get("cases")).get(case_id))
    if matrix_energy.get("status") != "accepted":
        energy_status = "review"
        energy_status_basis = "matrix_stationarity_assessment"
    elif source_energy_status == "accepted":
        energy_status = "accepted"
        energy_status_basis = "source_packet"
    elif case_id in retrospective_energy_cases and matrix_energy.get("status") == "accepted":
        energy_status = "accepted"
        energy_status_basis = "retrospective_matrix_stationarity_assessment"
    else:
        energy_status = source_energy_status or "review"
        energy_status_basis = "source_packet"
    r2_status = str(r2.get("status", "not_evaluated"))
    density_status = str(density.get("status", "not_evaluated"))
    status = "accepted" if energy_status == r2_status == density_status == "accepted" else "review"
    controls = _mapping(primary_summary.get("controls"))
    r2_fw = _mapping(r2_summary.get("pure_walking"))
    density_fw = _mapping(primary_summary.get("pure_walking"))
    return {
        "case": case_id,
        "status": status,
        "energy_status": energy_status,
        "energy_status_basis": energy_status_basis,
        "source_energy_status": source_energy_status,
        "source_energy_stationarity_reason": _mapping(primary_summary.get("stationarity")).get(
            "stationarity_reason_energy"
        ),
        "r2_status": r2_status,
        "density_status": density_status,
        "guide_family": primary_summary.get("guide_family"),
        "relative_alpha": controls.get("relative_alpha"),
        "contact_beta": controls.get("contact_beta"),
        "local_step_method": controls.get("local_step_method"),
        "drift_limiter": controls.get("drift_limiter"),
        "dt": controls.get("dt"),
        "walkers": controls.get("walkers"),
        "burn_tau": controls.get("burn_tau"),
        "production_tau": controls.get("production_tau"),
        "grid_extent": controls.get("grid_extent"),
        "n_bins": controls.get("n_bins"),
        "store_every": controls.get("store_every"),
        "seeds": primary_summary.get("seeds"),
        "energy": energy.get("value"),
        "energy_stderr": energy.get("stderr"),
        "energy_lda": energy.get("lda_value"),
        "energy_relative_delta_vs_lda": _relative_delta(energy),
        "r2": r2.get("value"),
        "r2_stderr": r2.get("stderr"),
        "r2_lda": r2.get("lda_value"),
        "rms_radius": rms.get("value"),
        "rms_mc_statistical_stderr": rms.get("mc_statistical_stderr"),
        "rms_lda": rms.get("lda_value"),
        "rms_fw_lag_systematic_relative_upper_bound": rms.get(
            "fw_lag_systematic_relative_upper_bound"
        ),
        "rms_relative_delta_vs_lda": _relative_delta(rms),
        "density_fw_relative_l2_vs_lda": density.get("fw_relative_l2_vs_lda"),
        "density_fw_relative_l2_vs_mixed": density.get("fw_relative_l2_vs_mixed"),
        "density_fw_lag_systematic_relative_l2_upper_bound": density.get(
            "fw_lag_systematic_relative_l2_upper_bound"
        ),
        "r2_selected_window_lags": _mapping(r2_fw.get("r2_aggregate_plateau_diagnostics")).get(
            "selected_window_lags"
        ),
        "r2_pooled_ancestor_ess_lower_min": _mapping(
            r2_fw.get("r2_aggregate_plateau_diagnostics")
        ).get("selected_window_pooled_ancestor_ess_lower_min"),
        "r2_seed_plateau_resolved_count": r2_fw.get("r2_seed_plateau_resolved_count"),
        "r2_seed_plateau_unresolved_count": r2_fw.get("r2_seed_plateau_unresolved_count"),
        "density_selected_window_lags": _mapping(
            _mapping(_mapping(density_fw.get("observables")).get("density")).get(
                "aggregate_plateau_diagnostics"
            )
        ).get("selected_window_lags"),
        "primary_source": _source_reference(primary, reference_root=reference_root),
        "r2_source": _source_reference(r2_source, reference_root=reference_root),
    }


def _load_verified_packet(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "run_manifest.json"
    summary_path = source_dir / "summary.json"
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError(
            f"source manifest verification failed for {source_dir}: " + "; ".join(errors)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != "dmc_benchmark_packet":
        raise ValueError(f"source packet has the wrong owner: {source_dir}")
    if manifest.get("result_schema_version") not in BENCHMARK_PACKET_SCHEMA_VERSIONS:
        raise ValueError(f"source packet has an unsupported schema: {source_dir}")
    if summary.get("schema_version") not in BENCHMARK_PACKET_SCHEMA_VERSIONS:
        raise ValueError(f"source summary has an unsupported schema: {source_dir}")
    case_id = summary.get("case_id")
    config = _mapping(manifest.get("config"))
    if not isinstance(case_id, str) or config.get("case") != case_id:
        raise ValueError(f"source packet case identity disagrees: {source_dir}")
    if "summary.json" not in {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }:
        raise ValueError(f"source packet manifest does not bind summary.json: {source_dir}")
    return {
        "directory": source_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "summary_path": summary_path.resolve(),
        "manifest": manifest,
        "summary": summary,
    }


def _validate_r2_supplement(primary: dict[str, Any], supplement: dict[str, Any]) -> None:
    first = primary["summary"]
    second = supplement["summary"]
    case_id = str(first.get("case_id"))
    if second.get("case_id") != case_id:
        raise ValueError(f"{case_id}: R2 supplement has the wrong case identity")
    matching_fields = (
        "seeds",
        "n_particles",
        "rod_length_ho",
        "guide_family",
        "guide_parameters",
        "controls",
    )
    for field in matching_fields:
        if not _semantic_equal(first.get(field), second.get(field)):
            raise ValueError(f"{case_id}: R2 supplement disagrees on {field}")
    first_impl = _mapping(_mapping(primary["manifest"].get("provenance")).get("implementation"))
    second_impl = _mapping(_mapping(supplement["manifest"].get("provenance")).get("implementation"))
    if first_impl.get("source_tree_sha256") != second_impl.get("source_tree_sha256"):
        raise ValueError(f"{case_id}: R2 supplement used a different implementation tree")
    first_energy = _mapping(_mapping(first.get("estimates")).get("energy"))
    second_energy = _mapping(_mapping(second.get("estimates")).get("energy"))
    if not _semantic_equal(first_energy, second_energy):
        raise ValueError(f"{case_id}: R2 supplement did not reproduce the primary energy")
    if _mapping(_mapping(second.get("estimates")).get("r2")).get("status") != "accepted":
        raise ValueError(f"{case_id}: R2 supplement is not accepted")


def _source_reference(
    source: dict[str, Any],
    *,
    reference_root: Path,
) -> dict[str, Any]:
    manifest = source["manifest"]
    return {
        "directory": _relative_locator(source["directory"], reference_root),
        "summary_path": _relative_locator(source["summary_path"], reference_root),
        "summary_sha256": file_sha256(source["summary_path"]),
        "manifest_path": _relative_locator(source["manifest_path"], reference_root),
        "manifest_sha256": file_sha256(source["manifest_path"]),
        "run_id": manifest.get("run_id"),
        "bundle_sha256": manifest.get("bundle_sha256"),
    }


def _verify_source_reference(
    reference_root: Path,
    case_id: str,
    observable_source: str,
    reference: object,
) -> list[str]:
    prefix = f"{case_id} {observable_source} source"
    if not isinstance(reference, dict):
        return [f"{prefix}: reference is invalid"]
    directory = _resolve_reference_path(reference_root, reference.get("directory"))
    manifest_path = _resolve_reference_path(reference_root, reference.get("manifest_path"))
    summary_path = _resolve_reference_path(reference_root, reference.get("summary_path"))
    errors: list[str] = []
    if manifest_path != directory / "run_manifest.json":
        errors.append(f"{prefix}: manifest path does not match the source directory")
    if summary_path != directory / "summary.json":
        errors.append(f"{prefix}: summary path does not match the source directory")
    manifest_identity_matches = manifest_path.is_file() and (
        file_sha256(manifest_path) == reference.get("manifest_sha256")
    )
    if not manifest_identity_matches:
        errors.append(f"{prefix}: manifest identity mismatch")
        return errors
    verified, manifest_errors = verify_run_manifest(manifest_path)
    if not verified:
        errors.extend(f"{prefix}: {error}" for error in manifest_errors)
        return errors
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _mapping(manifest.get("config"))
    if config.get("case") != case_id:
        errors.append(f"{prefix}: case identity mismatch")
    if manifest.get("run_id") != reference.get("run_id"):
        errors.append(f"{prefix}: run identity mismatch")
    if manifest.get("bundle_sha256") != reference.get("bundle_sha256"):
        errors.append(f"{prefix}: bundle identity mismatch")
    if not summary_path.is_file() or file_sha256(summary_path) != reference.get("summary_sha256"):
        errors.append(f"{prefix}: summary identity mismatch")
    return errors


def _write_matrix_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = _matrix_table_text(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _matrix_table_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("final matrix table requires at least one case")
    fields = [key for key in rows[0] if key not in {"primary_source", "r2_source"}]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _required_config_float(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} is not numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _relative_locator(path: Path, reference_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=reference_root.resolve())).as_posix()


def _resolve_reference_path(reference_root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return (path if path.is_absolute() else reference_root / path).resolve()


def _relative_delta(estimate: dict[str, Any]) -> float | None:
    delta = estimate.get("delta_vs_lda")
    reference = estimate.get("lda_value")
    if not isinstance(delta, (int, float)) or not isinstance(reference, (int, float)):
        return None
    if not np.isfinite(delta) or not np.isfinite(reference) or reference == 0.0:
        return None
    return float(delta / reference)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _semantic_equal(first: object, second: object) -> bool:
    return json.dumps(first, sort_keys=True, allow_nan=True) == json.dumps(
        second, sort_keys=True, allow_nan=True
    )
