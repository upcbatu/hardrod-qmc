from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.artifacts.manifest import (
    csv_text,
    file_sha256,
    verify_run_manifest,
    write_csv,
    write_json,
    write_run_manifest,
)
from hrdmc.production.matrix.sources import (
    load_energy_assessment_packet as _load_energy_assessment_packet,
)
from hrdmc.production.matrix.sources import (
    load_verified_packet as _load_verified_packet,
)
from hrdmc.production.matrix.sources import (
    mapping as _mapping,
)
from hrdmc.production.matrix.sources import (
    relative_locator as _relative_locator,
)
from hrdmc.production.matrix.sources import (
    required_config_float as _required_config_float,
)
from hrdmc.production.matrix.sources import (
    resolve_reference_path as _resolve_reference_path,
)
from hrdmc.production.matrix.sources import (
    semantic_equal as _semantic_equal,
)
from hrdmc.production.matrix.sources import (
    source_reference as _source_reference,
)
from hrdmc.production.matrix.sources import (
    validate_r2_supplement as _validate_r2_supplement,
)
from hrdmc.production.matrix.sources import (
    verify_source_reference as _verify_source_reference,
)
from hrdmc.statistics.equilibration import assess_matrix_energy_stationarity
from hrdmc.system.settings import THESIS_CASE_ORDER

FINAL_MATRIX_ASSEMBLY_RUN_NAME = "dmc_final_matrix_assembly"
REQUIRED_CASE_ORDER = THESIS_CASE_ORDER
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
    write_csv(table_path, rows, exclude=("primary_source", "r2_source"))
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
        status=status,
    )
    verified, errors = verify_final_benchmark_matrix_manifest(manifest_path)
    if not verified:
        raise RuntimeError("written final matrix failed verification: " + "; ".join(errors))
    return (payload, {"summary": summary_path, "table": table_path, "run_manifest": manifest_path})


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


def load_final_matrix_energy_selection(
    assembly_manifest_path: Path, *, case_id: str
) -> dict[str, Any]:
    path = assembly_manifest_path.resolve()
    verified, errors = verify_run_manifest(path)
    if not verified:
        raise ValueError("energy-assessment manifest verification failed: " + "; ".join(errors))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    summary_path = root / "final_matrix_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = _mapping(manifest.get("config"))
    if manifest.get("run_name") != FINAL_MATRIX_ASSEMBLY_RUN_NAME:
        raise ValueError("energy assessment has the wrong owner")
    if config.get("case_order") != list(REQUIRED_CASE_ORDER) or case_id not in REQUIRED_CASE_ORDER:
        raise ValueError(f"energy assessment does not support case {case_id}")
    sources = _mapping(config.get("sources"))
    if set(sources) != set(REQUIRED_CASE_ORDER):
        raise ValueError("energy-assessment primary sources are incomplete")
    primary: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for source_case in REQUIRED_CASE_ORDER:
        source, drift = _load_energy_assessment_packet(
            root, source_case, _mapping(sources.get(source_case)).get("primary")
        )
        primary[source_case] = source
        warnings.extend(f"{source_case}: {item}" for item in drift)
    energy_assessment = _assess_source_matrix_energy(
        primary,
        confidence_level=_required_config_float(config, "energy_confidence_level"),
        rhat_limit=_required_config_float(config, "energy_rhat_limit"),
        min_effective_samples=_required_config_float(config, "energy_min_effective_samples"),
    )
    if not _semantic_equal(summary.get("energy_stationarity_assessment"), energy_assessment):
        raise ValueError("energy assessment disagrees with its exact primary summaries")
    rows = cast(list[dict[str, Any]], summary["rows"])
    row = next(item for item in rows if item.get("case") == case_id)
    source = primary[case_id]
    expected = _energy_status_record(
        case_id,
        primary_summary=source["summary"],
        energy_assessment=energy_assessment,
        retrospective_energy_cases=set(config.get("retrospective_energy_cases", [])),
    )
    for field, value in expected.items():
        if not _semantic_equal(row.get(field), value):
            raise ValueError(f"{case_id}: selected energy {field} disagrees")
    reference = _source_reference(source, reference_root=root)
    if not _semantic_equal(row.get("primary_source"), reference):
        raise ValueError(f"{case_id}: selected energy source locator disagrees")
    energy_status = str(expected["energy_status"])
    status_basis = str(expected["energy_status_basis"])
    publication_status = (
        "accepted_with_retrospective_assessment"
        if energy_status == "accepted"
        and status_basis == "retrospective_matrix_stationarity_assessment"
        else "accepted"
        if energy_status == "accepted"
        else "unresolved"
    )
    source_manifest = source["manifest"]
    return {
        "verification_scope": "energy_assessment_and_selected_primary_summary",
        "case_id": case_id,
        "publication_accepted": energy_status == "accepted",
        "publication_status": publication_status,
        "energy_status": energy_status,
        "energy_status_basis": status_basis,
        "source_energy_status": expected["source_energy_status"],
        "source_energy_stationarity_reason": expected["source_energy_stationarity_reason"],
        "policy_timing": energy_assessment.get("policy_timing"),
        "assessment_scope": energy_assessment.get("scope"),
        "assessment_method": energy_assessment.get("method"),
        "case_assessment": _mapping(_mapping(energy_assessment.get("cases")).get(case_id)),
        "assessment_manifest_path": str(path),
        "assessment_manifest_sha256": file_sha256(path),
        "assessment_summary_path": str(summary_path),
        "assessment_summary_sha256": file_sha256(summary_path),
        "assessment_run_id": manifest.get("run_id"),
        "selected_summary_path": str(source["summary_path"]),
        "selected_summary_sha256": file_sha256(source["summary_path"]),
        "selected_manifest_path": str(source["manifest_path"]),
        "selected_manifest_sha256": file_sha256(source["manifest_path"]),
        "selected_run_id": source_manifest.get("run_id"),
        "source_plot_artifact_warnings": warnings,
    }


def verify_final_benchmark_matrix_manifest(path: Path) -> tuple[bool, list[str]]:
    try:
        verified, errors = verify_run_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (False, [f"final matrix manifest is unreadable: {exc}"])
    if not verified:
        return (False, errors)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    summary_path = root / "final_matrix_summary.json"
    if not summary_path.is_file():
        return (False, ["final matrix summary is missing"])
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (False, [f"final matrix summary is unreadable: {exc}"])
    config = _mapping(manifest.get("config"))
    rows = summary.get("rows")
    validation_errors = _matrix_declaration_errors(manifest, summary, config, rows)
    retrospective_cases = _retrospective_cases(config, summary, validation_errors)
    sources = config.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_CASE_ORDER):
        validation_errors.append("final matrix source identities are incomplete")
        return (False, validation_errors)
    primary_sources, r2_sources = _load_matrix_sources(root, sources, validation_errors)
    if set(primary_sources) != set(REQUIRED_CASE_ORDER) or set(r2_sources) != set(
        REQUIRED_CASE_ORDER
    ):
        return (False, validation_errors)
    _validate_matrix_supplements(primary_sources, r2_sources, validation_errors)
    try:
        energy_assessment = _assess_source_matrix_energy(
            primary_sources,
            confidence_level=_required_config_float(config, "energy_confidence_level"),
            rhat_limit=_required_config_float(config, "energy_rhat_limit"),
            min_effective_samples=_required_config_float(config, "energy_min_effective_samples"),
        )
    except (TypeError, ValueError) as exc:
        validation_errors.append(f"final matrix energy controls are invalid: {exc}")
        return (False, validation_errors)
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
    _validate_reconstructed_matrix(
        root=root,
        manifest=manifest,
        summary=summary,
        rows=rows,
        energy_assessment=energy_assessment,
        expected_rows=expected_rows,
        primary_sources=primary_sources,
        errors=validation_errors,
    )
    return (not validation_errors, validation_errors)


def _matrix_declaration_errors(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    config: dict[str, Any],
    rows: object,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("run_name") != FINAL_MATRIX_ASSEMBLY_RUN_NAME:
        errors.append("final matrix manifest has the wrong owner")
    artifacts = {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if artifacts != {"final_matrix_summary.json", "final_matrix_table.csv"}:
        errors.append("final matrix manifest has the wrong artifact set")
    case_order = config.get("case_order")
    if case_order != list(REQUIRED_CASE_ORDER):
        errors.append("final matrix manifest has the wrong case order")
    if summary.get("case_order") != case_order:
        errors.append("final matrix summary has the wrong case order")
    row_list = rows if isinstance(rows, list) else []
    valid_rows = bool(row_list) and all(isinstance(row, dict) for row in row_list)
    row_cases = [row.get("case") for row in row_list if isinstance(row, dict)]
    if not valid_rows or row_cases != case_order:
        errors.append("final matrix case identities disagree")
    if summary.get("status") != manifest.get("status"):
        errors.append("final matrix summary and manifest statuses disagree")
    for owner, value in (("manifest", config), ("summary", summary)):
        if value.get("source_locator_base") != "assembly_directory":
            errors.append(f"final matrix {owner} has the wrong source locator base")
    return errors


def _retrospective_cases(
    config: dict[str, Any], summary: dict[str, Any], errors: list[str]
) -> list[str]:
    cases = config.get("retrospective_energy_cases")
    if summary.get("retrospective_energy_cases") != cases:
        errors.append("final matrix retrospective energy declarations disagree")
    if not isinstance(cases, list):
        errors.append("final matrix retrospective energy cases are invalid")
        return []
    string_cases = [case for case in cases if isinstance(case, str)]
    valid = (
        len(string_cases) == len(cases)
        and len(string_cases) == len(set(string_cases))
        and set(string_cases).issubset(REQUIRED_CASE_ORDER)
    )
    if not valid:
        errors.append("final matrix retrospective energy cases are invalid")
        return []
    return string_cases


def _load_matrix_sources(
    root: Path,
    sources: dict[str, Any],
    errors: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    loaded_by_role: dict[str, dict[str, dict[str, Any]]] = {"primary": {}, "r2": {}}
    for case_id in REQUIRED_CASE_ORDER:
        case_sources = sources.get(case_id)
        if not isinstance(case_sources, dict):
            errors.append(f"{case_id}: source declaration is invalid")
            continue
        for role in loaded_by_role:
            reference = case_sources.get(role)
            reference_errors = _verify_source_reference(root, case_id, role, reference)
            errors.extend(reference_errors)
            if reference_errors or not isinstance(reference, dict):
                continue
            try:
                directory = _resolve_reference_path(root, reference.get("directory"))
                loaded_by_role[role][case_id] = _load_verified_packet(directory)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{case_id} {role} source: {exc}")
    return loaded_by_role["primary"], loaded_by_role["r2"]


def _validate_matrix_supplements(
    primary: dict[str, dict[str, Any]],
    r2: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for case_id in REQUIRED_CASE_ORDER:
        try:
            _validate_r2_supplement(primary[case_id], r2[case_id])
        except ValueError as exc:
            errors.append(str(exc))


def _validate_reconstructed_matrix(
    *,
    root: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    rows: object,
    energy_assessment: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    primary_sources: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    if not _semantic_equal(summary.get("energy_stationarity_assessment"), energy_assessment):
        errors.append("final matrix energy assessment disagrees with its sources")
    if not _semantic_equal(rows, expected_rows):
        errors.append("final matrix case records disagree with their sources")
    accepted = energy_assessment.get("status") == "accepted" and all(
        row["status"] == "accepted" for row in expected_rows
    )
    if summary.get("status") != ("accepted" if accepted else "review"):
        errors.append("final matrix status disagrees with its case records")
    if summary.get("interpretation") != FINAL_MATRIX_INTERPRETATION:
        errors.append("final matrix estimator interpretation disagrees")
    _validate_source_root(root, summary, primary_sources, errors)
    table_path = root / "final_matrix_table.csv"
    if not table_path.is_file():
        errors.append("final matrix table is missing")
        return
    expected_table = csv_text(expected_rows, exclude=("primary_source", "r2_source"))
    if table_path.read_text(encoding="utf-8") != expected_table:
        errors.append("final matrix CSV disagrees with its case records")


def _validate_source_root(
    root: Path,
    summary: dict[str, Any],
    primary_sources: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    parents = {source["directory"].parent.resolve() for source in primary_sources.values()}
    if len(parents) != 1:
        errors.append("final matrix primary sources do not share one source root")
        return
    if summary.get("source_root") != _relative_locator(next(iter(parents)), root):
        errors.append("final matrix source root disagrees with its sources")


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
    energy_status_record = _energy_status_record(
        case_id,
        primary_summary=primary_summary,
        energy_assessment=energy_assessment,
        retrospective_energy_cases=retrospective_energy_cases,
    )
    energy_status = str(energy_status_record["energy_status"])
    r2_status = str(r2.get("status", "not_evaluated"))
    density_status = str(density.get("status", "not_evaluated"))
    status = "accepted" if energy_status == r2_status == density_status == "accepted" else "review"
    controls = _mapping(primary_summary.get("controls"))
    r2_fw = _mapping(r2_summary.get("pure_walking"))
    density_fw = _mapping(primary_summary.get("pure_walking"))
    return {
        "case": case_id,
        "status": status,
        **energy_status_record,
        "r2_status": r2_status,
        "density_status": density_status,
        "guide_family": primary_summary.get("guide_family"),
        "relative_alpha": controls.get("relative_alpha"),
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


def _energy_status_record(
    case_id: str,
    *,
    primary_summary: dict[str, Any],
    energy_assessment: dict[str, Any],
    retrospective_energy_cases: set[str],
) -> dict[str, Any]:
    source_energy_status = str(primary_summary.get("energy_validation_status", ""))
    matrix_energy = _mapping(_mapping(energy_assessment.get("cases")).get(case_id))
    if matrix_energy.get("status") != "accepted":
        energy_status = "review"
        energy_status_basis = "matrix_stationarity_assessment"
    elif source_energy_status == "accepted":
        energy_status = "accepted"
        energy_status_basis = "source_packet"
    elif case_id in retrospective_energy_cases:
        energy_status = "accepted"
        energy_status_basis = "retrospective_matrix_stationarity_assessment"
    else:
        energy_status = source_energy_status or "review"
        energy_status_basis = "source_packet"
    return {
        "energy_status": energy_status,
        "energy_status_basis": energy_status_basis,
        "source_energy_status": source_energy_status,
        "source_energy_stationarity_reason": _mapping(primary_summary.get("stationarity")).get(
            "stationarity_reason_energy"
        ),
    }


def _relative_delta(estimate: dict[str, Any]) -> float | None:
    delta = estimate.get("delta_vs_lda")
    reference = estimate.get("lda_value")
    if not isinstance(delta, (int, float)) or not isinstance(reference, (int, float)):
        return None
    if not np.isfinite(delta) or not np.isfinite(reference) or reference == 0.0:
        return None
    return float(delta / reference)
