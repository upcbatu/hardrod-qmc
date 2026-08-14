from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import (
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    verify_run_manifest,
    write_csv,
    write_json,
    write_run_manifest,
)
from hrdmc.statistics.equilibration import (
    assess_matrix_energy_stationarity,
    energy_validation_status,
)
from hrdmc.system.settings import parse_case

ENERGY_STATIONARITY_ASSESSMENT_RUN_NAME = "dmc_energy_stationarity_assessment"
ENERGY_UNCERTAINTY_ACCEPTED_STATUSES = {
    "accepted",
    "conservative_error_inflated",
    "blocking_plateau_unresolved_correlated_error_available",
}


@dataclass(frozen=True)
class _LoadedEnergySource:
    case_id: str
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    stationarity: dict[str, Any]
    warnings: tuple[str, ...]

    def reference(self, *, reference_root: Path) -> dict[str, Any]:
        return {
            "summary_path": _relative_locator(self.summary_path, reference_root),
            "summary_sha256": file_sha256(self.summary_path),
            "manifest_path": _relative_locator(self.manifest_path, reference_root),
            "manifest_sha256": file_sha256(self.manifest_path),
            "run_id": _required_string(self.manifest, "run_id"),
        }


def run_energy_stationarity_assessment(
    source_summary_paths: Mapping[str, Path],
    *,
    expected_case_ids: Sequence[str],
    output_dir: Path,
    confidence_level: float = 0.95,
    rhat_limit: float = 1.05,
    min_effective_samples: float = 30.0,
) -> dict[str, Any]:
    """Assess one predeclared family of manifest-bound benchmark energies."""
    case_ids = _validate_case_family(source_summary_paths, expected_case_ids)
    root = output_dir.resolve()
    sources = {
        case_id: _load_benchmark_energy_source(
            Path(source_summary_paths[case_id]).resolve(),
            expected_case_id=case_id,
        )
        for case_id in case_ids
    }
    _validate_output_directory(root, sources.values())
    assessment = assess_matrix_energy_stationarity(
        {case_id: sources[case_id].stationarity for case_id in case_ids},
        confidence_level=confidence_level,
        rhat_limit=rhat_limit,
        min_effective_samples=min_effective_samples,
    )
    assessment["policy_timing"] = "retrospective"
    assessment["scope"] = "predeclared_forward_walking_candidate_family"
    rows = [
        _case_record(
            sources[case_id],
            _required_mapping(_required_mapping(assessment, "cases"), case_id),
            reference_root=root,
        )
        for case_id in case_ids
    ]
    status = "accepted" if all(row["publication_accepted"] for row in rows) else "review"
    source_references = {
        case_id: sources[case_id].reference(reference_root=root) for case_id in case_ids
    }
    payload: dict[str, Any] = {
        "status": status,
        "diagnostic": "simultaneous mixed-energy stationarity assessment",
        "case_order": list(case_ids),
        "source_locator_base": "assessment_directory",
        "energy_stationarity_assessment": assessment,
        "sources": source_references,
        "rows": rows,
        "publication_ready_within_energy_stationarity_scope": status == "accepted",
        "scientific_scope": (
            "A family-wise directional stationarity screen for the declared benchmark "
            "energy sources. It may reclassify only trace-stationarity failures with "
            "available conservative correlated-energy uncertainty; it does not qualify "
            "forward-walking plateau or genealogy support."
        ),
        "manifest_verification_warnings": [
            f"{case_id}: {warning}" for case_id in case_ids for warning in sources[case_id].warnings
        ],
    }
    config = {
        "case_order": list(case_ids),
        "source_locator_base": "assessment_directory",
        "confidence_level": confidence_level,
        "rhat_limit": rhat_limit,
        "min_effective_samples": min_effective_samples,
        "sources": source_references,
    }
    ensure_dir(root)
    summary_path = root / "summary.json"
    table_path = root / "case_table.csv"
    write_json(summary_path, payload)
    _write_case_table(table_path, rows)
    manifest_path = write_run_manifest(
        root,
        run_name=ENERGY_STATIONARITY_ASSESSMENT_RUN_NAME,
        config=config,
        artifacts=[summary_path, table_path],
        status=status,
    )
    payload["workflow_artifacts"] = {
        "summary": str(summary_path),
        "case_table": str(table_path),
        "run_manifest": str(manifest_path),
        "output_dir": str(root),
    }
    return payload


def load_energy_stationarity_selection(
    manifest_path: Path,
    *,
    case_id: str,
    selected_summary_path: Path,
) -> dict[str, Any]:
    """Verify one selected source against its complete candidate-family assessment."""
    path = manifest_path.resolve()
    root, manifest, summary_path, summary, config, case_order, config_sources = (
        _load_energy_selection_artifacts(path, case_id)
    )
    sources, assessment, expected_rows, expected_status = _reconstruct_energy_assessment(
        root, config, config_sources, case_order
    )
    if summary.get("energy_stationarity_assessment") != assessment:
        raise ValueError("energy assessment disagrees with its exact sources")
    if summary.get("rows") != expected_rows or summary.get("status") != expected_status:
        raise ValueError("energy assessment case records disagree with their sources")
    selected = selected_summary_path.resolve()
    source = sources[case_id]
    if source.summary_path != selected:
        raise ValueError("energy assessment selects a different candidate summary")
    selected_rows = [row for row in expected_rows if row["case_id"] == case_id]
    if len(selected_rows) != 1:
        raise ValueError("energy assessment does not contain one selected case record")
    row = selected_rows[0]
    return {
        **row,
        "assessment_manifest_path": str(path),
        "assessment_manifest_sha256": file_sha256(path),
        "assessment_summary_path": str(summary_path),
        "assessment_summary_sha256": file_sha256(summary_path),
        "assessment_run_id": _required_string(manifest, "run_id"),
        "assessment_method": assessment.get("method"),
        "assessment_scope": assessment.get("scope"),
        "policy_timing": assessment.get("policy_timing"),
        "case_assessment": _required_mapping(_required_mapping(assessment, "cases"), case_id),
    }


def _load_energy_selection_artifacts(
    path: Path, case_id: str
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    list[str],
    dict[str, Any],
]:
    verified, errors = verify_run_manifest(path)
    if not verified:
        raise ValueError("energy assessment manifest verification failed: " + "; ".join(errors))
    manifest = _load_mapping(path, "energy assessment manifest")
    if manifest.get("run_name") != ENERGY_STATIONARITY_ASSESSMENT_RUN_NAME:
        raise ValueError("energy assessment has the wrong artifact owner")
    artifact_paths = {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if artifact_paths != {"summary.json", "case_table.csv"}:
        raise ValueError("energy assessment has the wrong artifact set")
    root = path.parent
    summary_path = root / "summary.json"
    summary = _load_mapping(summary_path, "energy assessment summary")
    config = _required_mapping(manifest, "config")
    if summary.get("status") != manifest.get("status"):
        raise ValueError("energy assessment summary and manifest statuses disagree")
    case_order = _string_list(config.get("case_order"), "case_order")
    if summary.get("case_order") != case_order or case_id not in case_order:
        raise ValueError("energy assessment case identities disagree")
    if config.get("source_locator_base") != "assessment_directory":
        raise ValueError("energy assessment has the wrong source locator base")
    if summary.get("source_locator_base") != config.get("source_locator_base"):
        raise ValueError("energy assessment source locator declarations disagree")
    config_sources = _required_mapping(config, "sources")
    if set(config_sources) != set(case_order) or summary.get("sources") != config_sources:
        raise ValueError("energy assessment source declarations disagree")
    return root, manifest, summary_path, summary, config, case_order, config_sources


def _reconstruct_energy_assessment(
    root: Path,
    config: dict[str, Any],
    config_sources: dict[str, Any],
    case_order: Sequence[str],
) -> tuple[
    dict[str, _LoadedEnergySource],
    dict[str, Any],
    list[dict[str, Any]],
    str,
]:
    sources = {
        case: _load_declared_source(root, case, _required_mapping(config_sources, case))
        for case in case_order
    }
    assessment = assess_matrix_energy_stationarity(
        {case: sources[case].stationarity for case in case_order},
        confidence_level=_required_float(config.get("confidence_level"), "confidence_level"),
        rhat_limit=_required_float(config.get("rhat_limit"), "rhat_limit"),
        min_effective_samples=_required_float(
            config.get("min_effective_samples"), "min_effective_samples"
        ),
    )
    assessment["policy_timing"] = "retrospective"
    assessment["scope"] = "predeclared_forward_walking_candidate_family"
    cases = _required_mapping(assessment, "cases")
    rows = [
        _case_record(sources[case], _required_mapping(cases, case), reference_root=root)
        for case in case_order
    ]
    status = "accepted" if all(row["publication_accepted"] for row in rows) else "review"
    return sources, assessment, rows, status


def _validate_case_family(
    sources: Mapping[str, Path], expected_case_ids: Sequence[str]
) -> tuple[str, ...]:
    case_ids = tuple(expected_case_ids)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("expected case identities must be nonempty and unique")
    if any(parse_case(case_id).case_id != case_id for case_id in case_ids):
        raise ValueError("expected cases must use canonical harmonic-oscillator identifiers")
    if set(sources) != set(case_ids):
        raise ValueError("energy assessment sources must exactly match expected cases")
    return case_ids


def _load_benchmark_energy_source(
    summary_path: Path,
    *,
    expected_case_id: str,
) -> _LoadedEnergySource:
    if not summary_path.is_file():
        raise FileNotFoundError(f"energy source summary does not exist: {summary_path}")
    manifest_path = summary_path.parent / "run_manifest.json"
    summary = _load_mapping(summary_path, "benchmark summary")
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        summary_path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    if manifest.get("run_name") != "dmc_benchmark_packet":
        raise ValueError("energy assessment requires dmc_benchmark_packet inputs")
    if summary.get("status") != manifest.get("status"):
        raise ValueError("energy source summary and manifest statuses disagree")
    case_id = _required_string(summary, "case_id")
    if case_id != expected_case_id:
        raise ValueError("energy source has the wrong case identity")
    config = _required_mapping(manifest, "config")
    if config.get("case") != case_id:
        raise ValueError("energy source case identity disagrees with its manifest")
    for field in ("controls", "seeds"):
        if summary.get(field) != config.get(field):
            raise ValueError(f"energy source {field} disagrees with its manifest")
    stationarity = _required_mapping(summary, "stationarity")
    declared_energy_status = _required_string(summary, "energy_validation_status")
    if declared_energy_status != energy_validation_status(stationarity):
        raise ValueError("energy source validation status is not reproducible")
    energy = _required_mapping(_required_mapping(summary, "estimates"), "energy")
    if energy.get("status") != declared_energy_status:
        raise ValueError("energy source estimate status disagrees with validation")
    return _LoadedEnergySource(
        case_id=case_id,
        summary_path=summary_path,
        manifest_path=manifest_path,
        summary=summary,
        manifest=manifest,
        stationarity=stationarity,
        warnings=warnings,
    )


def _load_declared_source(
    root: Path,
    case_id: str,
    reference: Mapping[str, Any],
) -> _LoadedEnergySource:
    summary_path = _resolve_locator(root, reference.get("summary_path"))
    manifest_path = _resolve_locator(root, reference.get("manifest_path"))
    if manifest_path != summary_path.parent / "run_manifest.json":
        raise ValueError("energy assessment source manifest locator is invalid")
    source = _load_benchmark_energy_source(summary_path, expected_case_id=case_id)
    expected = source.reference(reference_root=root)
    if dict(reference) != expected:
        raise ValueError("energy assessment source binding disagrees")
    return source


def _case_record(
    source: _LoadedEnergySource,
    case_assessment: Mapping[str, Any],
    *,
    reference_root: Path,
) -> dict[str, Any]:
    source_status = _required_string(source.summary, "energy_validation_status")
    source_accepted = source_status == "accepted"
    override = _trace_override_eligibility(source)
    publication_accepted = case_assessment.get("status") == "accepted" and (
        source_accepted or override["eligible"]
    )
    publication_status = (
        "accepted"
        if publication_accepted and source_accepted
        else "accepted_with_retrospective_assessment"
        if publication_accepted
        else "unresolved"
    )
    return {
        "case_id": source.case_id,
        "source": source.reference(reference_root=reference_root),
        "source_energy_status": source_status,
        "source_energy_stationarity_reason": source.stationarity.get("stationarity_reason_energy"),
        "source_override_eligibility": override,
        "case_assessment": dict(case_assessment),
        "publication_accepted": publication_accepted,
        "publication_status": publication_status,
    }


def _trace_override_eligibility(source: _LoadedEnergySource) -> dict[str, Any]:
    source_status = source.summary.get("energy_validation_status")
    failures: list[str] = []
    if source_status == "accepted":
        failures.append("source_already_accepted")
    if source_status != "trace_nonstationary":
        failures.append("source_energy_status_is_not_trace_nonstationary")
    if source.stationarity.get("stationarity_energy") != "trace_nonstationary":
        failures.append("source_energy_chain_status_is_not_trace_nonstationary")
    if source.stationarity.get("mixed_energy_uncertainty_status") not in (
        ENERGY_UNCERTAINTY_ACCEPTED_STATUSES
    ):
        failures.append("source_energy_uncertainty_is_unresolved")
    return {
        "eligible": not failures,
        "criterion": (
            "only trace-stationarity failures with available conservative correlated-energy "
            "uncertainty may be reclassified"
        ),
        "failures": failures,
    }


def _write_case_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "case_id",
        "source_energy_status",
        "source_energy_stationarity_reason",
        "maximum_directional_z",
        "rhat",
        "effective_samples_min",
        "publication_status",
    ]
    table = []
    for row in rows:
        case_assessment = _required_mapping(row, "case_assessment")
        table.append(
            {
                "case_id": row.get("case_id"),
                "source_energy_status": row.get("source_energy_status"),
                "source_energy_stationarity_reason": row.get("source_energy_stationarity_reason"),
                "maximum_directional_z": case_assessment.get("maximum_directional_z"),
                "rhat": case_assessment.get("rhat"),
                "effective_samples_min": case_assessment.get("effective_samples_min"),
                "publication_status": row.get("publication_status"),
            }
        )
    return write_csv(path, table, fieldnames=fields)


def _validate_output_directory(
    root: Path,
    sources: Iterable[_LoadedEnergySource],
) -> None:
    source_directories = {source.summary_path.parent for source in sources}
    if any(
        root == source_dir or root in source_dir.parents or source_dir in root.parents
        for source_dir in source_directories
    ):
        raise ValueError("energy assessment output directory must not overlap an input run")
    for name in ("summary.json", "case_table.csv", "run_manifest.json"):
        if (root / name).exists():
            raise FileExistsError(
                f"energy assessment output artifact already exists: {root / name}"
            )


def _relative_locator(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=root.resolve())).as_posix()


def _resolve_locator(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("energy assessment source locators must be relative")
    return (root / value).resolve()


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _required_mapping(owner: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = owner.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(owner: Mapping[str, Any], field: str) -> str:
    value = owner.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{field} must contain unique nonempty strings")
    return list(value)
