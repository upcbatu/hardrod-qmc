from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.manifest import (
    file_sha256,
    load_manifest_bound_artifact,
    verify_run_manifest,
)
from hrdmc.statistics.equilibration import energy_validation_status


def load_verified_packet(source_dir: Path) -> dict[str, Any]:
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
    case_id = summary.get("case_id")
    config = mapping(manifest.get("config"))
    if not isinstance(case_id, str) or config.get("case") != case_id:
        raise ValueError(f"source packet case identity disagrees: {source_dir}")
    artifacts = {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if "summary.json" not in artifacts:
        raise ValueError(f"source packet manifest does not bind summary.json: {source_dir}")
    return {
        "directory": source_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "summary_path": summary_path.resolve(),
        "manifest": manifest,
        "summary": summary,
    }


def load_energy_assessment_packet(
    reference_root: Path, case_id: str, reference: object
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(reference, dict):
        raise ValueError(f"{case_id}: energy-assessment source reference is invalid")
    directory = resolve_reference_path(reference_root, reference.get("directory"))
    manifest_path = resolve_reference_path(reference_root, reference.get("manifest_path"))
    summary_path = resolve_reference_path(reference_root, reference.get("summary_path"))
    if manifest_path != directory / "run_manifest.json":
        raise ValueError(f"{case_id}: energy-assessment manifest path is invalid")
    if summary_path != directory / "summary.json":
        raise ValueError(f"{case_id}: energy-assessment summary path is invalid")
    if not manifest_path.is_file() or file_sha256(manifest_path) != reference.get(
        "manifest_sha256"
    ):
        raise ValueError(f"{case_id}: energy-assessment manifest identity mismatch")
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path, summary_path, allowed_unrelated_artifact_roots=("plots",)
    )
    if not summary_path.is_file() or file_sha256(summary_path) != reference.get("summary_sha256"):
        raise ValueError(f"{case_id}: energy-assessment summary identity mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_energy_assessment_identity(manifest, summary, reference, case_id)
    return (
        {
            "directory": directory,
            "manifest_path": manifest_path,
            "summary_path": summary_path,
            "manifest": manifest,
            "summary": summary,
        },
        warnings,
    )


def _validate_energy_assessment_identity(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    reference: dict[str, Any],
    case_id: str,
) -> None:
    if manifest.get("run_name") != "dmc_benchmark_packet":
        raise ValueError(f"{case_id}: energy-assessment source has the wrong owner")
    if summary.get("status") != manifest.get("status"):
        raise ValueError(f"{case_id}: energy-assessment source statuses disagree")
    config = mapping(manifest.get("config"))
    if config.get("case") != case_id or summary.get("case_id") != case_id:
        raise ValueError(f"{case_id}: energy-assessment source case identity mismatch")
    stationarity = mapping(summary.get("stationarity"))
    declared = summary.get("energy_validation_status")
    if declared != energy_validation_status(stationarity):
        raise ValueError(f"{case_id}: energy-assessment source energy status is not reproducible")
    estimate_status = mapping(mapping(summary.get("estimates")).get("energy")).get("status")
    if estimate_status != declared:
        raise ValueError(f"{case_id}: energy-assessment source estimate status disagrees")
    if manifest.get("run_id") != reference.get("run_id"):
        raise ValueError(f"{case_id}: energy-assessment source run identity mismatch")


def validate_r2_supplement(primary: dict[str, Any], supplement: dict[str, Any]) -> None:
    first, second = primary["summary"], supplement["summary"]
    case_id = str(first.get("case_id"))
    if second.get("case_id") != case_id:
        raise ValueError(f"{case_id}: R2 supplement has the wrong case identity")
    if source_tree_sha256(primary["manifest"]) != source_tree_sha256(supplement["manifest"]):
        raise ValueError(f"{case_id}: R2 supplement used a different implementation")
    for field in (
        "seeds",
        "n_particles",
        "rod_length_ho",
        "guide_family",
        "guide_parameters",
        "controls",
    ):
        if not semantic_equal(first.get(field), second.get(field)):
            raise ValueError(f"{case_id}: R2 supplement disagrees on {field}")
    if not semantic_equal(
        mapping(mapping(first.get("estimates")).get("energy")),
        mapping(mapping(second.get("estimates")).get("energy")),
    ):
        raise ValueError(f"{case_id}: R2 supplement did not reproduce the primary energy")
    if mapping(mapping(second.get("estimates")).get("r2")).get("status") != "accepted":
        raise ValueError(f"{case_id}: R2 supplement is not accepted")


def source_tree_sha256(manifest: dict[str, Any]) -> str | None:
    value = mapping(mapping(manifest.get("provenance")).get("implementation")).get(
        "source_tree_sha256"
    )
    return value if isinstance(value, str) and len(value) == 64 else None


def source_reference(source: dict[str, Any], *, reference_root: Path) -> dict[str, Any]:
    manifest = source["manifest"]
    return {
        "directory": relative_locator(source["directory"], reference_root),
        "summary_path": relative_locator(source["summary_path"], reference_root),
        "summary_sha256": file_sha256(source["summary_path"]),
        "manifest_path": relative_locator(source["manifest_path"], reference_root),
        "manifest_sha256": file_sha256(source["manifest_path"]),
        "run_id": manifest.get("run_id"),
        "bundle_sha256": manifest.get("bundle_sha256"),
    }


def verify_source_reference(root: Path, case_id: str, role: str, reference: object) -> list[str]:
    prefix = f"{case_id} {role} source"
    if not isinstance(reference, dict):
        return [f"{prefix}: reference is invalid"]
    directory = resolve_reference_path(root, reference.get("directory"))
    manifest = resolve_reference_path(root, reference.get("manifest_path"))
    summary = resolve_reference_path(root, reference.get("summary_path"))
    errors = []
    if manifest != directory / "run_manifest.json" or summary != directory / "summary.json":
        errors.append(f"{prefix}: declared paths are invalid")
    if not manifest.is_file() or file_sha256(manifest) != reference.get("manifest_sha256"):
        return [*errors, f"{prefix}: manifest identity mismatch"]
    valid, manifest_errors = verify_run_manifest(manifest)
    if not valid:
        errors.extend(f"{prefix}: {error}" for error in manifest_errors)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key, label in (("run_id", "run identity"), ("bundle_sha256", "bundle identity")):
        if payload.get(key) != reference.get(key):
            errors.append(f"{prefix}: {label} mismatch")
    if not summary.is_file() or file_sha256(summary) != reference.get("summary_sha256"):
        errors.append(f"{prefix}: summary identity mismatch")
    return errors


def required_config_float(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} is not numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def relative_locator(path: Path, reference_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=reference_root.resolve())).as_posix()


def resolve_reference_path(reference_root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return (path if path.is_absolute() else reference_root / path).resolve()


def mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def semantic_equal(first: object, second: object) -> bool:
    return json.dumps(first, sort_keys=True, allow_nan=True) == json.dumps(
        second, sort_keys=True, allow_nan=True
    )
