from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from hrdmc.artifacts.manifest import verify_run_manifest
from hrdmc.system.settings import TrappedCase

GUIDE_REGISTRY_RUN_NAME = "final_matrix_guide_registry"
GUIDE_REGISTRY_SCHEMA = "final_matrix_guide_registry_v1"


@dataclass(frozen=True)
class ValidatedReducedTGGuideArtifact:
    relative_alpha: float
    summary_path: Path


def validate_production_reduced_tg_binding(
    *,
    case: TrappedCase,
    guide_family: str,
    relative_alpha: float | None,
    source: str,
) -> None:
    """Require registry-backed optimized widths for finite-diameter production."""
    if guide_family != "reduced-tg":
        raise ValueError("production supports only the reduced-tg guide family")
    if relative_alpha is None:
        if source != "explicit":
            raise ValueError("a guide source without relative_alpha is invalid")
        return
    if source == "explicit":
        raise ValueError("optimized relative_alpha requires a validated guide artifact")
    artifact = load_validated_reduced_tg_guide(Path(source), case=case)
    if float(relative_alpha) != artifact.relative_alpha:
        raise ValueError("guide width or artifact identity does not match validation")


def load_validated_reduced_tg_guide(
    path: Path,
    *,
    case: TrappedCase,
) -> ValidatedReducedTGGuideArtifact:
    """Load an optimized reduced-TG width from its verified registry artifact."""
    resolved_path = path.resolve()
    registry = _load_verified_registry(resolved_path)
    if registry.get("schema_version") != GUIDE_REGISTRY_SCHEMA:
        raise ValueError("guide registry has the wrong schema")
    if registry.get("status") != "validated":
        raise ValueError("guide registry is not validated")
    record = _validated_registry_record(registry, case=case)
    parameters = record.get("validated_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("guide registry record has no validated_parameters")
    if parameters.get("guide_family") != "reduced-tg":
        raise ValueError("guide registry record has the wrong guide family")
    try:
        alpha = float(parameters["relative_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("validated relative_alpha must be numeric") from exc
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("validated relative_alpha must be finite and positive")
    return ValidatedReducedTGGuideArtifact(
        relative_alpha=alpha,
        summary_path=resolved_path,
    )


def _load_verified_registry(summary_path: Path) -> dict[str, object]:
    manifest_path = summary_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("guide registry summary has no run manifest")
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError("guide registry manifest failed verification: " + "; ".join(errors))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != GUIDE_REGISTRY_RUN_NAME:
        raise ValueError("guide registry manifest has the wrong owner")
    if manifest.get("status") != "validated":
        raise ValueError("guide registry manifest is not validated")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("guide registry summary must be a mapping")
    return payload


def _validated_registry_record(
    registry: dict[str, object],
    *,
    case: TrappedCase,
) -> dict[str, object]:
    records = registry.get("records")
    if not isinstance(records, dict):
        raise ValueError("guide registry has no case records")
    record = records.get(case.case_id)
    if not isinstance(record, dict):
        raise ValueError(f"guide registry has no validated record for {case.case_id}")
    if record.get("check_count") != 21 or record.get("failed_checks") != []:
        raise ValueError("guide registry does not preserve the 21-check validation")
    if not _is_sha256(record.get("source_summary_sha256")):
        raise ValueError("guide registry has an invalid source_summary_sha256")
    if not _is_sha256(registry.get("source_validation_tree_sha256")):
        raise ValueError("guide registry has no source implementation identity")
    return record


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
