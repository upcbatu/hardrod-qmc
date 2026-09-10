"""Guide inputs shared by researcher VMC and DMC commands."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from hrdmc.artifacts.manifest import file_sha256
from hrdmc.system.guide_registry import load_validated_reduced_tg_guide
from hrdmc.system.settings import TrappedCase


@dataclass(frozen=True)
class GuideSelection:
    relative_alpha: float | None
    source: str
    validation: str
    candidate_path: str | None = None
    candidate_sha256: str | None = None


def select_guide(
    case: TrappedCase,
    *,
    alpha: float | None = None,
    alpha_from: Path | None = None,
    registry: Path | None = None,
) -> GuideSelection:
    if sum(value is not None for value in (alpha, alpha_from, registry)) > 1:
        raise ValueError("choose only one of alpha, alpha-from, or guide registry")
    if registry is not None:
        if case.rod_length == 0:
            return GuideSelection(None, "explicit", "exact_hard_point")
        artifact = load_validated_reduced_tg_guide(registry, case=case)
        return GuideSelection(artifact.relative_alpha, str(artifact.summary_path), "validated")
    candidate_path = candidate_hash = None
    if alpha_from is not None:
        path = alpha_from.expanduser().resolve()
        alpha = _candidate_alpha(path, case)
        candidate_path, candidate_hash = str(path), file_sha256(path)
    if alpha is None:
        if case.rod_length == 0:
            return GuideSelection(None, "explicit", "exact_hard_point")
        raise ValueError("finite A requires --alpha, --alpha-from, or --preset thesis")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    return GuideSelection(
        alpha, "explicit", "not_independently_validated", candidate_path, candidate_hash
    )


def _candidate_alpha(path: Path, case: TrappedCase) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "reduced_tg_relative_alpha_optimization_v1"
    ):
        raise ValueError("alpha-from must be an alpha optimization summary.json")
    if payload.get("case_id") != case.case_id:
        raise ValueError("alpha candidate belongs to a different case")
    try:
        return float(payload["recommended_relative_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("alpha candidate has no numeric recommended_relative_alpha") from exc
