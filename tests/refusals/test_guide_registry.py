from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hrdmc.artifacts.manifest import write_json, write_run_manifest
from hrdmc.production.matrix.method import DEFAULT_GUIDE_VALIDATION_SUMMARY
from hrdmc.system.guide_registry import (
    GUIDE_REGISTRY_RUN_NAME,
    load_validated_reduced_tg_guide,
)
from hrdmc.system.settings import parse_case


def _registry(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    payload = json.loads(DEFAULT_GUIDE_VALIDATION_SUMMARY.read_text(encoding="utf-8"))
    mutate(payload)
    root = tmp_path / "registry"
    root.mkdir()
    summary = root / "summary.json"
    write_json(summary, payload)
    write_run_manifest(
        root,
        run_name=GUIDE_REGISTRY_RUN_NAME,
        config={},
        artifacts=[summary],
        status="validated",
    )
    return summary


def test_registry_rejects_a_missing_finite_case(tmp_path: Path) -> None:
    path = _registry(tmp_path, lambda payload: payload["records"].pop("N10_A1"))

    with pytest.raises(ValueError, match="no validated record"):
        load_validated_reduced_tg_guide(path, case=parse_case("N10_A1"))


def test_registry_rejects_a_nonpositive_optimized_width(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["records"]["N10_A1"]["validated_parameters"]["relative_alpha"] = 0.0

    path = _registry(tmp_path, mutate)
    with pytest.raises(ValueError, match="finite and positive"):
        load_validated_reduced_tg_guide(path, case=parse_case("N10_A1"))
