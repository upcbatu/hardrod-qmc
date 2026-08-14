from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import file_sha256, load_manifest_bound_artifact


@dataclass(frozen=True)
class LoadedBenchmarkPacket:
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    verification_warnings: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return string(self.summary, "case_id")

    @property
    def controls(self) -> dict[str, Any]:
        return mapping(self.summary.get("controls"), "controls")

    @property
    def pure_config(self) -> dict[str, Any]:
        return mapping(self.summary.get("pure_config"), "pure_config")

    @property
    def seeds(self) -> tuple[int, ...]:
        return seeds(self.summary.get("seeds"))

    @property
    def dt(self) -> float:
        return positive(self.controls.get("dt"), "dt")

    @property
    def walkers(self) -> int:
        return positive_int(self.controls.get("walkers"), "walkers")

    def reference(self) -> dict[str, Any]:
        return {
            "summary_path": str(self.summary_path),
            "summary_sha256": file_sha256(self.summary_path),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "run_id": string(self.manifest, "run_id"),
            "dt": self.dt,
            "walkers": self.walkers,
        }


@dataclass(frozen=True)
class AnchorSources:
    assembly_manifest_path: Path
    assembly_manifest: dict[str, Any]
    density: LoadedBenchmarkPacket
    r2: LoadedBenchmarkPacket


def load_manifest_bound_benchmark_packet(summary_path: Path) -> LoadedBenchmarkPacket:
    path = summary_path.resolve()
    manifest_path = path.parent / "run_manifest.json"
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    summary = _json_mapping(path)
    if manifest.get("run_name") != "dmc_benchmark_packet":
        raise ValueError(f"not a benchmark packet: {path}")
    if manifest.get("status") != summary.get("status"):
        raise ValueError(f"summary/manifest status mismatch: {path}")
    config = mapping(manifest.get("config"), "manifest config")
    if config.get("case") != summary.get("case_id"):
        raise ValueError(f"case identity mismatch: {path}")
    if mapping(config.get("controls"), "manifest controls") != mapping(
        summary.get("controls"), "summary controls"
    ):
        raise ValueError(f"control identity mismatch: {path}")
    return LoadedBenchmarkPacket(path, manifest_path, summary, manifest, tuple(warnings))


def load_final_matrix_anchor_sources(path: Path, *, case_id: str) -> AnchorSources:
    manifest_path = path.resolve()
    summary_path = manifest_path.parent / "final_matrix_summary.json"
    manifest, _ = load_manifest_bound_artifact(manifest_path, summary_path)
    summary = _json_mapping(summary_path)
    if manifest.get("run_name") != "dmc_final_matrix_assembly":
        raise ValueError("final matrix manifest has the wrong owner")
    row = next((item for item in summary.get("rows", []) if item.get("case") == case_id), None)
    if not isinstance(row, dict):
        raise ValueError(f"final matrix has no {case_id} row")
    base = summary_path.parent
    density = load_manifest_bound_benchmark_packet(
        resolve(base, mapping(row.get("primary_source"), "primary source")["summary_path"])
    )
    r2 = load_manifest_bound_benchmark_packet(
        resolve(base, mapping(row.get("r2_source"), "R2 source")["summary_path"])
    )
    if density.case_id != case_id or r2.case_id != case_id:
        raise ValueError("final matrix source case mismatch")
    return AnchorSources(manifest_path, manifest, density, r2)


def _json_mapping(path: Path) -> dict[str, Any]:
    return mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a nonempty string")
    return result


def positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def positive_or_zero(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be non-negative")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def seeds(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value)
    ):
        raise ValueError("seeds must contain at least two integer identities")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError("seed identities must be unique")
    return result


def resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
