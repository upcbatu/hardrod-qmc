from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import ensure_dir, write_csv, write_json, write_run_manifest
from hrdmc.uncertainty.forward_walking.sources import mapping

FW_SENSITIVITY_RUN_NAME = "dmc_fw_sensitivity"


def write_fw_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    command: list[str] | None,
) -> dict[str, str]:
    root = ensure_dir(output_dir.resolve())
    summary = root / "summary.json"
    comparison = payload.get("observable_comparison")
    observable_rows = []
    if isinstance(comparison, dict):
        for name in ("r2", "rms_radius", "density"):
            observable_rows.append({"observable": name, **mapping(comparison.get(name), name)})
    observable_table = write_csv(root / "observable_comparison.csv", observable_rows)
    shell_table = write_csv(root / "shell_comparison.csv", _shell_rows(comparison))
    write_json(summary, payload)
    manifest = write_run_manifest(
        root,
        run_name=FW_SENSITIVITY_RUN_NAME,
        config={
            "case_id": payload["case_id"],
            "treatments": payload["treatments"],
            "sampling_design": payload["sampling_design"],
            "command": command,
        },
        artifacts=[summary, observable_table, shell_table],
        status=str(payload["status"]),
    )
    return {
        "summary": str(summary),
        "observable_table": str(observable_table),
        "shell_table": str(shell_table),
        "run_manifest": str(manifest),
        "output_dir": str(root),
    }


def empty_artifacts(output_dir: Path | None) -> dict[str, str | None]:
    return {
        "summary": None,
        "observable_table": None,
        "shell_table": None,
        "run_manifest": None,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
    }


def _shell_rows(comparison: object) -> list[dict[str, Any]]:
    if not isinstance(comparison, dict):
        return []
    shell = comparison.get("shell_peaks")
    if not isinstance(shell, dict):
        return []
    names = (
        "centers",
        "anchor_envelope",
        "candidate_envelope",
        "anchor_peak_positions",
        "candidate_peak_positions",
    )
    arrays = [shell.get(name, []) for name in names]
    return [dict(zip(names, values, strict=True)) for values in zip(*arrays, strict=True)]
