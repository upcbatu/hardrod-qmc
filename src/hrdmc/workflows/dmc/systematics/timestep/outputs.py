from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hrdmc.artifacts import build_run_provenance, ensure_dir, write_json, write_run_manifest
from hrdmc.workflows.dmc.systematics.timestep.contract import LoadedTimeStepPoint


def _validate_output_separation(
    output_dir: Path,
    source_artifact_paths: Sequence[Path],
) -> None:
    for source_artifact_path in source_artifact_paths:
        run_dir = source_artifact_path.parent
        if (
            output_dir == run_dir
            or output_dir.is_relative_to(run_dir)
            or run_dir.is_relative_to(output_dir)
        ):
            raise ValueError(
                f"output_dir must not overlap an input artifact or run directory: {output_dir}"
            )


def _write_point_table(
    output_dir: Path,
    points: Sequence[LoadedTimeStepPoint],
) -> Path:
    path = output_dir / "point_table.csv"
    fields = (
        "dt",
        "energy",
        "conservative_stderr",
        "case_id",
        "run_name",
        "result_schema_version",
        "run_id",
        "bundle_sha256",
        "run_status",
        "energy_status",
        "energy_publication_accepted",
        "energy_publication_status",
        "energy_status_basis",
        "source_energy_publication_accepted",
        "source_energy_publication_status",
        "energy_assessment_manifest_sha256",
        "energy_assessment_run_id",
        "seed_count",
        "seeds",
        "summary_path",
        "summary_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_verification",
        "local_acceptance_fraction_mean",
        "invalid_proposal_fraction_max",
        "metropolis_rejection_fraction_max",
        "configuration_esjd_mean",
        "log_weight_span_max",
        "rhat_energy",
        "neff_energy",
        "population_weight_status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            payload = point.to_dict()
            payload["seeds"] = ",".join(str(seed) for seed in point.seeds)
            payload.update(
                {
                    "energy_publication_accepted": point.energy_quality.get("publication_accepted"),
                    "energy_publication_status": point.energy_quality.get("publication_status"),
                    "energy_status_basis": point.energy_quality.get("status_basis"),
                    "source_energy_publication_accepted": point.energy_quality.get(
                        "source_publication_accepted"
                    ),
                    "source_energy_publication_status": point.energy_quality.get(
                        "source_publication_status"
                    ),
                    "energy_assessment_manifest_sha256": (
                        point.energy_quality_assessment or {}
                    ).get("assessment_manifest_sha256"),
                    "energy_assessment_run_id": (point.energy_quality_assessment or {}).get(
                        "assessment_run_id"
                    ),
                }
            )
            payload.update(point.telemetry)
            writer.writerow({field: payload.get(field, "") for field in fields})
    return path


def persist_timestep_outputs(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    config: dict[str, Any],
    points: Sequence[LoadedTimeStepPoint],
    command: list[str] | None,
    run_name: str,
    schema_version: str,
    status: str,
) -> dict[str, str]:
    """Persist the fixed summary, CSV projection, and manifest artifact set."""

    root = ensure_dir(output_dir.resolve())
    summary_path = root / "summary.json"
    table_path = _write_point_table(root, points)
    payload["artifacts"] = {
        "summary": str(summary_path),
        "point_table": str(table_path),
        "run_manifest": str(root / "run_manifest.json"),
        "output_dir": str(root),
    }
    write_json(summary_path, payload)
    manifest_path = write_run_manifest(
        root,
        run_name=run_name,
        config=config,
        artifacts=[summary_path, table_path],
        schema_version=schema_version,
        provenance=build_run_provenance(command),
        status=status,
    )
    return {
        "summary": str(summary_path),
        "point_table": str(table_path),
        "run_manifest": str(manifest_path),
        "output_dir": str(root),
    }
