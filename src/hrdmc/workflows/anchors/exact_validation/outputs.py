from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hrdmc.io.artifacts import ensure_dir, write_json_atomic, write_run_manifest
from hrdmc.plotting import write_exact_validation_packet_plots
from hrdmc.workflows.dmc.rn_block import RNRunControls, rn_run_config


def anchor_row_from_trapped(payload: dict[str, Any]) -> dict[str, Any]:
    comparison = payload["exact_comparison"]
    return {
        "anchor_id": payload["anchor_id"],
        "anchor_type": payload["anchor_type"],
        "passed": payload["status"] == "passed",
        "gate_status": payload["status"],
        "primary_metric": "max_relative_observable_error",
        "primary_abs_error": comparison["max_relative_observable_error"],
        "tolerance": comparison["max_relative_tolerance"],
        "energy_abs_error": payload["absolute_energy_error"],
        "pure_r2_relative_error": comparison["pure_r2_relative_error"],
        "pure_rms_relative_error": comparison["pure_rms_relative_error"],
        "pure_density_relative_l2": comparison["pure_density_relative_l2"],
        "density_bin_count": comparison["density_bin_count"],
        "density_shape_min_bins": comparison["density_shape_min_bins"],
        "density_resolution_gate": comparison["density_resolution_gate"],
        "density_accounting_gate": comparison["density_accounting_gate"],
        "density_accounting_abs_error": comparison["density_accounting_abs_error"],
        "density_accounting_tolerance": comparison["density_accounting_tolerance"],
        "mixed_r2_relative_error_diagnostic": comparison[
            "mixed_r2_relative_error_diagnostic"
        ],
        "mixed_rms_relative_error_diagnostic": comparison[
            "mixed_rms_relative_error_diagnostic"
        ],
        "mixed_density_relative_l2_diagnostic": comparison[
            "mixed_density_relative_l2_diagnostic"
        ],
        "pure_mixed_density_relative_l2_diagnostic": comparison[
            "pure_mixed_density_relative_l2_diagnostic"
        ],
        "mixed_density_closer_than_fw_to_exact_diagnostic": comparison[
            "mixed_density_closer_than_fw_to_exact_diagnostic"
        ],
    }


def anchor_row_from_homogeneous(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_id": payload["anchor_id"],
        "anchor_type": payload["anchor_type"],
        "passed": payload["status"] == "passed",
        "gate_status": payload["status"],
        "primary_metric": "energy_per_particle",
        "primary_abs_error": payload["max_energy_per_particle_abs_error"],
        "tolerance": payload["tolerance_energy_per_particle_abs"],
    }


def write_packet_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    plot_formats: tuple[str, ...],
) -> list[Path]:
    root = ensure_dir(output_dir)
    summary_path = write_json_atomic(root / "summary.json", payload)
    table_path = _write_anchor_table(root / "anchor_table.csv", payload["anchor_table"])
    plot_paths = write_exact_validation_packet_plots(
        root,
        payload,
        formats=plot_formats,
    )
    payload["plots"] = plot_paths
    write_json_atomic(summary_path, payload)
    return [summary_path, table_path, *[root / path for path in plot_paths]]


def write_exact_validation_manifest(
    output_dir: Path,
    payload: dict[str, Any],
    artifacts: list[Path],
    controls: RNRunControls,
    seeds: list[int],
    command: list[str],
) -> None:
    write_run_manifest(
        output_dir,
        run_name="rn_block_exact_validation_packet",
        config=rn_run_config(
            run_kind="rn_block_exact_validation_packet",
            cases=[str(row["anchor_id"]) for row in payload["anchor_table"]],
            seeds=seeds,
            controls=controls,
            parallel_workers=int(payload["parallel_workers_requested"]),
        ),
        artifacts=artifacts,
        schema_version=str(payload["schema_version"]),
        provenance={"command": command},
        status=str(payload["status"]),
    )


def _write_anchor_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = [
        "anchor_id",
        "anchor_type",
        "gate_status",
        "primary_metric",
        "primary_abs_error",
        "tolerance",
        "energy_abs_error",
        "pure_r2_relative_error",
        "pure_rms_relative_error",
        "pure_density_relative_l2",
        "density_bin_count",
        "density_shape_min_bins",
        "density_resolution_gate",
        "density_accounting_gate",
        "density_accounting_abs_error",
        "density_accounting_tolerance",
        "mixed_r2_relative_error_diagnostic",
        "mixed_rms_relative_error_diagnostic",
        "mixed_density_relative_l2_diagnostic",
        "pure_mixed_density_relative_l2_diagnostic",
        "mixed_density_closer_than_fw_to_exact_diagnostic",
        "passed",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path
