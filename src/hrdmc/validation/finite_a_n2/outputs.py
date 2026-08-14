from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import ensure_dir, write_csv, write_json_atomic, write_run_manifest
from hrdmc.system.settings import DMCRunControls, dmc_run_config


def write_finite_a_n2_reference_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
) -> list[Path]:
    root = ensure_dir(output_dir)
    summary_path = write_json_atomic(root / "summary.json", payload)
    table_path = _write_case_table(root / "case_table.csv", payload["case_results"])
    return [summary_path, table_path]
def write_finite_a_n2_reference_manifest(
    output_dir: Path,
    payload: dict[str, Any],
    artifacts: list[Path],
    controls: DMCRunControls,
    seeds: list[int],
    command: list[str],
) -> None:
    write_run_manifest(
        output_dir,
        run_name="finite_a_n2_reference_packet",
        config=dmc_run_config(
            run_kind="finite_a_n2_reference_packet",
            cases=[str(row["case_id"]) for row in payload["case_table"]],
            seeds=seeds,
            controls=controls,
            parallel_workers=int(payload["parallel_workers_requested"]),
        ),
        artifacts=artifacts,
        status=str(payload["status"]),
    )
def _write_case_table(path: Path, cases: list[dict[str, Any]]) -> Path:
    rows = [_case_row(case) for case in cases]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return write_csv(path, rows, fieldnames=fields)
def _case_row(case: dict[str, Any]) -> dict[str, Any]:
    comparison = case["comparison"]
    checks = comparison["checks"]
    energy = comparison["energy"]
    r2 = comparison["r2"]
    rms = comparison["rms"]
    density = comparison["density"]
    return {
        "case_id": case["case_id"],
        "status": case["status"],
        "benchmark_packet_status": case["benchmark_packet"]["status"],
        "energy_reference_status": checks["energy_reference"],
        "pure_r2_reference_status": checks["pure_r2_reference"],
        "pure_rms_reference_status": checks["pure_rms_reference"],
        "pure_density_reference_status": checks["pure_density_reference"],
        "density_accounting_status": checks["density_accounting"],
        "proposal_family": case.get("proposal_family", ""),
        "guide_family": case.get("guide_family", ""),
        "target_family": case.get("target_family", ""),
        "energy": energy["dmc"],
        "energy_reference": energy["reference"],
        "energy_abs_error": energy["abs_error"],
        "pure_r2": r2["pure_fw"],
        "r2_reference": r2["reference"],
        "pure_r2_relative_error": r2["pure_relative_error"],
        "rms_radius": rms["pure_fw"],
        "rms_reference": rms["reference"],
        "pure_rms_relative_error": rms["pure_relative_error"],
        "pure_density_relative_l2": density["pure_relative_l2"],
        "density_accounting_abs_error": density["density_accounting_abs_error"],
        "mixed_r2_relative_error_diagnostic": r2["mixed_relative_error_diagnostic"],
        "fw_r2_closer_than_mixed_diagnostic": r2["fw_closer_than_mixed_diagnostic"],
        "mixed_rms_relative_error_diagnostic": rms["mixed_relative_error_diagnostic"],
        "fw_rms_closer_than_mixed_diagnostic": rms["fw_closer_than_mixed_diagnostic"],
        "mixed_density_relative_l2_diagnostic": density["mixed_relative_l2_diagnostic"],
        "fw_density_closer_than_mixed_diagnostic": density["fw_closer_than_mixed_diagnostic"],
    }
