from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hrdmc.io.artifacts import ensure_dir


def write_benchmark_packet_seed_table(
    output_dir: Path,
    seed_payloads: list[dict[str, Any]],
) -> Path:
    fields = [
        "seed",
        "status",
        "rn_mixed_energy",
        "rn_r2_radius",
        "rn_rms_radius",
        "r2_schema_status",
        "r2_plateau_status",
        "r2_plateau_value",
        "paper_rms_radius",
        "lag_max_block_count",
        "lag_max_weight_ess_min",
    ]
    path = ensure_dir(output_dir) / "seed_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in seed_payloads:
            writer.writerow(_seed_table_row(payload))
    return path


def write_benchmark_packet_table(output_dir: Path, payload: dict[str, Any]) -> Path:
    path = ensure_dir(output_dir) / "packet_table.csv"
    paper = payload["paper_values"]
    row = {
        "case_id": payload["case_id"],
        "status": payload["status"],
        "energy_status": payload["energy_claim_status"],
        "pure_fw_status": payload["pure_fw_claim_status"],
        "energy": paper["energy"]["value"],
        "energy_stderr": paper["energy"]["stderr"],
        "energy_delta_vs_lda": paper["energy"]["delta_vs_lda"],
        "pure_r2": paper["r2"]["value"],
        "pure_r2_stderr": paper["r2"]["stderr"],
        "pure_r2_delta_vs_lda": paper["r2"]["delta_vs_lda"],
        "paper_rms_radius": paper["rms"]["value"],
        "paper_rms_radius_stderr": paper["rms"]["stderr"],
        "paper_rms_delta_vs_lda": paper["rms"]["delta_vs_lda"],
        "density_status": paper["density"]["status"],
        "pair_distance_density_status": paper["pair_distance_density"]["status"],
        "structure_factor_status": paper["structure_factor"]["status"],
        "mixed_density_l2_diagnostic": paper["density"]["mixed_diagnostic_density_l2"],
    }
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def _seed_table_row(payload: dict[str, Any]) -> dict[str, Any]:
    r2 = payload["pure_walking"]["observable_results"].get("r2", {})
    rn = payload["rn_summary"]
    lag_steps = r2.get("lag_steps", [])
    lag_max = lag_steps[-1] if lag_steps else ""
    return {
        "seed": payload["seed"],
        "status": payload["status"],
        "rn_mixed_energy": rn["mixed_energy"],
        "rn_r2_radius": rn["r2_radius"],
        "rn_rms_radius": rn["rms_radius"],
        "r2_schema_status": r2.get("schema_status", ""),
        "r2_plateau_status": r2.get("plateau_status", ""),
        "r2_plateau_value": r2.get("plateau_value", ""),
        "paper_rms_radius": r2.get("paper_rms_radius", ""),
        "lag_max_block_count": _lag_dict_get(r2.get("block_count_by_lag", {}), lag_max),
        "lag_max_weight_ess_min": _lag_dict_get(
            r2.get("block_weight_ess_min_by_lag", {}),
            lag_max,
        ),
    }


def _lag_dict_get(values: object, lag: object) -> object:
    if not isinstance(values, dict):
        return ""
    return values.get(lag, values.get(str(lag), ""))
