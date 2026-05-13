from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hrdmc.io.artifacts import ensure_dir


def write_pure_walking_seed_table(output_dir: Path, seed_payloads: list[dict[str, Any]]) -> Path:
    fields = [
        "seed",
        "status",
        "rn_mixed_energy",
        "rn_r2_radius",
        "rn_rms_radius",
        "r2_schema_status",
        "r2_plateau_status",
        "r2_plateau_value",
        "r2_plateau_stderr",
        "paper_rms_radius",
        "paper_rms_radius_stderr",
        "lag_max_block_count",
        "lag_max_weight_ess_min",
    ]
    path = ensure_dir(output_dir) / "seed_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in seed_payloads:
            writer.writerow(seed_table_row(payload))
    return path


def seed_table_row(payload: dict[str, Any]) -> dict[str, Any]:
    pure = payload["pure_walking"]
    rn = payload["rn_summary"]
    r2 = pure["observable_results"].get("r2", {})
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
        "r2_plateau_stderr": r2.get("plateau_stderr", ""),
        "paper_rms_radius": r2.get("paper_rms_radius", ""),
        "paper_rms_radius_stderr": r2.get("paper_rms_radius_stderr", ""),
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
