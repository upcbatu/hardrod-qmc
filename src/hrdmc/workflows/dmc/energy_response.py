from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from hrdmc.estimators import (
    EnergyResponsePoint,
    lambda_from_omega,
    trap_r2_from_energy_response,
)
from hrdmc.io.artifacts import ensure_dir
from hrdmc.workflows.dmc.rn_block import RNCase

ESSENTIAL_POINT_FIELDS = ("omega", "mixed_energy")
PAPER_GATE_FIELDS = (
    "mixed_energy_conservative_stderr",
    "rn_weight_status",
    "density_accounting_clean",
    "valid_finite_clean",
    "blocking_plateau_energy",
    "stationarity_energy",
)


def reanalyze_trap_r2_energy_response(
    *,
    base_case: RNCase,
    summary_paths: list[Path],
    degree: int,
) -> dict[str, Any]:
    """Fit HF trap R2/RMS from existing RN-DMC summary artifacts only."""

    rows = collect_energy_response_rows(summary_paths, base_case=base_case)
    if len(rows) < degree + 1:
        raise ValueError("not enough existing energy points for requested polynomial degree")
    points = tuple(
        EnergyResponsePoint(
            lambda_value=float(row["lambda_value"]),
            omega=float(row["omega"]),
            energy=float(row["mixed_energy"]),
            energy_stderr=_optional_float(row.get("mixed_energy_conservative_stderr")),
            label=str(row["case_id"]),
            metadata={
                "source_summary": row["source_summary"],
                "energy_point_gate": row["energy_point_gate"],
                "final_classification": row.get("final_classification", ""),
                "rn_weight_status": row.get("rn_weight_status", ""),
            },
        )
        for row in rows
    )
    estimator_result = trap_r2_from_energy_response(
        points,
        n_particles=base_case.n_particles,
        omega0=base_case.omega,
        degree=degree,
    )
    point_gate_all = all(row["energy_point_gate"] == "ENERGY_POINT_GO" for row in rows)
    response_status = (
        estimator_result.fit_response_status
        if point_gate_all
        else "ENERGY_RESPONSE_POINT_NO_GO"
    )
    return {
        "schema_version": "trap_r2_energy_response_reanalysis_v1",
        "analysis_mode": "offline_summary_reanalysis",
        "base_case": base_case.case_id,
        "n_particles": base_case.n_particles,
        "rod_length": base_case.rod_length,
        "omega0": base_case.omega,
        "lambda0": lambda_from_omega(base_case.omega),
        "summary_paths": [str(path) for path in summary_paths],
        "point_count": len(rows),
        "energy_point_gate_all": point_gate_all,
        "fit_response_status": estimator_result.fit_response_status,
        "response_status": response_status,
        "paper_grade_eligible": response_status == "ENERGY_RESPONSE_GO",
        "energy_response": estimator_result,
        "points": rows,
        "claim_boundary": (
            "Offline Hellmann-Feynman trap R2/RMS reanalysis of existing RN-DMC "
            "energy artifacts; no DMC sampling was performed. Paper-grade status "
            "requires every energy point to carry and pass RN-DMC gate metadata."
        ),
    }


def collect_energy_response_rows(
    summary_paths: list[Path],
    *,
    base_case: RNCase,
) -> list[dict[str, Any]]:
    """Collect lambda-energy rows from RN stationarity or prior response summaries."""

    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _payload_case_rows(payload):
            rows.append(_point_row_from_case_row(row, source_summary=path, base_case=base_case))
    rows.sort(key=lambda row: float(row["lambda_value"]))
    _validate_unique_lambdas(rows)
    return rows


def energy_point_gate_from_row(row: dict[str, Any]) -> str:
    """Classify one existing DMC energy point for HF paper-grade eligibility."""

    missing = [field for field in PAPER_GATE_FIELDS if field not in row]
    if missing:
        return "ENERGY_POINT_METADATA_UNAVAILABLE"
    if row.get("rn_weight_status") == "RN_WEIGHT_NO_GO":
        return "ENERGY_POINT_RN_WEIGHT_NO_GO"
    if row.get("rn_weight_status") != "RN_WEIGHT_GO":
        return "ENERGY_POINT_RN_WEIGHT_WARNING"
    if not bool(row.get("density_accounting_clean", False)):
        return "ENERGY_POINT_DENSITY_ACCOUNTING_NO_GO"
    if not bool(row.get("valid_finite_clean", False)):
        return "ENERGY_POINT_HYGIENE_NO_GO"
    if row.get("stationarity_energy") == "NO_GO_STATIONARITY":
        return "ENERGY_POINT_STATIONARITY_NO_GO"
    if not bool(row.get("blocking_plateau_energy", False)):
        return "ENERGY_POINT_BLOCKING_NO_GO"
    return "ENERGY_POINT_GO"


def write_response_point_table(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    fields = [
        "case_id",
        "relative_lambda_offset",
        "lambda_value",
        "omega",
        "mixed_energy",
        "mixed_energy_conservative_stderr",
        "energy_point_gate",
        "final_classification",
        "rn_weight_status",
        "stationarity_energy",
        "blocking_plateau_energy",
        "density_accounting_clean",
        "valid_finite_clean",
        "guide_batch_backend",
        "source_summary",
    ]
    path = ensure_dir(output_dir) / "energy_response_points.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _payload_case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("points"), list):
        return [
            dict(point.get("case_row", point))
            for point in payload["points"]
            if isinstance(point, dict)
        ]
    if isinstance(payload.get("cases"), list):
        return [dict(row) for row in payload["cases"] if isinstance(row, dict)]
    if "mixed_energy" in payload:
        return [dict(payload)]
    raise ValueError("summary payload does not contain RN-DMC case rows")


def _point_row_from_case_row(
    row: dict[str, Any],
    *,
    source_summary: Path,
    base_case: RNCase,
) -> dict[str, Any]:
    _require_fields(row, source_summary=source_summary, fields=ESSENTIAL_POINT_FIELDS)
    omega = float(row["omega"])
    lambda_value = float(row.get("lambda_value", lambda_from_omega(omega)))
    lambda0 = lambda_from_omega(base_case.omega)
    out = {
        "case_id": str(
            row.get(
                "case_id",
                f"N{base_case.n_particles}_a{base_case.rod_length:g}_omega{omega:g}",
            )
        ),
        "omega": omega,
        "lambda_value": lambda_value,
        "relative_lambda_offset": lambda_value / lambda0 - 1.0,
        "mixed_energy": float(row["mixed_energy"]),
        "mixed_energy_conservative_stderr": _optional_float(
            row.get("mixed_energy_conservative_stderr")
        ),
        "final_classification": row.get("final_classification", ""),
        "rn_weight_status": row.get("rn_weight_status", ""),
        "stationarity_energy": row.get("stationarity_energy", ""),
        "blocking_plateau_energy": row.get("blocking_plateau_energy", False),
        "density_accounting_clean": row.get("density_accounting_clean", False),
        "valid_finite_clean": row.get("valid_finite_clean", False),
        "guide_batch_backend": row.get("guide_batch_backend", ""),
        "source_summary": str(source_summary),
    }
    out["energy_point_gate"] = energy_point_gate_from_row(row)
    return out


def _require_fields(
    row: dict[str, Any],
    *,
    source_summary: Path,
    fields: tuple[str, ...],
) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source_summary} is missing required HF response fields: {joined}")


def _validate_unique_lambdas(rows: list[dict[str, Any]]) -> None:
    values = [float(row["lambda_value"]) for row in rows]
    if len(set(values)) != len(values):
        raise ValueError("summary inputs contain duplicate lambda values")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
