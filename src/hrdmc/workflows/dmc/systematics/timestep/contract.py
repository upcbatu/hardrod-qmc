from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.analysis.timestep_extrapolation import TimeStepPoint

TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION = "dmc_timestep_extrapolation_v4"
TIMESTEP_EXTRAPOLATION_RUN_NAME = "dmc_timestep_extrapolation"
SUPPORTED_INPUTS = {
    ("dmc_benchmark_packet", "dmc_benchmark_packet_v3"),
    ("dmc_trapped_stationarity_grid", "dmc_trapped_stationarity_grid_v2"),
}
PUBLICATION_READY_WORKFLOW_STATUSES = {
    "accepted",
    "accepted_with_warnings",
    "accepted_with_model_bound",
}
ENERGY_CHAIN_ACCEPTED_STATUSES = {"accepted", "spread_warning"}
ENERGY_REPORTING_POLICY_TIMINGS = {"prospective", "retrospective"}


@dataclass(frozen=True)
class EnergyReportingResolutionPolicy:
    """Explicit practical resolution used to qualify time-step model ambiguity."""

    resolution: float
    confidence_level: float
    energy_unit: str
    rationale: str
    timing: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("energy reporting resolution must be finite and positive")
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "energy reporting confidence level must lie strictly between zero and one"
            )
        if not self.energy_unit.strip():
            raise ValueError("energy reporting resolution unit must be non-empty")
        if not self.rationale.strip():
            raise ValueError("energy reporting resolution rationale must be non-empty")
        if self.timing not in ENERGY_REPORTING_POLICY_TIMINGS:
            raise ValueError(
                "energy reporting policy timing must be 'prospective' or 'retrospective'"
            )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "resolution": self.resolution,
            "confidence_level": self.confidence_level,
            "energy_unit": self.energy_unit,
            "rationale": self.rationale,
            "timing": self.timing,
        }


@dataclass(frozen=True)
class LoadedTimeStepPoint:
    point: TimeStepPoint
    case_id: str
    identity: dict[str, Any]
    summary_path: Path
    summary_sha256: str
    manifest_path: Path
    manifest_sha256: str
    run_name: str
    result_schema_version: str
    run_id: str
    bundle_sha256: str
    run_status: str
    energy_status: str
    energy_quality: dict[str, Any]
    energy_quality_assessment: dict[str, Any] | None
    seeds: tuple[int, ...]
    manifest_verification_warnings: tuple[str, ...]
    controls: dict[str, Any]
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.point.to_dict(),
            "case_id": self.case_id,
            "summary_path": str(self.summary_path),
            "summary_sha256": self.summary_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "run_name": self.run_name,
            "result_schema_version": self.result_schema_version,
            "run_id": self.run_id,
            "bundle_sha256": self.bundle_sha256,
            "run_status": self.run_status,
            "energy_status": self.energy_status,
            "energy_quality": self.energy_quality,
            "energy_quality_assessment": self.energy_quality_assessment,
            "seeds": list(self.seeds),
            "seed_count": len(self.seeds),
            "manifest_verification": (
                "summary_bound_with_unrelated_artifact_warnings"
                if self.manifest_verification_warnings
                else "verified"
            ),
            "manifest_verification_warnings": list(self.manifest_verification_warnings),
            "controls": self.controls,
            "telemetry": self.telemetry,
        }
