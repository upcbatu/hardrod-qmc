from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.analysis import (
    CHAIN_ACCEPTED,
    CORRELATED_ERROR_AGREEMENT,
    CORRELATED_ERROR_DISAGREEMENT,
)
from hrdmc.artifacts import (
    build_run_provenance,
    config_fingerprint,
    ensure_dir,
    file_sha256,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.estimators import (
    BASE_TRAP_QUADRATIC_COUPLING,
    PairedEnergyResponsePoint,
    lambda_from_relative_offset,
    paired_trap_r2_from_energy_response,
)
from hrdmc.workflows.dmc.guide_validation import (
    validate_production_contact_guide_binding,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.stationarity import summarize_stationarity_case
from hrdmc.workflows.dmc.trapped import (
    DMCRunControls,
    TrappedCase,
    controls_to_dict,
    dmc_progress_bar,
)

DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION = "dmc_energy_response_point_v3"
ENERGY_RESPONSE_REANALYSIS_SCHEMA_VERSION = "trap_r2_energy_response_reanalysis_v4"
ENERGY_RESPONSE_PLAN_SCHEMA_VERSION = "dmc_energy_response_ladder_plan_v3"
ENERGY_RESPONSE_OFFLINE_PLAN_SCHEMA_VERSION = "dmc_energy_response_reanalysis_plan_v1"
ACCEPTED_FINITE_DIFFERENCE_STATUSES = {
    "scale_shift_within_r2_confidence_half_width",
    "difference_scales_identical",
}


@dataclass(frozen=True)
class EnergyResponseLadderControls:
    """Scientific controls shared by every point in a paired HF ladder.

    The ESS and weight-span thresholds are recorded numerical diagnostics,
    not universal physical constants. They are deliberately explicit so a
    result cannot inherit a different acceptance policy during resume.
    """

    run: DMCRunControls
    seeds: tuple[int, ...]
    relative_lambda_offsets: tuple[float, ...]
    initialization: InitializationControls
    guide_family: str
    parallel_workers: int | None
    ess_warning_fraction: float = 0.20
    ess_invalid_fraction: float = 0.10
    log_weight_span_warning: float = 50.0
    guide_parameter_source: str = "explicit"
    guide_parameter_source_sha256: str | None = None
    guide_parameter_source_manifest_sha256: str | None = None
    guide_parameter_source_identity_fingerprint: str | None = None

    def validate(self) -> None:
        if self.run.response_lambda is not None:
            raise ValueError("base HF run controls must not set response_lambda")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("HF response requires at least two unique paired seeds")
        validate_five_point_offsets(self.relative_lambda_offsets)
        self.initialization.validate()
        if self.parallel_workers is not None and self.parallel_workers < 0:
            raise ValueError("parallel_workers must be non-negative")
        if not 0.0 <= self.ess_invalid_fraction < self.ess_warning_fraction <= 1.0:
            raise ValueError("ESS thresholds must satisfy 0 <= invalid < warning <= 1")
        if not math.isfinite(self.log_weight_span_warning) or self.log_weight_span_warning <= 0:
            raise ValueError("log_weight_span_warning must be finite and positive")
        if self.guide_family not in {
            "reduced-tg",
            "contact-corrected-reduced-tg",
        }:
            raise ValueError("HF response supports reduced-TG guide families only")
        if self.run.relative_alpha is None or self.run.relative_alpha <= 0.0:
            raise ValueError("HF response requires an explicit positive relative_alpha")
        if self.guide_family == "contact-corrected-reduced-tg":
            if self.run.contact_beta is None:
                raise ValueError("contact-corrected HF response requires contact_beta")
            if self.guide_parameter_source == "explicit" or any(
                digest is None
                for digest in (
                    self.guide_parameter_source_sha256,
                    self.guide_parameter_source_manifest_sha256,
                    self.guide_parameter_source_identity_fingerprint,
                )
            ):
                raise ValueError(
                    "contact-corrected HF response requires a validated guide identity"
                )
        elif self.run.contact_beta is not None:
            raise ValueError("reduced-tg HF response must not set contact_beta")
        elif self.guide_parameter_source != "explicit" or any(
            digest is not None
            for digest in (
                self.guide_parameter_source_sha256,
                self.guide_parameter_source_manifest_sha256,
                self.guide_parameter_source_identity_fingerprint,
            )
        ):
            raise ValueError("reduced-tg HF response must use explicit guide parameters")
        for field_name, digest in (
            ("guide parameter source", self.guide_parameter_source_sha256),
            ("guide parameter source manifest", self.guide_parameter_source_manifest_sha256),
            (
                "guide parameter source identity fingerprint",
                self.guide_parameter_source_identity_fingerprint,
            ),
        ):
            if digest is None:
                continue
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{field_name} SHA-256 is invalid")


def run_energy_response_ladder(
    *,
    case: TrappedCase,
    controls: EnergyResponseLadderControls,
    output_dir: Path,
    command: list[str] | None,
    resume: bool = False,
    dry_run: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Run or resume a fixed-guide paired five-point HF energy ladder."""

    controls.validate()
    validate_production_contact_guide_binding(
        case=case,
        guide_family=controls.guide_family,
        relative_alpha=controls.run.relative_alpha,
        contact_beta=controls.run.contact_beta,
        source=controls.guide_parameter_source,
        source_sha256=controls.guide_parameter_source_sha256,
        source_manifest_sha256=controls.guide_parameter_source_manifest_sha256,
        source_identity_fingerprint=(controls.guide_parameter_source_identity_fingerprint),
    )
    root = output_dir.resolve()
    provenance = build_run_provenance(command)
    implementation = _required_implementation_identity(provenance)
    runtime = _required_runtime_identity(provenance)
    plan = build_energy_response_plan(
        case,
        controls,
        implementation=implementation,
        runtime=runtime,
    )
    if dry_run:
        return {
            "status": "planned",
            "plan": plan,
            "output_dir": str(root),
        }

    _prepare_ladder_output(root, plan=plan, resume=resume)
    plan_path = root / "plan.json"
    point_paths: list[Path] = []
    shared_controls = shared_run_controls(case, controls)
    guide = guide_identity(case, controls)
    for offset in controls.relative_lambda_offsets:
        point_dir = root / response_offset_label(offset)
        point_path = point_dir / "summary.json"
        point_controls = replace(
            controls.run,
            response_lambda=lambda_from_relative_offset(offset),
        )
        expected = expected_point_identity(
            case=case,
            relative_lambda_offset=offset,
            run_controls=point_controls,
            guide=guide,
            shared_controls=shared_controls,
            implementation=implementation,
            runtime=runtime,
        )
        if resume and point_path.exists():
            _load_verified_completed_point(point_path, expected=expected)
            point_paths.append(point_path)
            continue
        if point_dir.exists() and any(point_dir.iterdir()):
            raise ValueError(
                f"response point directory is incomplete or mismatched: {point_dir}; "
                "use a new output directory"
            )
        point_dir.mkdir(parents=True, exist_ok=True)
        with dmc_progress_bar(
            controls=point_controls,
            seed_count=len(controls.seeds),
            label=f"HF response {case.case_id} offset={offset:+g}",
            enabled=progress,
        ) as bar:
            stationarity = summarize_stationarity_case(
                case,
                point_controls,
                list(controls.seeds),
                parallel_workers=controls.parallel_workers,
                progress=bar,
                trace_output_dir=None,
                ess_warning_fraction=controls.ess_warning_fraction,
                ess_invalid_fraction=controls.ess_invalid_fraction,
                log_weight_span_warning=controls.log_weight_span_warning,
                initialization=controls.initialization,
                collective_rn=None,
                guide_family=controls.guide_family,
            )
        point = build_energy_response_point(
            case=case,
            relative_lambda_offset=offset,
            stationarity=stationarity,
            run_controls=point_controls,
            guide=guide,
            shared_controls=shared_controls,
            implementation=implementation,
            runtime=runtime,
        )
        write_json(point_path, point)
        write_run_manifest(
            point_dir,
            run_name="dmc_energy_response_point",
            config=response_point_manifest_config(point),
            artifacts=[point_path],
            schema_version=DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION,
            provenance=provenance,
            status=str(point["energy_point_status"]),
        )
        point_paths.append(point_path)

    response = reanalyze_trap_r2_energy_response(
        base_case=case,
        summary_paths=point_paths,
    )
    response["plan"] = plan
    response_path = root / "summary.json"
    table_path = write_response_point_table(root, response["points"])
    response["artifacts"] = {
        "summary": str(response_path),
        "point_table": str(table_path),
        "plan": str(plan_path),
        "output_dir": str(root),
    }
    write_json(response_path, response)
    write_run_manifest(
        root,
        run_name="dmc_energy_response_ladder",
        config=plan,
        artifacts=[plan_path, response_path, table_path, *point_paths],
        schema_version=ENERGY_RESPONSE_REANALYSIS_SCHEMA_VERSION,
        provenance=provenance,
        status=str(response["status"]),
    )
    return response


def build_energy_response_plan(
    case: TrappedCase,
    controls: EnergyResponseLadderControls,
    *,
    implementation: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    controls.validate()
    shared_controls = shared_run_controls(case, controls)
    guide = guide_identity(case, controls)
    identity: dict[str, Any] = {
        "schema_version": ENERGY_RESPONSE_PLAN_SCHEMA_VERSION,
        "case_id": case.case_id,
        "base_lambda": BASE_TRAP_QUADRATIC_COUPLING,
        "relative_lambda_offsets": list(controls.relative_lambda_offsets),
        "lambda_values": [
            lambda_from_relative_offset(offset) for offset in controls.relative_lambda_offsets
        ],
        "paired_seeds": list(controls.seeds),
        "guide": guide,
        "guide_fingerprint": config_fingerprint(guide),
        "guide_parameter_source": controls.guide_parameter_source,
        "guide_parameter_source_sha256": controls.guide_parameter_source_sha256,
        "guide_parameter_source_manifest_sha256": (controls.guide_parameter_source_manifest_sha256),
        "guide_parameter_source_identity_fingerprint": (
            controls.guide_parameter_source_identity_fingerprint
        ),
        "shared_run_controls": shared_controls,
        "run_controls_fingerprint": config_fingerprint(shared_controls),
        "implementation": implementation,
        "implementation_fingerprint": _implementation_fingerprint(implementation),
        "runtime": runtime,
        "runtime_fingerprint": config_fingerprint(runtime),
        "method": "fixed-guide local DMC paired five-point energy response",
    }
    identity["plan_fingerprint"] = _energy_response_plan_fingerprint(identity)
    return identity


def shared_run_controls(
    case: TrappedCase,
    controls: EnergyResponseLadderControls,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "paired_seeds": list(controls.seeds),
        "base_run_controls": controls_to_dict(controls.run),
        "initialization": asdict(controls.initialization),
        "parallel_workers": controls.parallel_workers,
        "guide_parameter_identity": guide_parameter_identity(controls),
        "diagnostic_thresholds": {
            "ess_warning_fraction": controls.ess_warning_fraction,
            "ess_invalid_fraction": controls.ess_invalid_fraction,
            "log_weight_span_warning": controls.log_weight_span_warning,
            "interpretation": "recorded conservative numerical diagnostics",
        },
    }


def guide_identity(
    case: TrappedCase,
    controls: EnergyResponseLadderControls,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "guide_family": controls.guide_family,
        "relative_alpha": controls.run.relative_alpha,
        "contact_beta": controls.run.contact_beta,
        "parameter_source": guide_parameter_identity(controls),
    }


def guide_parameter_identity(
    controls: EnergyResponseLadderControls,
) -> dict[str, str | None]:
    """Return the portable identity of the parameters held fixed by the ladder."""

    return {
        "kind": (
            "validated_contact_guide"
            if controls.guide_family == "contact-corrected-reduced-tg"
            else "explicit"
        ),
        "summary_sha256": controls.guide_parameter_source_sha256,
        "manifest_sha256": controls.guide_parameter_source_manifest_sha256,
        "identity_fingerprint": controls.guide_parameter_source_identity_fingerprint,
    }


def build_energy_response_point(
    *,
    case: TrappedCase,
    relative_lambda_offset: float,
    stationarity: dict[str, Any],
    run_controls: DMCRunControls,
    guide: dict[str, Any],
    shared_controls: dict[str, Any],
    implementation: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Build one HF point after checking what every seed actually sampled."""

    lambda_value = lambda_from_relative_offset(relative_lambda_offset)
    if run_controls.response_lambda is None or not np.isclose(
        run_controls.response_lambda,
        lambda_value,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("run controls do not sample the planned response lambda")
    _validate_stationarity_binding(
        case=case,
        stationarity=stationarity,
        run_controls=run_controls,
        guide=guide,
    )
    raw_seed_rows = stationarity["seed_summaries"]
    seed_energies = [
        {"seed": int(row["seed"]), "energy": float(row["mixed_energy"])} for row in raw_seed_rows
    ]
    point_controls = controls_to_dict(run_controls)
    payload: dict[str, Any] = {
        "schema_version": DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION,
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "lambda0": BASE_TRAP_QUADRATIC_COUPLING,
        "relative_lambda_offset": float(relative_lambda_offset),
        "lambda_value": lambda_value,
        "guide": guide,
        "guide_fingerprint": config_fingerprint(guide),
        "point_run_controls": point_controls,
        "point_controls_fingerprint": config_fingerprint(point_controls),
        "shared_run_controls": shared_controls,
        "run_controls_fingerprint": config_fingerprint(shared_controls),
        "implementation": implementation,
        "implementation_fingerprint": _implementation_fingerprint(implementation),
        "runtime": runtime,
        "runtime_fingerprint": config_fingerprint(runtime),
        "mixed_energy": float(stationarity["mixed_energy"]),
        "mixed_energy_conservative_stderr": float(stationarity["mixed_energy_conservative_stderr"]),
        "population_weight_status": stationarity["population_weight_status"],
        "valid_finite_clean": bool(stationarity["valid_finite_clean"]),
        "stationarity_energy": stationarity["stationarity_energy"],
        "blocking_plateau_energy": bool(stationarity.get("blocking_plateau_energy", False)),
        "final_classification": stationarity.get("final_classification", ""),
        "method_status": stationarity.get("method_status", ""),
        "mixed_energy_error_estimator_status": stationarity.get(
            "mixed_energy_error_estimator_status",
            "",
        ),
        "seed_count": len(seed_energies),
        "seed_energies": seed_energies,
        "guide_batch_backend": stationarity.get("guide_batch_backend", ""),
        "target_backend": stationarity.get("target_backend", ""),
        "proposal_backend": stationarity.get("proposal_backend", ""),
        "resolved_guide_family": stationarity["resolved_guide_family"],
        "method": "fixed-guide local DMC energy response",
    }
    payload["energy_point_status"] = energy_point_status_from_row(payload)
    payload["point_identity_fingerprint"] = config_fingerprint(
        expected_point_identity(
            case=case,
            relative_lambda_offset=relative_lambda_offset,
            run_controls=run_controls,
            guide=guide,
            shared_controls=shared_controls,
            implementation=implementation,
            runtime=runtime,
        )
    )
    return payload


def reanalyze_trap_r2_energy_response(
    *,
    base_case: TrappedCase,
    summary_paths: list[Path],
    confidence_level: float = 0.95,
    minimum_seed_count: int = 2,
) -> dict[str, Any]:
    """Aggregate a manifest-verified paired fixed-guide HF response."""

    rows = collect_energy_response_rows(summary_paths, base_case=base_case)
    points = tuple(
        PairedEnergyResponsePoint(
            seed=int(seed_energy["seed"]),
            relative_lambda_offset=float(row["relative_lambda_offset"]),
            lambda_value=float(row["lambda_value"]),
            energy=float(seed_energy["energy"]),
            label=str(row["case_id"]),
            metadata={"source_summary": row["source_summary"]},
        )
        for row in rows
        for seed_energy in row["seed_energies"]
    )
    estimator_result = paired_trap_r2_from_energy_response(
        points,
        n_particles=base_case.n_particles,
        lambda0=BASE_TRAP_QUADRATIC_COUPLING,
        confidence_level=confidence_level,
        minimum_seed_count=minimum_seed_count,
    )
    all_points_valid = all(row["energy_point_status"] == "accepted" for row in rows)
    if not all_points_valid:
        status = "energy_point_validity_failure"
    elif estimator_result.radius_status != "positive_r2":
        status = "nonpositive_r2"
    elif estimator_result.pure_r2_confidence_interval[0] <= 0.0:
        status = "r2_confidence_interval_crosses_zero"
    elif estimator_result.finite_difference_status not in ACCEPTED_FINITE_DIFFERENCE_STATUSES:
        status = "finite_difference_scale_unresolved"
    else:
        status = "accepted"
    return {
        "schema_version": ENERGY_RESPONSE_REANALYSIS_SCHEMA_VERSION,
        "status": status,
        "analysis_mode": "offline_manifest_verified_paired_seed_reanalysis",
        "base_case": base_case.case_id,
        "n_particles": base_case.n_particles,
        "rod_length": base_case.rod_length,
        "lambda0": BASE_TRAP_QUADRATIC_COUPLING,
        "summary_paths": [str(path) for path in summary_paths],
        "point_count": len(rows),
        "seed_count": estimator_result.seed_count,
        "guide_fingerprint": rows[0]["guide_fingerprint"],
        "guide_parameter_identity": rows[0]["guide"]["parameter_source"],
        "run_controls_fingerprint": rows[0]["run_controls_fingerprint"],
        "implementation_fingerprint": rows[0]["implementation_fingerprint"],
        "runtime_fingerprint": rows[0]["runtime_fingerprint"],
        "all_points_valid": all_points_valid,
        "point_validity_status": (
            "all_energy_points_valid" if all_points_valid else "energy_point_validity_failure"
        ),
        "radius_status": estimator_result.radius_status,
        "ladder_status": estimator_result.ladder_status,
        "finite_difference_status": estimator_result.finite_difference_status,
        "energy_response": asdict(estimator_result),
        "points": rows,
        "method": {
            "observable": "trap mean-squared radius and RMS radius",
            "response": "paired symmetric five-point Hellmann-Feynman response",
            "extrapolation": "Richardson extrapolation per seed",
            "uncertainty": "Student-t interval across paired seeds; scale shift separate",
        },
    }


def run_energy_response_reanalysis(
    *,
    base_case: TrappedCase,
    summary_paths: list[Path],
    output_dir: Path | None,
    command: list[str] | None,
    write: bool,
    confidence_level: float = 0.95,
    minimum_seed_count: int = 2,
) -> dict[str, Any]:
    """Reanalyze verified HF points and optionally write one immutable result bundle."""

    payload = reanalyze_trap_r2_energy_response(
        base_case=base_case,
        summary_paths=summary_paths,
        confidence_level=confidence_level,
        minimum_seed_count=minimum_seed_count,
    )
    provenance = build_run_provenance(command)
    plan = _build_offline_reanalysis_plan(
        base_case=base_case,
        summary_paths=summary_paths,
        confidence_level=confidence_level,
        minimum_seed_count=minimum_seed_count,
        provenance=provenance,
    )
    payload["plan"] = plan
    if not write:
        payload["artifacts"] = {}
        return payload
    if output_dir is None:
        raise ValueError("output_dir is required when writing HF reanalysis")
    root = output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}; use a new path")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    table_path = write_response_point_table(root, payload["points"])
    summary_path = root / "summary.json"
    payload["artifacts"] = {
        "summary": str(summary_path),
        "point_table": str(table_path),
        "plan": str(plan_path),
        "output_dir": str(root),
    }
    write_json(summary_path, payload)
    write_run_manifest(
        root,
        run_name="dmc_energy_response_reanalysis",
        config=plan,
        artifacts=[plan_path, summary_path, table_path],
        schema_version=ENERGY_RESPONSE_REANALYSIS_SCHEMA_VERSION,
        provenance=provenance,
        status=str(payload["status"]),
    )
    return payload


def _build_offline_reanalysis_plan(
    *,
    base_case: TrappedCase,
    summary_paths: list[Path],
    confidence_level: float,
    minimum_seed_count: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    sources = []
    for summary_path in summary_paths:
        resolved = summary_path.expanduser().resolve()
        manifest_path = resolved.parent / "run_manifest.json"
        sources.append(
            {
                "summary_path": str(resolved),
                "summary_sha256": file_sha256(resolved),
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            }
        )
    identity: dict[str, Any] = {
        "schema_version": ENERGY_RESPONSE_OFFLINE_PLAN_SCHEMA_VERSION,
        "base_case": base_case.case_id,
        "confidence_level": confidence_level,
        "minimum_seed_count": minimum_seed_count,
        "sources": sources,
        "implementation": _required_implementation_identity(provenance),
        "runtime": _required_runtime_identity(provenance),
        "method": "offline manifest-verified paired-seed energy response",
    }
    identity["plan_fingerprint"] = config_fingerprint(identity)
    return identity


def collect_energy_response_rows(
    summary_paths: list[Path],
    *,
    base_case: TrappedCase,
) -> list[dict[str, Any]]:
    if not summary_paths:
        raise ValueError("summary_paths must contain the five response-point packets")
    rows = [
        _point_row_from_packet(
            json.loads(path.read_text(encoding="utf-8")),
            source_summary=path,
            base_case=base_case,
        )
        for path in summary_paths
    ]
    rows.sort(key=lambda row: float(row["relative_lambda_offset"]))
    _validate_unique_response_points(rows)
    _validate_shared_identity(rows)
    _validate_shared_seed_set(rows)
    return rows


def energy_point_status_from_row(row: dict[str, Any]) -> str:
    stderr = _optional_float(row.get("mixed_energy_conservative_stderr"))
    if stderr is None or not math.isfinite(stderr) or stderr <= 0.0:
        return "uncertainty_metadata_missing"
    population_status = row.get("population_weight_status")
    if population_status in {None, ""}:
        return "population_weight_metadata_missing"
    if population_status != "accepted":
        return str(population_status)
    if "valid_finite_clean" not in row or "stationarity_energy" not in row:
        return "stationarity_metadata_missing"
    if not bool(row["valid_finite_clean"]):
        return "nonfinite_samples"
    if row.get("method_status") != "accepted":
        return "method_validation_failure"
    stationarity_status = str(row["stationarity_energy"])
    if stationarity_status == "spread_warning":
        return "seed_spread_warning"
    if stationarity_status != CHAIN_ACCEPTED:
        return "energy_trace_nonstationary"
    uncertainty_status = energy_uncertainty_status(row)
    if uncertainty_status != "accepted":
        return uncertainty_status
    return "accepted"


def energy_uncertainty_status(row: dict[str, Any]) -> str:
    if bool(row.get("blocking_plateau_energy", False)):
        return "accepted"
    stderr = _optional_float(row.get("mixed_energy_conservative_stderr"))
    if stderr is None or not math.isfinite(stderr) or stderr <= 0.0:
        return "correlated_error_unresolved"
    correlated_status = row.get("mixed_energy_error_estimator_status")
    if correlated_status == CORRELATED_ERROR_AGREEMENT:
        return "accepted"
    if correlated_status == CORRELATED_ERROR_DISAGREEMENT:
        return "correlated_error_estimates_disagree"
    return "correlated_error_unresolved"


def expected_point_identity(
    *,
    case: TrappedCase,
    relative_lambda_offset: float,
    run_controls: DMCRunControls,
    guide: dict[str, Any],
    shared_controls: dict[str, Any],
    implementation: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION,
        "case_id": case.case_id,
        "relative_lambda_offset": float(relative_lambda_offset),
        "lambda_value": lambda_from_relative_offset(relative_lambda_offset),
        "guide_fingerprint": config_fingerprint(guide),
        "point_controls_fingerprint": config_fingerprint(controls_to_dict(run_controls)),
        "run_controls_fingerprint": config_fingerprint(shared_controls),
        "implementation_fingerprint": _implementation_fingerprint(implementation),
        "runtime_fingerprint": config_fingerprint(runtime),
    }


def response_point_manifest_config(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "point_identity_fingerprint": point["point_identity_fingerprint"],
        "case_id": point["case_id"],
        "relative_lambda_offset": point["relative_lambda_offset"],
        "lambda_value": point["lambda_value"],
        "guide": point["guide"],
        "point_run_controls": point["point_run_controls"],
        "shared_run_controls": point["shared_run_controls"],
        "implementation_fingerprint": point["implementation_fingerprint"],
        "runtime_fingerprint": point["runtime_fingerprint"],
    }


def validate_five_point_offsets(offsets: tuple[float, ...]) -> None:
    clean = sorted(float(offset) for offset in offsets)
    if len(clean) != 5 or any(not math.isfinite(offset) for offset in clean):
        raise ValueError("response ladder requires exactly five finite offsets")
    if len(set(clean)) != 5 or any(1.0 + offset <= 0.0 for offset in clean):
        raise ValueError("response offsets must be unique and keep lambda positive")
    inner = clean[3]
    expected = [-2.0 * inner, -inner, 0.0, inner, 2.0 * inner]
    if inner <= 0.0 or any(
        not math.isclose(observed, target, rel_tol=0.0, abs_tol=1.0e-15)
        for observed, target in zip(clean, expected, strict=True)
    ):
        raise ValueError("response ladder must be exactly [-2h,-h,0,h,2h]")


def parse_five_point_offsets(value: str) -> tuple[float, ...]:
    offsets = tuple(sorted(float(item) for item in value.split(",") if item.strip()))
    validate_five_point_offsets(offsets)
    return offsets


def response_offset_label(offset: float) -> str:
    sign = "p" if offset >= 0.0 else "m"
    magnitude = f"{abs(offset):.10g}".replace(".", "p")
    return f"lambda_offset_{sign}{magnitude}"


def write_response_point_table(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    fields = [
        "case_id",
        "lambda0",
        "relative_lambda_offset",
        "lambda_value",
        "local_step_method",
        "drift_limiter",
        "mixed_energy",
        "mixed_energy_conservative_stderr",
        "seed_count",
        "energy_point_status",
        "guide_fingerprint",
        "point_controls_fingerprint",
        "run_controls_fingerprint",
        "implementation_fingerprint",
        "runtime_fingerprint",
        "guide_batch_backend",
        "target_backend",
        "proposal_backend",
        "source_summary",
    ]
    path = ensure_dir(output_dir) / "energy_response_points.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field, "") for field in fields}
            point_controls = row.get("point_run_controls")
            if isinstance(point_controls, dict):
                values["local_step_method"] = point_controls.get("local_step_method", "")
                values["drift_limiter"] = point_controls.get("drift_limiter", "none")
            writer.writerow(values)
    return path


def _prepare_ladder_output(root: Path, *, plan: dict[str, Any], resume: bool) -> None:
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {root}; use --resume or a new path")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing.get("plan_fingerprint")
        if existing_fingerprint != _energy_response_plan_fingerprint(existing):
            raise ValueError("existing HF plan fingerprint is inconsistent with its payload")
        if existing_fingerprint != plan["plan_fingerprint"]:
            raise ValueError("existing HF plan does not match current controls or implementation")
    elif resume and any(root.iterdir()):
        raise ValueError("cannot resume an HF output directory without plan.json")
    else:
        write_json(plan_path, plan)


def _load_verified_completed_point(
    point_path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = point_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"completed response point has no manifest: {point_path}")
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError(f"response point manifest failed verification: {'; '.join(errors)}")
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {
        str(entry.get("path")) for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if point_path.name not in artifact_paths:
        raise ValueError(f"response point summary is not covered by its manifest: {point_path}")
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"completed response point identity mismatch: {point_path}")
    expected_fingerprint = config_fingerprint(expected)
    if payload.get("point_identity_fingerprint") != expected_fingerprint:
        raise ValueError(f"response point identity fingerprint mismatch: {point_path}")
    if manifest.get("run_name") != "dmc_energy_response_point":
        raise ValueError(f"unexpected response point manifest owner: {manifest_path}")
    if manifest.get("result_schema_version") != DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION:
        raise ValueError(f"unexpected response point result schema: {manifest_path}")
    if manifest.get("config") != response_point_manifest_config(payload):
        raise ValueError(f"response point manifest config mismatch: {manifest_path}")
    provenance_implementation = manifest.get("provenance", {}).get("implementation", {})
    if (
        _implementation_fingerprint(provenance_implementation)
        != expected["implementation_fingerprint"]
    ):
        raise ValueError(f"response point implementation mismatch: {manifest_path}")
    provenance_runtime = _required_runtime_identity(manifest.get("provenance", {}))
    if config_fingerprint(provenance_runtime) != expected["runtime_fingerprint"]:
        raise ValueError(f"response point runtime mismatch: {manifest_path}")
    return payload


def _point_row_from_packet(
    payload: dict[str, Any],
    *,
    source_summary: Path,
    base_case: TrappedCase,
) -> dict[str, Any]:
    required = (
        "case_id",
        "n_particles",
        "rod_length",
        "lambda0",
        "relative_lambda_offset",
        "lambda_value",
        "guide",
        "guide_fingerprint",
        "point_run_controls",
        "point_controls_fingerprint",
        "shared_run_controls",
        "run_controls_fingerprint",
        "implementation",
        "implementation_fingerprint",
        "runtime",
        "runtime_fingerprint",
        "mixed_energy",
        "mixed_energy_conservative_stderr",
        "seed_energies",
        "point_identity_fingerprint",
        "resolved_guide_family",
    )
    if payload.get("schema_version") != DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION:
        raise ValueError(f"{source_summary} is not a current energy-response point")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{source_summary} is missing: {', '.join(missing)}")
    if str(payload["case_id"]) != base_case.case_id:
        raise ValueError(f"{source_summary} does not match base case {base_case.case_id}")
    if int(payload["n_particles"]) != base_case.n_particles or not np.isclose(
        float(payload["rod_length"]), base_case.rod_length, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"{source_summary} particle count or rod length does not match")
    offset = float(payload["relative_lambda_offset"])
    expected_lambda = lambda_from_relative_offset(offset)
    if not np.isclose(float(payload["lambda0"]), BASE_TRAP_QUADRATIC_COUPLING):
        raise ValueError(f"{source_summary} has the wrong base coupling")
    if not np.isclose(float(payload["lambda_value"]), expected_lambda, rtol=0.0, atol=1e-12):
        raise ValueError(f"{source_summary} lambda is inconsistent with its offset")
    for field, value in (
        ("guide_fingerprint", config_fingerprint(payload["guide"])),
        ("point_controls_fingerprint", config_fingerprint(payload["point_run_controls"])),
        ("run_controls_fingerprint", config_fingerprint(payload["shared_run_controls"])),
        ("implementation_fingerprint", _implementation_fingerprint(payload["implementation"])),
        ("runtime_fingerprint", config_fingerprint(payload["runtime"])),
    ):
        if payload[field] != value:
            raise ValueError(f"{source_summary} {field} is inconsistent with its payload")
    expected = {
        "schema_version": DMC_ENERGY_RESPONSE_POINT_SCHEMA_VERSION,
        "case_id": base_case.case_id,
        "relative_lambda_offset": offset,
        "lambda_value": expected_lambda,
        "guide_fingerprint": payload["guide_fingerprint"],
        "point_controls_fingerprint": payload["point_controls_fingerprint"],
        "run_controls_fingerprint": payload["run_controls_fingerprint"],
        "implementation_fingerprint": payload["implementation_fingerprint"],
        "runtime_fingerprint": payload["runtime_fingerprint"],
    }
    _load_verified_completed_point(source_summary, expected=expected)
    seed_energies = _seed_energies_from_packet(payload, source_summary=source_summary)
    _validate_packet_semantics(
        payload,
        seed_energies=seed_energies,
        source_summary=source_summary,
        base_case=base_case,
    )
    out = dict(payload)
    out["seed_ids"] = tuple(int(row["seed"]) for row in seed_energies)
    out["seed_energies"] = seed_energies
    out["seed_count"] = len(seed_energies)
    out["source_summary"] = str(source_summary)
    out["energy_point_status"] = energy_point_status_from_row(payload)
    return out


def _validate_stationarity_binding(
    *,
    case: TrappedCase,
    stationarity: dict[str, Any],
    run_controls: DMCRunControls,
    guide: dict[str, Any],
) -> None:
    required = (
        "case_id",
        "seeds",
        "mixed_energy",
        "mixed_energy_conservative_stderr",
        "population_weight_status",
        "valid_finite_clean",
        "stationarity_energy",
        "seed_summaries",
        "guide_family",
        "resolved_guide_family",
    )
    missing = [name for name in required if name not in stationarity]
    if missing:
        raise ValueError(f"stationarity summary is missing: {', '.join(missing)}")
    if stationarity["case_id"] != case.case_id:
        raise ValueError("stationarity summary belongs to a different case")
    if stationarity["guide_family"] != guide["guide_family"]:
        raise ValueError("stationarity summary used a different guide family")
    expected_resolved_family = f"quadratic-response({guide['guide_family']})"
    if stationarity["resolved_guide_family"] != expected_resolved_family:
        raise ValueError("stationarity summary did not resolve the quadratic-response guide")
    rows = stationarity["seed_summaries"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("stationarity summary has no seed rows")
    if [int(row["seed"]) for row in rows] != [int(seed) for seed in stationarity["seeds"]]:
        raise ValueError("stationarity seed summaries do not match the paired seed order")
    for row in rows:
        if row.get("local_step_method") != run_controls.local_step_method:
            raise ValueError("stationarity seed used a different local step method")
        if row.get("drift_limiter") != run_controls.drift_limiter:
            raise ValueError("stationarity seed used a different drift limiter")
        _require_optional_float_match(
            row.get("relative_alpha"),
            run_controls.relative_alpha,
            field="relative_alpha",
        )
        _require_optional_float_match(
            row.get("contact_beta"),
            run_controls.contact_beta,
            field="contact_beta",
        )
        _require_optional_float_match(
            row.get("response_lambda"),
            run_controls.response_lambda,
            field="response_lambda",
        )
        if row.get("guide_family") != guide["guide_family"]:
            raise ValueError("stationarity seed used a different guide family")
        if row.get("resolved_guide_family") != expected_resolved_family:
            raise ValueError("stationarity seed did not resolve the quadratic-response guide")


def _validate_packet_semantics(
    payload: dict[str, Any],
    *,
    seed_energies: list[dict[str, float | int]],
    source_summary: Path,
    base_case: TrappedCase,
) -> None:
    point_controls = payload["point_run_controls"]
    shared_controls = payload["shared_run_controls"]
    guide = payload["guide"]
    if not isinstance(point_controls, dict) or not isinstance(shared_controls, dict):
        raise ValueError(f"{source_summary} run controls are malformed")
    base_controls = shared_controls.get("base_run_controls")
    if not isinstance(base_controls, dict):
        raise ValueError(f"{source_summary} has no shared base controls")
    if shared_controls.get("case_id") != base_case.case_id:
        raise ValueError(f"{source_summary} shared controls belong to another case")
    response_lambda = point_controls.get("response_lambda")
    if response_lambda is None or not np.isclose(
        float(response_lambda),
        float(payload["lambda_value"]),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError(f"{source_summary} point controls do not bind the sampled lambda")
    point_base = dict(point_controls)
    point_base.pop("response_lambda", None)
    if point_base != base_controls:
        raise ValueError(f"{source_summary} point controls differ from shared base controls")
    if not isinstance(guide, dict) or guide.get("case_id") != base_case.case_id:
        raise ValueError(f"{source_summary} guide belongs to another case")
    family = guide.get("guide_family")
    if family not in {"reduced-tg", "contact-corrected-reduced-tg"}:
        raise ValueError(f"{source_summary} has an unsupported guide family")
    parameter_identity = guide.get("parameter_source")
    if parameter_identity != shared_controls.get("guide_parameter_identity"):
        raise ValueError(f"{source_summary} guide parameter identity is inconsistent")
    _validate_guide_parameter_identity(
        parameter_identity,
        guide_family=str(family),
        source_summary=source_summary,
    )
    for field in ("relative_alpha", "contact_beta"):
        _require_optional_float_match(
            guide.get(field),
            _optional_float(base_controls.get(field)),
            field=field,
        )
    expected_resolved = f"quadratic-response({family})"
    if payload.get("resolved_guide_family") != expected_resolved:
        raise ValueError(f"{source_summary} does not bind the quadratic-response wrapper")
    paired_seeds = shared_controls.get("paired_seeds")
    seed_ids = [int(row["seed"]) for row in seed_energies]
    if not isinstance(paired_seeds, list) or seed_ids != [int(seed) for seed in paired_seeds]:
        raise ValueError(f"{source_summary} seed set differs from shared paired seeds")
    if int(payload.get("seed_count", -1)) != len(seed_energies):
        raise ValueError(f"{source_summary} seed_count does not match seed energies")
    mean_energy = float(np.mean([float(row["energy"]) for row in seed_energies]))
    if not np.isclose(mean_energy, float(payload["mixed_energy"]), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{source_summary} mixed energy does not match paired seed energies")


def _seed_energies_from_packet(
    payload: dict[str, Any],
    *,
    source_summary: Path,
) -> list[dict[str, float | int]]:
    values = payload.get("seed_energies")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{source_summary} is missing seed-level energies")
    out: list[dict[str, float | int]] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, dict) or "seed" not in value or "energy" not in value:
            raise ValueError(f"{source_summary} contains a malformed seed energy")
        seed = int(value["seed"])
        energy = float(value["energy"])
        if seed in seen or not math.isfinite(energy):
            raise ValueError(f"{source_summary} contains duplicate or non-finite seed energy")
        seen.add(seed)
        out.append({"seed": seed, "energy": energy})
    return sorted(out, key=lambda row: int(row["seed"]))


def _validate_unique_response_points(rows: list[dict[str, Any]]) -> None:
    offsets = np.asarray([row["relative_lambda_offset"] for row in rows], dtype=float)
    if offsets.size != 5:
        raise ValueError("response reanalysis requires exactly five points")
    validate_five_point_offsets(tuple(float(value) for value in offsets))


def _validate_shared_identity(rows: list[dict[str, Any]]) -> None:
    for field in (
        "guide_fingerprint",
        "run_controls_fingerprint",
        "implementation_fingerprint",
        "runtime_fingerprint",
    ):
        if len({str(row[field]) for row in rows}) != 1:
            raise ValueError(f"all response points must share {field}")


def _validate_shared_seed_set(rows: list[dict[str, Any]]) -> None:
    if len({tuple(int(seed) for seed in row["seed_ids"]) for row in rows}) != 1:
        raise ValueError("all response points must contain the same paired seed set")


def _validate_guide_parameter_identity(
    identity: Any,
    *,
    guide_family: str,
    source_summary: Path,
) -> None:
    required_keys = {
        "kind",
        "summary_sha256",
        "manifest_sha256",
        "identity_fingerprint",
    }
    if not isinstance(identity, dict) or set(identity) != required_keys:
        raise ValueError(f"{source_summary} guide parameter identity is malformed")
    digests = tuple(identity[key] for key in required_keys - {"kind"})
    if guide_family == "contact-corrected-reduced-tg":
        if identity["kind"] != "validated_contact_guide" or not all(
            _is_sha256(digest) for digest in digests
        ):
            raise ValueError(f"{source_summary} does not bind a validated contact-guide artifact")
        return
    if identity["kind"] != "explicit" or any(digest is not None for digest in digests):
        raise ValueError(f"{source_summary} has an invalid reduced-TG parameter source")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_implementation_identity(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("run provenance is not a mapping")
    implementation = provenance.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError("run provenance has no implementation identity")
    _implementation_fingerprint(implementation)
    return implementation


def _required_runtime_identity(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("run provenance is not a mapping")
    runtime = {
        "python_version": provenance.get("python_version"),
        "platform": provenance.get("platform"),
        "dependencies": provenance.get("dependencies"),
    }
    if not isinstance(runtime["python_version"], str) or not runtime["python_version"]:
        raise ValueError("run provenance has no Python runtime version")
    if not isinstance(runtime["platform"], str) or not runtime["platform"]:
        raise ValueError("run provenance has no platform identity")
    dependencies = runtime["dependencies"]
    if not isinstance(dependencies, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in dependencies.items()
    ):
        raise ValueError("run provenance has no dependency identity")
    return runtime


def _energy_response_plan_fingerprint(plan: dict[str, Any]) -> str:
    fingerprint_payload = dict(plan)
    fingerprint_payload.pop("plan_fingerprint", None)
    # The content digest is scientific identity; the local source path is not.
    fingerprint_payload.pop("guide_parameter_source", None)
    return config_fingerprint(fingerprint_payload)


def _implementation_fingerprint(implementation: dict[str, Any]) -> str:
    value = implementation.get("source_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("implementation identity has no source-tree SHA-256")
    return value


def _require_optional_float_match(
    observed: Any,
    expected: float | None,
    *,
    field: str,
) -> None:
    if expected is None:
        if observed is not None:
            raise ValueError(f"stationarity seed unexpectedly set {field}")
        return
    if observed is None or not np.isclose(float(observed), expected, rtol=0.0, atol=1.0e-14):
        raise ValueError(f"stationarity seed used a different {field}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
