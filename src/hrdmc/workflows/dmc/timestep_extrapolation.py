from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.analysis.timestep_extrapolation import (
    TimeStepPoint,
    analyze_time_step_extrapolation,
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

TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION = "dmc_timestep_extrapolation_v2"
SUPPORTED_INPUTS = {
    ("dmc_benchmark_packet", "dmc_benchmark_packet_v3"),
    ("dmc_trapped_stationarity_grid", "dmc_trapped_stationarity_grid_v2"),
}
PUBLICATION_READY_WORKFLOW_STATUSES = {"accepted", "accepted_with_warnings"}
ENERGY_CHAIN_ACCEPTED_STATUSES = {"accepted", "spread_warning"}


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
    run_status: str
    energy_status: str
    energy_quality: dict[str, Any]
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
            "run_status": self.run_status,
            "energy_status": self.energy_status,
            "energy_quality": self.energy_quality,
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


def run_timestep_extrapolation_workflow(
    summary_paths: Sequence[Path],
    *,
    output_dir: Path | None,
    command: list[str] | None = None,
    write_artifacts: bool = True,
    sensitivity_sigma: float = 2.0,
    fit_alpha: float = 0.05,
) -> dict[str, Any]:
    """Verify DMC summaries and extrapolate their mixed energy to zero time step."""

    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when artifacts are written")
    resolved_paths = [Path(path).resolve() for path in summary_paths]
    if len(resolved_paths) < 3:
        raise ValueError("time-step extrapolation requires at least three summaries")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("time-step summary paths must be unique")
    if write_artifacts:
        assert output_dir is not None
        _validate_output_separation(output_dir.resolve(), resolved_paths)

    loaded = [_load_time_step_point(path) for path in resolved_paths]
    reference_identity = loaded[0].identity
    mismatches = [
        (point.summary_path, _different_identity_fields(reference_identity, point.identity))
        for point in loaded[1:]
        if point.identity != reference_identity
    ]
    if mismatches:
        details = "; ".join(f"{path}: {', '.join(fields)}" for path, fields in mismatches)
        raise ValueError(
            "time-step summaries do not share the required case, guide, source, "
            f"walker, initialization, and drift identity: {details}"
        )

    loaded.sort(key=lambda item: item.point.dt)
    analysis = analyze_time_step_extrapolation(
        [item.point for item in loaded],
        sensitivity_sigma=sensitivity_sigma,
        fit_alpha=fit_alpha,
    )
    identity_fingerprint = config_fingerprint(reference_identity)
    inputs = [item.to_dict() for item in loaded]
    control_variation = _control_variation(loaded)
    sampling_design = _sampling_design(loaded)
    input_quality = _input_quality(loaded)
    cross_timestep_covariance = _cross_timestep_covariance(loaded)
    unresolved_reasons: list[str] = []
    if analysis.classification == "fit_inadequate":
        unresolved_reasons.append("fit_inadequate")
    elif analysis.classification == "model_sensitive":
        unresolved_reasons.append("model_sensitive")
    if analysis.fit_window_status != "accepted":
        unresolved_reasons.append("fit_window_unresolved")
    if not input_quality["publication_accepted"]:
        unresolved_reasons.append("input_quality_unresolved")
    if cross_timestep_covariance["status"] != "accepted":
        unresolved_reasons.append("cross_timestep_covariance_unresolved")

    if analysis.classification == "fit_inadequate":
        workflow_status = analysis.classification
    elif analysis.classification == "model_sensitive":
        workflow_status = analysis.classification
    elif analysis.fit_window_status != "accepted":
        workflow_status = "fit_window_unresolved"
    elif not input_quality["publication_accepted"]:
        workflow_status = "input_quality_unresolved"
    elif cross_timestep_covariance["status"] != "accepted":
        workflow_status = "cross_timestep_covariance_unresolved"
    elif input_quality["status"] == "accepted_with_warnings":
        workflow_status = "accepted_with_warnings"
    else:
        workflow_status = "accepted"
    payload: dict[str, Any] = {
        "schema_version": TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
        "status": workflow_status,
        "classification": analysis.classification,
        "unresolved_reasons": unresolved_reasons,
        "diagnostic": "DMC mixed-energy time-step extrapolation",
        "case_id": loaded[0].case_id,
        "identity": reference_identity,
        "identity_fingerprint": identity_fingerprint,
        "point_count": len(loaded),
        "sensitivity_sigma": sensitivity_sigma,
        "fit_alpha": fit_alpha,
        "input_summaries": inputs,
        "input_control_variation": control_variation,
        "sampling_design": sampling_design,
        "input_quality": input_quality,
        "cross_timestep_covariance": cross_timestep_covariance,
        "extrapolation": analysis.to_dict(),
        "reference_energy_used_for_model_selection": False,
        "scientific_scope": (
            "The fit quantifies finite-time-step dependence at fixed case, guide, "
            "source implementation, walker population, initialization, propagator, "
            "and population method. Sampling-effort variation remains visible but "
            "does not change the mixed-energy estimand once each input is accepted."
        ),
    }
    if workflow_status in PUBLICATION_READY_WORKFLOW_STATUSES:
        analysis_payload = analysis.to_dict()
        payload.update(
            {
                "extrapolated_energy": analysis_payload["candidate_zero_step_energy"],
                "extrapolated_energy_statistical_stderr": analysis_payload[
                    "candidate_zero_step_energy_statistical_stderr"
                ],
                "extrapolated_energy_model_spread_systematic_allowance": (
                    analysis_payload["model_spread_systematic_allowance"]
                ),
            }
        )
    config = {
        "case_id": loaded[0].case_id,
        "identity": reference_identity,
        "identity_fingerprint": identity_fingerprint,
        "sensitivity_sigma": sensitivity_sigma,
        "fit_alpha": fit_alpha,
        "sampling_design": sampling_design,
        "input_quality": input_quality,
        "cross_timestep_covariance": cross_timestep_covariance,
        "inputs": [
            {
                "summary_path": str(item.summary_path),
                "summary_sha256": item.summary_sha256,
                "manifest_path": str(item.manifest_path),
                "manifest_sha256": item.manifest_sha256,
                "dt": item.point.dt,
            }
            for item in loaded
        ],
    }
    artifacts: dict[str, str | None] = {
        "summary": None,
        "point_table": None,
        "run_manifest": None,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
    }
    if write_artifacts:
        assert output_dir is not None
        root = ensure_dir(output_dir.resolve())
        summary_path = root / "summary.json"
        table_path = _write_point_table(root, loaded)
        payload["artifacts"] = {
            "summary": str(summary_path),
            "point_table": str(table_path),
            "run_manifest": str(root / "run_manifest.json"),
            "output_dir": str(root),
        }
        write_json(summary_path, payload)
        manifest_path = write_run_manifest(
            root,
            run_name="dmc_timestep_extrapolation",
            config=config,
            artifacts=[summary_path, table_path],
            schema_version=TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
            provenance=build_run_provenance(command),
            status=workflow_status,
        )
        artifacts = {
            "summary": str(summary_path),
            "point_table": str(table_path),
            "run_manifest": str(manifest_path),
            "output_dir": str(root),
        }
    else:
        payload["artifacts"] = artifacts
    payload["workflow_artifacts"] = artifacts
    return payload


def _load_time_step_point(summary_path: Path) -> LoadedTimeStepPoint:
    if not summary_path.is_file():
        raise FileNotFoundError(f"time-step summary does not exist: {summary_path}")
    manifest_path = summary_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
    manifest = _load_mapping(manifest_path, "run manifest")
    summary = _load_mapping(summary_path, "summary")
    warnings = _verify_summary_binding(
        summary_path=summary_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    run_name = _required_string(manifest, "run_name")
    result_schema = _required_string(manifest, "result_schema_version")
    if (run_name, result_schema) not in SUPPORTED_INPUTS:
        raise ValueError(f"unsupported time-step input {run_name}/{result_schema}: {summary_path}")
    if summary.get("schema_version") != result_schema:
        raise ValueError(f"summary schema does not match its manifest: {summary_path}")
    if summary.get("status") != manifest.get("status"):
        raise ValueError(f"summary status does not match its manifest: {summary_path}")

    config = _required_mapping(manifest, "config")
    manifest_controls = _required_mapping(config, "controls")
    summary_controls = summary.get("controls")
    if summary_controls is None:
        controls = manifest_controls
    elif not isinstance(summary_controls, dict):
        raise ValueError(f"summary controls must be a mapping: {summary_path}")
    else:
        controls = summary_controls
        if controls != manifest_controls:
            raise ValueError(f"summary controls do not match the manifest: {summary_path}")
    _validate_step_tau_controls(controls, summary_path=summary_path)
    case_id = _case_id(summary, config, run_name=run_name)
    _verify_summary_method_identity(summary, config, summary_path=summary_path)

    stationarity, energy, stderr, energy_status = _energy_fields(
        summary,
        run_name=run_name,
        summary_path=summary_path,
    )
    _verify_conservative_stderr(stationarity, stderr, summary_path=summary_path)
    if stationarity.get("base_numerics_valid") is not True:
        raise ValueError(f"time-step point failed base numerical checks: {summary_path}")
    if stationarity.get("population_weights_controlled") is not True:
        raise ValueError(f"time-step point has uncontrolled population weights: {summary_path}")

    identity = _scientific_identity(
        summary,
        manifest,
        controls=controls,
        stationarity=stationarity,
        case_id=case_id,
    )
    seeds = _verified_seeds(
        summary,
        config,
        stationarity,
        run_name=run_name,
        summary_path=summary_path,
    )
    dt = _required_positive_float(controls.get("dt"), "dt")
    return LoadedTimeStepPoint(
        point=TimeStepPoint(
            dt=dt,
            energy=energy,
            conservative_stderr=stderr,
            label=str(summary_path),
        ),
        case_id=case_id,
        identity=identity,
        summary_path=summary_path,
        summary_sha256=file_sha256(summary_path),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        run_name=run_name,
        result_schema_version=result_schema,
        run_status=_required_string(manifest, "status"),
        energy_status=energy_status,
        energy_quality=_energy_input_quality(stationarity, reported_status=energy_status),
        seeds=seeds,
        manifest_verification_warnings=tuple(warnings),
        controls=controls,
        telemetry=_point_telemetry(stationarity),
    )


def _verify_summary_binding(
    *,
    summary_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[str]:
    """Verify the selected summary without rejecting changed unrelated plots."""

    try:
        relative_summary = summary_path.relative_to(manifest_path.parent).as_posix()
    except ValueError as exc:
        raise ValueError("summary must be inside its run-manifest directory") from exc
    entries_value = manifest.get("artifacts")
    if not isinstance(entries_value, list):
        raise ValueError(f"run manifest has no artifact records: {manifest_path}")
    entries = [entry for entry in entries_value if isinstance(entry, dict)]
    matches = [entry for entry in entries if entry.get("path") == relative_summary]
    if len(matches) != 1:
        raise ValueError(f"summary is not uniquely recorded by its run manifest: {summary_path}")
    entry = matches[0]
    if entry.get("size_bytes") != summary_path.stat().st_size:
        raise ValueError(f"summary size does not match its manifest: {summary_path}")
    if entry.get("sha256") != file_sha256(summary_path):
        raise ValueError(f"summary digest does not match its manifest: {summary_path}")

    _, errors = verify_run_manifest(manifest_path)
    unrelated: list[str] = []
    for error in errors:
        artifact_path = _artifact_error_path(error)
        if (
            artifact_path is not None
            and artifact_path != relative_summary
            and Path(artifact_path).parts[:1] == ("plots",)
        ):
            unrelated.append(error)
            continue
        raise ValueError(f"run manifest verification failed: {manifest_path}: {error}")
    return unrelated


def _artifact_error_path(error: str) -> str | None:
    for prefix in ("missing artifact: ", "size mismatch: ", "sha256 mismatch: "):
        if error.startswith(prefix):
            return error[len(prefix) :]
    return None


def _case_id(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    run_name: str,
) -> str:
    if run_name == "dmc_benchmark_packet":
        case_id = _required_string(summary, "case_id")
        if config.get("case") != case_id:
            raise ValueError("benchmark summary case does not match its manifest")
        return case_id
    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("stationarity time-step summary must contain exactly one case")
    case_id = _required_string(cases[0], "case_id")
    if config.get("cases") != [case_id]:
        raise ValueError("stationarity summary case does not match its manifest")
    return case_id


def _verify_summary_method_identity(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    fields = (
        "guide_family",
        "initialization_mode",
        "init_width_log_sigma",
        "breathing_preburn_steps",
        "breathing_preburn_log_step",
    )
    for field in fields:
        if summary.get(field) != config.get(field):
            raise ValueError(f"summary {field} does not match its manifest: {summary_path}")
    summary_guide = _required_mapping(summary, "guide_parameters")
    manifest_guide = _manifest_guide_parameters(config)
    if summary_guide != manifest_guide:
        raise ValueError(f"guide identity does not match the manifest: {summary_path}")


def _manifest_guide_parameters(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("guide_parameters")
    if isinstance(nested, dict):
        return nested
    controls = _required_mapping(config, "controls")
    return {
        "relative_alpha": controls.get("relative_alpha"),
        "contact_beta": controls.get("contact_beta"),
        "source": config.get("guide_parameter_source"),
        "source_sha256": config.get("guide_parameter_source_sha256"),
        "source_manifest_sha256": config.get("guide_parameter_source_manifest_sha256"),
        "source_identity_fingerprint": config.get("guide_parameter_source_identity_fingerprint"),
    }


def _energy_fields(
    summary: dict[str, Any],
    *,
    run_name: str,
    summary_path: Path,
) -> tuple[dict[str, Any], float, float, str]:
    if run_name == "dmc_benchmark_packet":
        stationarity = _required_mapping(summary, "stationarity")
        estimates = _required_mapping(summary, "estimates")
        energy_estimate = _required_mapping(estimates, "energy")
        energy = _required_float(energy_estimate.get("value"), "energy value")
        stderr = _required_positive_float(
            energy_estimate.get("stderr"),
            "energy conservative stderr",
        )
        stationarity_energy = _required_float(
            stationarity.get("mixed_energy"),
            "stationarity mixed energy",
        )
        stationarity_stderr = _required_positive_float(
            stationarity.get("mixed_energy_conservative_stderr"),
            "stationarity conservative energy stderr",
        )
        if energy != stationarity_energy or stderr != stationarity_stderr:
            raise ValueError(
                f"benchmark energy estimate is not the stationarity conservative value: "
                f"{summary_path}"
            )
        return (
            stationarity,
            energy,
            stderr,
            str(energy_estimate.get("status", "unknown")),
        )
    cases = summary["cases"]
    stationarity = cases[0]
    assert isinstance(stationarity, dict)
    return (
        stationarity,
        _required_float(stationarity.get("mixed_energy"), "mixed energy"),
        _required_positive_float(
            stationarity.get("mixed_energy_conservative_stderr"),
            "conservative energy stderr",
        ),
        str(stationarity.get("final_classification", "unknown")),
    )


def _verified_seeds(
    summary: dict[str, Any],
    config: dict[str, Any],
    stationarity: dict[str, Any],
    *,
    run_name: str,
    summary_path: Path,
) -> tuple[int, ...]:
    config_seeds = _seed_list(config.get("seeds"), "manifest seeds")
    if len(set(config_seeds)) != len(config_seeds):
        raise ValueError(f"manifest seeds must be unique: {summary_path}")
    reported: list[tuple[str, tuple[int, ...]]] = [
        ("stationarity seeds", _seed_list(stationarity.get("seeds"), "stationarity seeds")),
        (
            "stationarity seed summaries",
            _seed_ids_from_rows(
                stationarity.get("seed_summaries"),
                "stationarity seed_summaries",
            ),
        ),
    ]
    if run_name == "dmc_benchmark_packet":
        reported.extend(
            [
                ("benchmark seeds", _seed_list(summary.get("seeds"), "benchmark seeds")),
                (
                    "benchmark seed results",
                    _seed_ids_from_rows(summary.get("seed_results"), "benchmark seed_results"),
                ),
            ]
        )
    for description, seeds in reported:
        if seeds != config_seeds:
            raise ValueError(f"{description} do not match manifest seeds: {summary_path}")
    for owner in (summary, stationarity):
        seed_count = owner.get("seed_count")
        if seed_count is not None and seed_count != len(config_seeds):
            raise ValueError(f"seed_count does not match manifest seeds: {summary_path}")
    return config_seeds


def _seed_list(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list")
    return tuple(_required_int(seed, description) for seed in value)


def _seed_ids_from_rows(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{description} must contain seed mappings")
    return tuple(_required_int(row.get("seed"), description) for row in value)


def _energy_semantics(
    summary: dict[str, Any],
    stationarity: dict[str, Any],
) -> dict[str, str]:
    estimator_labels: list[str] = []
    for value in (
        stationarity.get("energy_estimator"),
        _nested_energy_estimator(summary),
    ):
        if isinstance(value, str) and value:
            estimator_labels.append(value)
    if not estimator_labels or any(
        "mixed" not in label.lower()
        or "local" not in label.lower()
        or "energy" not in label.lower()
        for label in estimator_labels
    ):
        raise ValueError("time-step extrapolation requires a mixed local-energy estimator")
    return {
        "estimator": "mixed_local_energy",
        "energy_unit": _consistent_semantic_string(
            summary,
            stationarity,
            "energy_unit",
        ),
        "report_energy_unit": _consistent_semantic_string(
            summary,
            stationarity,
            "report_energy_unit",
        ),
        "energy_coordinate": _consistent_semantic_string(
            summary,
            stationarity,
            "energy_coordinate",
        ),
    }


def _nested_energy_estimator(summary: dict[str, Any]) -> Any:
    estimates = summary.get("estimates")
    if isinstance(estimates, dict):
        energy = estimates.get("energy")
        if isinstance(energy, dict):
            return energy.get("estimator")
    method = summary.get("method")
    return method.get("energy") if isinstance(method, dict) else None


def _consistent_semantic_string(
    summary: dict[str, Any],
    stationarity: dict[str, Any],
    field: str,
) -> str:
    values = [value for value in (summary.get(field), stationarity.get(field)) if value is not None]
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{field} must be recorded for time-step extrapolation")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"summary and case {field} values differ")
    return str(values[0])


def _verify_conservative_stderr(
    stationarity: dict[str, Any],
    stderr: float,
    *,
    summary_path: Path,
) -> None:
    components = (
        stationarity.get("mixed_energy_seed_stderr"),
        stationarity.get("mixed_energy_blocking_stderr"),
        stationarity.get("mixed_energy_correlated_stderr"),
    )
    finite_components = [
        float(value)
        for value in components
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if finite_components and stderr + 1.0e-15 < max(finite_components):
        raise ValueError(
            f"declared conservative energy stderr is smaller than a component: {summary_path}"
        )


def _scientific_identity(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    *,
    controls: dict[str, Any],
    stationarity: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    provenance = _required_mapping(manifest, "provenance")
    implementation = _required_mapping(provenance, "implementation")
    source_tree_sha256 = _required_string(implementation, "source_tree_sha256")
    cases = summary.get("cases")
    case_summary = (
        cases[0]
        if isinstance(cases, list) and len(cases) == 1 and isinstance(cases[0], dict)
        else summary
    )
    return {
        "case_id": case_id,
        "case_parameterization": case_summary.get("case_parameterization"),
        "guide_family": summary.get("guide_family"),
        "guide_parameters": _required_mapping(summary, "guide_parameters"),
        "implementation_source_tree_sha256": source_tree_sha256,
        "walkers": _required_int(controls.get("walkers"), "walkers"),
        "local_step_method": controls.get("local_step_method"),
        "drift_limiter": controls.get("drift_limiter"),
        "response_lambda": controls.get("response_lambda"),
        "population_control": {
            "ess_resample_fraction": controls.get("ess_resample_fraction", 0.35),
        },
        "energy_semantics": _energy_semantics(summary, stationarity),
        "initialization": {
            "mode": summary.get("initialization_mode"),
            "init_width_log_sigma": summary.get("init_width_log_sigma"),
            "breathing_preburn_steps": summary.get("breathing_preburn_steps"),
            "breathing_preburn_log_step": summary.get("breathing_preburn_log_step"),
        },
        "collective_rn": summary.get(
            "collective_rn",
            summary.get("collective_rn_controls"),
        ),
    }


def _control_variation(points: Sequence[LoadedTimeStepPoint]) -> dict[str, list[Any]]:
    fields = (
        "dt",
        "burn_tau",
        "production_tau",
        "store_every",
        "grid_extent",
        "n_bins",
    )
    return {
        field: _distinct_values([point.controls.get(field) for point in points]) for field in fields
    } | {
        "seed_sets": [list(point.seeds) for point in points],
        "run_statuses": _distinct_values([point.run_status for point in points]),
        "energy_statuses": _distinct_values([point.energy_status for point in points]),
    }


def _sampling_design(points: Sequence[LoadedTimeStepPoint]) -> dict[str, Any]:
    comparisons: dict[str, list[Any]] = {
        "burn_tau": _distinct_values([point.controls.get("burn_tau") for point in points]),
        "production_tau": _distinct_values(
            [point.controls.get("production_tau") for point in points]
        ),
        "grid_extent": _distinct_values([point.controls.get("grid_extent") for point in points]),
        "n_bins": _distinct_values([point.controls.get("n_bins") for point in points]),
        "trace_spacing_tau": _distinct_values(
            [
                round(
                    point.point.dt
                    * _required_int(point.controls.get("store_every"), "store_every"),
                    15,
                )
                for point in points
            ]
        ),
    }
    varied_fields = [field for field, values in comparisons.items() if len(values) > 1]
    return {
        "status": "uniform" if not varied_fields else "varied",
        "varied_fields": varied_fields,
        "comparisons": comparisons,
        "interpretation": (
            "These sampling-effort and diagnostic-grid differences are advisory for "
            "mixed-energy WLS once every point passes its numerical checks."
        ),
    }


def _cross_timestep_covariance(points: Sequence[LoadedTimeStepPoint]) -> dict[str, Any]:
    pairwise_overlap: list[dict[str, Any]] = []
    for first_index, first in enumerate(points):
        for second in points[first_index + 1 :]:
            overlap = sorted(set(first.seeds) & set(second.seeds))
            pairwise_overlap.append(
                {
                    "first_dt": first.point.dt,
                    "second_dt": second.point.dt,
                    "overlap_count": len(overlap),
                    "overlap_seeds": overlap,
                }
            )
    overlapping = [pair for pair in pairwise_overlap if pair["overlap_count"]]
    return {
        "status": "accepted" if not overlapping else "unresolved",
        "method": "diagonal weighted least squares",
        "pairwise_seed_overlap": pairwise_overlap,
        "overlapping_pair_count": len(overlapping),
        "publication_requirement": (
            "Disjoint seed sets are required because cross-time-step covariance is "
            "not estimated by this summary-level analysis."
        ),
    }


def _input_quality(points: Sequence[LoadedTimeStepPoint]) -> dict[str, Any]:
    rows = [
        {
            "summary_path": str(point.summary_path),
            "run_status": point.run_status,
            "energy_status": point.energy_status,
            **point.energy_quality,
        }
        for point in points
    ]
    unresolved = [row for row in rows if not row["publication_accepted"]]
    warning_rows = [
        row
        for row in rows
        if row["publication_accepted"] and row["publication_status"] != "accepted"
    ]
    return {
        "status": (
            "unresolved"
            if unresolved
            else "accepted_with_warnings"
            if warning_rows
            else "accepted"
        ),
        "publication_accepted": not unresolved,
        "publication_accepted_statuses": ["accepted", "accepted_with_precision_warning"],
        "points": rows,
        "unresolved_point_count": len(unresolved),
        "precision_warning_point_count": len(warning_rows),
        "interpretation": (
            "Energy-specific method failures remain unresolved. Precision warnings "
            "with a valid energy chain and conservative correlated-error estimate "
            "remain visible but do not veto the mixed-energy extrapolation."
        ),
    }


def _energy_input_quality(
    stationarity: dict[str, Any],
    *,
    reported_status: str,
) -> dict[str, Any]:
    validation_passed = stationarity.get("validation_passed") is True
    method_status = stationarity.get("method_status")
    chain_status = stationarity.get("stationarity_energy")
    publication_accepted = (
        validation_passed
        and method_status == "accepted"
        and chain_status in ENERGY_CHAIN_ACCEPTED_STATUSES
    )
    precision_warning = publication_accepted and (
        reported_status != "accepted" or chain_status == "spread_warning"
    )
    return {
        "validation_passed": validation_passed,
        "method_status": method_status,
        "energy_chain_status": chain_status,
        "precision_status": stationarity.get("precision_status"),
        "publication_accepted": publication_accepted,
        "publication_status": (
            "accepted_with_precision_warning"
            if precision_warning
            else "accepted"
            if publication_accepted
            else "unresolved"
        ),
    }


def _point_telemetry(stationarity: dict[str, Any]) -> dict[str, Any]:
    seed_rows_value = stationarity.get("seed_summaries")
    seed_rows = (
        [row for row in seed_rows_value if isinstance(row, dict)]
        if isinstance(seed_rows_value, list)
        else []
    )

    def values(field: str) -> list[float]:
        return [
            float(value)
            for row in seed_rows
            if isinstance((value := row.get(field)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]

    def mean(field: str) -> float | None:
        observed = values(field)
        return None if not observed else float(math.fsum(observed) / len(observed))

    def maximum(field: str) -> float | None:
        observed = values(field)
        return None if not observed else float(max(observed))

    return {
        "local_acceptance_fraction_mean": mean("local_acceptance_fraction_mean"),
        "invalid_proposal_fraction_max": maximum("invalid_proposal_fraction_max"),
        "metropolis_rejection_fraction_max": maximum("metropolis_rejection_fraction_max"),
        "configuration_esjd_mean": mean("configuration_esjd_mean"),
        "log_weight_span_max": _optional_finite_float(stationarity.get("log_weight_span_max")),
        "rhat_energy": _optional_finite_float(stationarity.get("rhat_energy")),
        "neff_energy": _optional_finite_float(stationarity.get("neff_energy")),
        "population_weight_status": stationarity.get("population_weight_status"),
    }


def _validate_step_tau_controls(
    controls: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    dt = _required_positive_float(controls.get("dt"), "dt")
    for steps_field, tau_field, allow_zero in (
        ("burn_in_steps", "burn_tau", True),
        ("production_steps", "production_tau", False),
    ):
        if steps_field not in controls or tau_field not in controls:
            continue
        steps = _required_int(controls.get(steps_field), steps_field)
        tau = _required_float(controls.get(tau_field), tau_field)
        if steps <= 0 or tau < 0.0 or (not allow_zero and tau == 0.0):
            raise ValueError(f"invalid {steps_field}/{tau_field} controls: {summary_path}")
        expected_steps = max(1, int(round(tau / dt)))
        product_consistent = tau == 0.0 or math.isclose(
            steps * dt,
            tau,
            rel_tol=1.0e-12,
            abs_tol=0.5 * dt + 1.0e-15,
        )
        if steps != expected_steps or not product_consistent:
            raise ValueError(f"{steps_field} * dt is inconsistent with {tau_field}: {summary_path}")


def _distinct_values(values: Sequence[Any]) -> list[Any]:
    distinct: list[Any] = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    return distinct


def _different_identity_fields(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    return sorted(
        key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key)
    )


def _validate_output_separation(output_dir: Path, summary_paths: Sequence[Path]) -> None:
    for summary_path in summary_paths:
        run_dir = summary_path.parent
        if (
            output_dir == run_dir
            or output_dir.is_relative_to(run_dir)
            or run_dir.is_relative_to(output_dir)
        ):
            raise ValueError(
                f"output_dir must not overlap an input summary or run directory: {output_dir}"
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
        "run_status",
        "energy_status",
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
            payload.update(point.telemetry)
            writer.writerow({field: payload.get(field, "") for field in fields})
    return path


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(value: Any, description: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _required_positive_float(value: Any, description: str) -> float:
    number = _required_float(value, description)
    if number <= 0.0:
        raise ValueError(f"{description} must be positive")
    return number


def _optional_finite_float(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _required_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    return value
