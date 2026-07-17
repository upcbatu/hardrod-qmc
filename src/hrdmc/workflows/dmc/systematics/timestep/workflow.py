from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hrdmc.analysis.timestep_extrapolation import analyze_time_step_extrapolation
from hrdmc.artifacts import config_fingerprint
from hrdmc.workflows.dmc.systematics.timestep.assessment import (
    _apply_energy_quality_assessment,
    _control_variation,
    _cross_timestep_covariance,
    _input_quality,
    _practical_resolution_assessment,
    _sampling_design,
)
from hrdmc.workflows.dmc.systematics.timestep.contract import (
    PUBLICATION_READY_WORKFLOW_STATUSES,
    TIMESTEP_EXTRAPOLATION_RUN_NAME,
    TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
    EnergyReportingResolutionPolicy,
)
from hrdmc.workflows.dmc.systematics.timestep.outputs import (
    _validate_output_separation,
    persist_timestep_outputs,
)
from hrdmc.workflows.dmc.systematics.timestep.payload import attach_workflow_artifacts
from hrdmc.workflows.dmc.systematics.timestep.sources import (
    _different_identity_fields,
    _load_time_step_point,
    _required_mapping,
)


def run_timestep_extrapolation_workflow(
    summary_paths: Sequence[Path],
    *,
    output_dir: Path | None,
    command: list[str] | None = None,
    write_artifacts: bool = True,
    sensitivity_sigma: float = 2.0,
    fit_alpha: float = 0.05,
    energy_assessment_manifest: Path | None = None,
    energy_reporting_policy: EnergyReportingResolutionPolicy | None = None,
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
        resolved_output = output_dir.resolve()
        if resolved_output.exists() and (
            not resolved_output.is_dir() or any(resolved_output.iterdir())
        ):
            raise FileExistsError(
                f"time-step extrapolation output directory is not empty: {resolved_output}"
            )
        protected_artifacts = list(resolved_paths)
        if energy_assessment_manifest is not None:
            protected_artifacts.append(energy_assessment_manifest.resolve())
        _validate_output_separation(resolved_output, protected_artifacts)

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

    energy_quality_assessment: dict[str, Any] | None = None
    if energy_assessment_manifest is not None:
        loaded, energy_quality_assessment = _apply_energy_quality_assessment(
            loaded,
            energy_assessment_manifest=energy_assessment_manifest,
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
    practical_resolution = _practical_resolution_assessment(
        analysis,
        policy=energy_reporting_policy,
        input_quality=input_quality,
        cross_timestep_covariance=cross_timestep_covariance,
        energy_semantics=_required_mapping(reference_identity, "energy_semantics"),
    )
    model_bound_accepted = practical_resolution["accepted_with_model_bound"] is True
    unresolved_reasons: list[str] = []
    if analysis.classification == "fit_inadequate":
        unresolved_reasons.append("fit_inadequate")
    elif analysis.classification == "model_sensitive" and not model_bound_accepted:
        unresolved_reasons.append("model_sensitive")
    if analysis.fit_window_status != "accepted":
        unresolved_reasons.append("fit_window_unresolved")
    if not input_quality["publication_accepted"]:
        unresolved_reasons.append("input_quality_unresolved")
    if cross_timestep_covariance["status"] != "accepted":
        unresolved_reasons.append("cross_timestep_covariance_unresolved")

    if model_bound_accepted:
        workflow_status = "accepted_with_model_bound"
    elif analysis.classification == "fit_inadequate":
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
        "qualified_systematics": (
            [
                "leading_model_difference_bounded_below_reporting_resolution",
                "fit_window_difference_bounded_below_reporting_resolution",
            ]
            if model_bound_accepted
            else []
        ),
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
        "energy_quality_assessment": energy_quality_assessment,
        "cross_timestep_covariance": cross_timestep_covariance,
        "energy_reporting_policy": (
            None if energy_reporting_policy is None else energy_reporting_policy.to_dict()
        ),
        "practical_resolution_assessment": practical_resolution,
        "extrapolation": analysis.to_dict(),
        "reference_energy_used_for_model_selection": False,
        "publication_ready_within_fixed_population_timestep_scope": (
            workflow_status in PUBLICATION_READY_WORKFLOW_STATUSES
        ),
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
                "extrapolated_energy_leading_model_intercept_spread": analysis_payload[
                    "leading_model_intercept_spread"
                ],
                "uncertainty_component_combination": (
                    "statistical and systematic components are reported separately; "
                    "systematic upper allowances are not combined in quadrature"
                ),
            }
        )
        if model_bound_accepted:
            payload.update(
                {
                    "extrapolated_energy_model_order_upper_allowance": (
                        practical_resolution["model_order"]["upper_allowance"]
                    ),
                    "extrapolated_energy_fit_window_upper_allowance": (
                        practical_resolution["fit_window"]["upper_allowance"]
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
        "energy_reporting_policy": (
            None if energy_reporting_policy is None else energy_reporting_policy.to_dict()
        ),
        "energy_quality_assessment": energy_quality_assessment,
        "inputs": [
            {
                "summary_path": str(item.summary_path),
                "summary_sha256": item.summary_sha256,
                "manifest_path": str(item.manifest_path),
                "manifest_sha256": item.manifest_sha256,
                "run_id": item.run_id,
                "bundle_sha256": item.bundle_sha256,
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
        artifacts = persist_timestep_outputs(
            output_dir=output_dir,
            payload=payload,
            config=config,
            points=loaded,
            command=command,
            run_name=TIMESTEP_EXTRAPOLATION_RUN_NAME,
            schema_version=TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
            status=workflow_status,
        )
    else:
        payload["artifacts"] = artifacts
    return attach_workflow_artifacts(payload, artifacts)
