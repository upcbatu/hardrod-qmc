from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import config_fingerprint
from hrdmc.statistics.timestep_fit import analyze_time_step_extrapolation
from hrdmc.uncertainty.timestep.assessment import (
    _apply_energy_quality_assessment,
    _control_variation,
    _cross_timestep_covariance,
    _input_quality,
    _practical_resolution_assessment,
    _sampling_design,
)
from hrdmc.uncertainty.timestep.contract import (
    PUBLICATION_READY_WORKFLOW_STATUSES,
    TIMESTEP_EXTRAPOLATION_RUN_NAME,
    EnergyReportingResolutionPolicy,
)
from hrdmc.uncertainty.timestep.outputs import (
    _validate_output_separation,
    persist_timestep_outputs,
)
from hrdmc.uncertainty.timestep.sources import (
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
    loaded, reference_identity = _load_inputs(
        summary_paths,
        output_dir=output_dir,
        write_artifacts=write_artifacts,
        energy_assessment_manifest=energy_assessment_manifest,
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
    workflow_status, unresolved_reasons, model_bound_accepted = _workflow_status(
        analysis, input_quality, cross_timestep_covariance, practical_resolution
    )
    payload = _payload(
        loaded=loaded,
        analysis=analysis,
        reference_identity=reference_identity,
        identity_fingerprint=identity_fingerprint,
        inputs=inputs,
        sensitivity_sigma=sensitivity_sigma,
        fit_alpha=fit_alpha,
        control_variation=control_variation,
        sampling_design=sampling_design,
        input_quality=input_quality,
        energy_quality_assessment=energy_quality_assessment,
        cross_timestep_covariance=cross_timestep_covariance,
        energy_reporting_policy=energy_reporting_policy,
        practical_resolution=practical_resolution,
        workflow_status=workflow_status,
        unresolved_reasons=unresolved_reasons,
        model_bound_accepted=model_bound_accepted,
    )
    config = _artifact_config(
        loaded=loaded,
        reference_identity=reference_identity,
        identity_fingerprint=identity_fingerprint,
        sensitivity_sigma=sensitivity_sigma,
        fit_alpha=fit_alpha,
        sampling_design=sampling_design,
        input_quality=input_quality,
        cross_timestep_covariance=cross_timestep_covariance,
        energy_reporting_policy=energy_reporting_policy,
        energy_quality_assessment=energy_quality_assessment,
    )
    artifacts = {
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
            status=workflow_status,
        )
    payload["workflow_artifacts"] = artifacts
    return payload


def _payload(
    *,
    loaded: list[Any],
    analysis: Any,
    reference_identity: dict[str, Any],
    identity_fingerprint: str,
    inputs: list[dict[str, Any]],
    sensitivity_sigma: float,
    fit_alpha: float,
    control_variation: dict[str, Any],
    sampling_design: dict[str, Any],
    input_quality: dict[str, Any],
    energy_quality_assessment: dict[str, Any] | None,
    cross_timestep_covariance: dict[str, Any],
    energy_reporting_policy: EnergyReportingResolutionPolicy | None,
    practical_resolution: dict[str, Any],
    workflow_status: str,
    unresolved_reasons: list[str],
    model_bound_accepted: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
            "walker population, initialization, propagator, "
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
    return payload


def _artifact_config(
    *,
    loaded: list[Any],
    reference_identity: dict[str, Any],
    identity_fingerprint: str,
    sensitivity_sigma: float,
    fit_alpha: float,
    sampling_design: dict[str, Any],
    input_quality: dict[str, Any],
    cross_timestep_covariance: dict[str, Any],
    energy_reporting_policy: EnergyReportingResolutionPolicy | None,
    energy_quality_assessment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
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
                "dt": item.point.dt,
            }
            for item in loaded
        ],
    }


def _load_inputs(
    summary_paths: Sequence[Path],
    *,
    output_dir: Path | None,
    write_artifacts: bool,
    energy_assessment_manifest: Path | None,
) -> tuple[list[Any], dict[str, Any]]:
    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when artifacts are written")
    resolved = [Path(path).resolve() for path in summary_paths]
    if len(resolved) < 3:
        raise ValueError("time-step extrapolation requires at least three summaries")
    if len(set(resolved)) != len(resolved):
        raise ValueError("time-step summary paths must be unique")
    if write_artifacts:
        assert output_dir is not None
        target = output_dir.resolve()
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(
                f"time-step extrapolation output directory is not empty: {target}"
            )
        protected = list(resolved)
        if energy_assessment_manifest is not None:
            protected.append(energy_assessment_manifest.resolve())
        _validate_output_separation(target, protected)
    loaded = [_load_time_step_point(path) for path in resolved]
    identity = loaded[0].identity
    mismatches = [
        (point.summary_path, _different_identity_fields(identity, point.identity))
        for point in loaded[1:]
        if point.identity != identity
    ]
    if mismatches:
        details = "; ".join(f"{path}: {', '.join(fields)}" for path, fields in mismatches)
        raise ValueError(
            "time-step summaries do not share the required case, guide, source, "
            f"walker, initialization, and drift identity: {details}"
        )
    return loaded, identity


def _workflow_status(
    analysis: Any,
    input_quality: dict[str, Any],
    covariance: dict[str, Any],
    resolution: dict[str, Any],
) -> tuple[str, list[str], bool]:
    model_bound = resolution["accepted_with_model_bound"] is True
    reasons = []
    if analysis.classification == "fit_inadequate":
        reasons.append("fit_inadequate")
    elif analysis.classification == "model_sensitive" and not model_bound:
        reasons.append("model_sensitive")
    checks = (
        (analysis.fit_window_status == "accepted", "fit_window_unresolved"),
        (input_quality["publication_accepted"], "input_quality_unresolved"),
        (covariance["status"] == "accepted", "cross_timestep_covariance_unresolved"),
    )
    reasons.extend(message for accepted, message in checks if not accepted)
    if model_bound:
        status = "accepted_with_model_bound"
    elif analysis.classification in ("fit_inadequate", "model_sensitive"):
        status = analysis.classification
    elif reasons:
        status = reasons[0]
    elif input_quality["status"] == "accepted_with_warnings":
        status = "accepted_with_warnings"
    else:
        status = "accepted"
    return status, reasons, model_bound


def _different_identity_fields(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key)
    )
