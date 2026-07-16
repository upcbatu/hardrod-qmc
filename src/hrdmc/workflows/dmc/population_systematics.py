from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.analysis import assess_matrix_energy_stationarity
from hrdmc.analysis.population_systematics import (
    PopulationEnergyPoint,
    PopulationLadderAssessment,
    TimeStepPopulationInteraction,
    analyze_population_ladder,
    analyze_timestep_population_interaction,
)
from hrdmc.artifacts import (
    build_run_provenance,
    config_fingerprint,
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    write_json,
    write_run_manifest,
)
from hrdmc.workflows.dmc.benchmark_packet.matrix_assembly import (
    load_final_matrix_energy_selection,
)
from hrdmc.workflows.dmc.benchmark_packet.selection import energy_validation_status
from hrdmc.workflows.dmc.trapped import parse_case

POPULATION_SYSTEMATICS_SCHEMA_VERSION = "dmc_population_systematics_v3"
POPULATION_SYSTEMATICS_RUN_NAME = "dmc_population_systematics"
FIXED_ENERGY_REPORTING_RESOLUTION = 0.01
FIXED_ENERGY_REPORTING_UNIT = "hbar*Omega"
FIXED_ENERGY_STATIONARITY_CONFIDENCE_LEVEL = 0.95
FIXED_ENERGY_STATIONARITY_RHAT_LIMIT = 1.05
FIXED_ENERGY_STATIONARITY_MIN_EFFECTIVE_SAMPLES = 30.0
SUPPORTED_INPUTS = {
    ("dmc_benchmark_packet", "dmc_benchmark_packet_v3"),
    ("dmc_trapped_stationarity_grid", "dmc_trapped_stationarity_grid_v2"),
}
ENERGY_CHAIN_ACCEPTED_STATUSES = {"accepted", "spread_warning"}
ENERGY_UNCERTAINTY_ACCEPTED_STATUSES = {
    "accepted",
    "conservative_error_inflated",
    "blocking_plateau_unresolved_correlated_error_available",
}
PUBLICATION_READY_STATUSES = {
    "accepted_finite_population_bound",
    "accepted_population_limit",
    "accepted_with_warnings",
}


@dataclass(frozen=True)
class LoadedPopulationPoint:
    point: PopulationEnergyPoint
    dt: float
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
    stationarity: dict[str, Any]
    controls: dict[str, Any]
    manifest_verification_warnings: tuple[str, ...]
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.point.to_dict(),
            "dt": self.dt,
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
            "controls": self.controls,
            "manifest_verification": (
                "summary_bound_with_unrelated_artifact_warnings"
                if self.manifest_verification_warnings
                else "verified"
            ),
            "manifest_verification_warnings": list(self.manifest_verification_warnings),
            "telemetry": self.telemetry,
        }


def run_population_systematics_workflow(
    summary_paths: Sequence[Path],
    *,
    reporting_resolution: float,
    output_dir: Path | None,
    command: list[str] | None = None,
    write_artifacts: bool = True,
    confidence_level: float = 0.95,
    fit_alpha: float = 0.05,
    interaction_dt: float | None = None,
    selected_dt: float | None = None,
    energy_assessment_manifest: Path | None = None,
    apply_simultaneous_energy_stationarity: bool = False,
) -> dict[str, Any]:
    """Assess finite-walker energy sensitivity and one timestep interaction.

    When two timestep treatments are supplied, ``selected_dt`` identifies the
    treatment whose finite-population or population-limit result is being
    qualified.  The smaller timestep remains the reference treatment for the
    four-corner interaction; it is not selected implicitly.
    """

    _validate_reporting_policy(
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
        fit_alpha=fit_alpha,
    )
    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when artifacts are written")
    resolved_paths = [Path(path).resolve() for path in summary_paths]
    if len(resolved_paths) < 2:
        raise ValueError("population systematics requires at least W and 2W summaries")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("population summary paths must be unique")
    if energy_assessment_manifest is not None and apply_simultaneous_energy_stationarity:
        raise ValueError(
            "final-matrix and simultaneous population energy assessments are mutually exclusive"
        )
    source_artifact_paths = list(resolved_paths)
    if energy_assessment_manifest is not None:
        source_artifact_paths.append(energy_assessment_manifest.resolve())
    if write_artifacts:
        assert output_dir is not None
        _validate_output_separation(output_dir.resolve(), source_artifact_paths)

    loaded = [_load_population_point(path) for path in resolved_paths]
    loaded.sort(key=lambda item: (item.dt, item.point.walkers))
    _validate_shared_identity(loaded)
    _validate_reporting_unit(loaded[0].identity)
    energy_quality_assessment: dict[str, Any] | None = None
    if apply_simultaneous_energy_stationarity:
        loaded, energy_quality_assessment = _apply_simultaneous_energy_stationarity_assessment(
            loaded
        )
    elif energy_assessment_manifest is not None:
        loaded, energy_quality_assessment = _apply_energy_quality_assessment(
            loaded,
            energy_assessment_manifest=energy_assessment_manifest.resolve(),
        )
    groups = _group_by_timestep(loaded)
    if len(groups) > 2:
        raise ValueError(
            "population workflow accepts one reference timestep and at most one coarse timestep"
        )
    reference_fine_dt = min(groups)
    coarse_dt = max(groups) if len(groups) == 2 else None
    if len(groups) == 2 and selected_dt is None:
        raise ValueError("selected_dt is required when two timestep treatments are supplied")
    selected_timestep = _resolve_selected_timestep(
        groups,
        selected_dt=selected_dt,
    )
    selected_dt_basis = "explicit_selected_dt" if len(groups) == 2 else "only_supplied_timestep"
    selected_treatment_role = (
        "reference_fine" if selected_timestep == reference_fine_dt else "coarse"
    )

    analyses: dict[float, PopulationLadderAssessment] = {}
    for dt, group in groups.items():
        role = "reference fine timestep" if dt == reference_fine_dt else "coarse timestep"
        _validate_population_group(group, name=role)
        analyses[dt] = analyze_population_ladder(
            [item.point for item in group],
            reporting_resolution=reporting_resolution,
            confidence_level=confidence_level,
            fit_alpha=fit_alpha,
        )
    reference_fine_group = groups[reference_fine_dt]
    reference_fine_analysis = analyses[reference_fine_dt]
    coarse_group = None if coarse_dt is None else groups[coarse_dt]
    coarse_analysis = None if coarse_dt is None else analyses[coarse_dt]
    selected_analysis = analyses[selected_timestep]
    selected_walkers = selected_analysis.reference_walkers
    selected_walkers_basis = "reference_population_of_selected_timestep_ladder"

    interaction: TimeStepPopulationInteraction | None = None
    if len(groups) == 2:
        assert coarse_dt is not None
        assert coarse_group is not None
        if interaction_dt is not None and not math.isclose(
            interaction_dt,
            coarse_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("interaction_dt does not match the supplied coarse timestep")
        _validate_cross_timestep_controls(reference_fine_group, coarse_group)
        fine_pair = _reference_doubling_pair(reference_fine_group)
        coarse_pair = _reference_doubling_pair(coarse_group)
        interaction = analyze_timestep_population_interaction(
            [item.point for item in fine_pair],
            [item.point for item in coarse_pair],
            reporting_resolution=reporting_resolution,
            confidence_level=confidence_level,
        )
    elif interaction_dt is not None:
        raise ValueError("interaction_dt requires a supplied coarse-timestep W/2W pair")

    input_quality = _input_quality(loaded)
    classification = selected_analysis.classification
    unresolved_reasons: list[str] = []
    if not input_quality["publication_accepted"]:
        unresolved_reasons.append("input_quality_unresolved")
    if classification == "additional_population_point_required":
        unresolved_reasons.append("additional_population_point_required")
    elif classification == "population_sensitive":
        unresolved_reasons.append("population_sensitive")
    if interaction is None:
        unresolved_reasons.append("timestep_population_interaction_not_assessed")
    elif not interaction.bounded_below_reporting_resolution:
        unresolved_reasons.append("timestep_population_interaction_unresolved")

    accepted_classification = classification in {
        "accepted_finite_population_bound",
        "accepted_population_limit",
    }
    if not input_quality["publication_accepted"]:
        status = "input_quality_unresolved"
    elif not accepted_classification:
        status = classification
    elif interaction is None:
        status = "timestep_population_interaction_not_assessed"
    elif not interaction.bounded_below_reporting_resolution:
        status = "timestep_population_interaction_unresolved"
    elif input_quality["status"] == "accepted_with_warnings":
        status = "accepted_with_warnings"
    else:
        status = classification
    publication_ready = status in PUBLICATION_READY_STATUSES

    reporting_policy = {
        "resolution": reporting_resolution,
        "confidence_level": confidence_level,
        "energy_unit": FIXED_ENERGY_REPORTING_UNIT,
        "rationale": (
            "predeclared absolute thesis reporting resolution for population "
            "and timestep-population sensitivity"
        ),
        "timing": "prospective",
    }
    statistical_method_policy = {
        "contrast_standard_error": (
            "max(matched-replicate contrast SEM, coefficient-weighted "
            "source-run quadrature standard error)"
        ),
        "confidence_rule": "apply the Student-t critical value once",
        "legacy_l1_role": (
            "worst_case_arbitrary_covariance_envelope retained as a diagnostic only"
        ),
        "method_correction_timing": "retrospective",
        "method_correction_scope": "applied uniformly to every population case",
    }
    sampling_design = _sampling_design(loaded)
    identity = loaded[0].identity
    identity_fingerprint = config_fingerprint(identity)
    reference_fine_payload = reference_fine_analysis.to_dict()
    coarse_payload = None if coarse_analysis is None else coarse_analysis.to_dict()
    selected_payload = selected_analysis.to_dict()
    population_bounds = {
        "reference_fine_last_doubling_upper_allowance": (
            reference_fine_analysis.last_doubling.upper_allowance
        ),
        "coarse_last_doubling_upper_allowance": (
            None if coarse_analysis is None else coarse_analysis.last_doubling.upper_allowance
        ),
        "selected_last_doubling_upper_allowance": (selected_analysis.last_doubling.upper_allowance),
        "timestep_population_interaction_upper_allowance": (
            None if interaction is None else interaction.upper_allowance
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": POPULATION_SYSTEMATICS_SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "unresolved_reasons": unresolved_reasons,
        "qualified_systematics": _qualified_systematics(
            publication_ready=publication_ready,
            classification=classification,
            interaction=interaction,
        ),
        "diagnostic": "DMC mixed-energy walker-population sensitivity",
        "case_id": loaded[0].case_id,
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "energy_reporting_policy": reporting_policy,
        "statistical_method_policy": statistical_method_policy,
        "sampling_design": sampling_design,
        "fit_alpha": fit_alpha,
        "reference_fine_dt": reference_fine_dt,
        "coarse_dt": coarse_dt,
        "selected_dt": selected_timestep,
        "selected_dt_basis": selected_dt_basis,
        "selected_treatment_role": selected_treatment_role,
        "selected_walkers": selected_walkers,
        "selected_walkers_basis": selected_walkers_basis,
        "timestep_population_interaction_status": (
            "not_assessed"
            if interaction is None
            else "bounded_below_reporting_resolution"
            if interaction.bounded_below_reporting_resolution
            else "unresolved"
        ),
        "input_summaries": [point.to_dict() for point in loaded],
        "input_quality": input_quality,
        "energy_quality_assessment": energy_quality_assessment,
        "reference_fine_population_ladder": reference_fine_payload,
        "coarse_population_ladder": coarse_payload,
        "selected_population_ladder": selected_payload,
        "population_bounds": population_bounds,
        "timestep_population_interaction": (None if interaction is None else interaction.to_dict()),
        "publication_ready_within_population_systematic_scope": publication_ready,
        "uncertainty_component_combination": (
            "source-run uncertainties are propagated in quadrature only within each "
            "declared linear contrast; population, Richardson-window, and timestep-"
            "population allowances remain separate from the zero-step systematic"
        ),
        "scientific_scope": (
            "The artifact bounds finite-walker sensitivity at fixed physical and "
            "sampler controls and requires a separate four-corner timestep-population "
            "interaction before publication readiness. It does not combine or replace "
            "the separate zero-time-step extrapolation."
        ),
    }
    if publication_ready:
        payload["selected_population_last_doubling_upper_allowance"] = (
            selected_analysis.last_doubling.upper_allowance
        )
        if classification == "accepted_population_limit":
            assert selected_analysis.inverse_population_fit is not None
            assert selected_analysis.richardson_window is not None
            assert selected_analysis.population_limit_correction is not None
            correction = selected_analysis.population_limit_correction
            payload.update(
                {
                    "selected_energy_population_basis": ("population_limit_at_selected_timestep"),
                    "population_limit_energy_at_selected_timestep": (
                        selected_analysis.inverse_population_fit.intercept
                    ),
                    "population_limit_energy_statistical_stderr": (
                        selected_analysis.inverse_population_fit.intercept_stderr
                    ),
                    "population_limit_model_window_upper_allowance": (
                        selected_analysis.richardson_window.upper_allowance
                    ),
                    "population_limit_correction_at_selected_timestep": correction.value,
                    "population_limit_correction_statistical_stderr": (
                        correction.conservative_standard_error
                    ),
                    "population_limit_correction_matched_seed_standard_error": (
                        correction.matched_seed_standard_error
                    ),
                    "population_limit_correction_source_run_quadrature_standard_error": (
                        correction.source_run_quadrature_standard_error
                    ),
                    "population_limit_correction_worst_case_arbitrary_covariance_"
                    "standard_error_envelope": (
                        correction.worst_case_arbitrary_covariance_standard_error_envelope
                    ),
                    "population_limit_correction_basis": (
                        "E_infinity(selected_dt) - E(selected_dt, selected_walkers)"
                    ),
                    "downstream_zero_timestep_population_scope": (
                        "apply_population_limit_correction_to_selected_finite_population_"
                        "zero_timestep_energy"
                    ),
                }
            )
        else:
            selected_reference_point = _reference_population_point(selected_analysis)
            payload.update(
                {
                    "selected_energy_population_basis": ("finite_population_at_selected_walkers"),
                    "finite_population_energy_at_selected_timestep": (
                        selected_reference_point.energy
                    ),
                    "finite_population_energy_statistical_stderr": (
                        selected_reference_point.conservative_stderr
                    ),
                    "finite_population_walkers": selected_reference_point.walkers,
                    "finite_population_energy_basis": (
                        "selected timestep reference population W; the W-to-2W "
                        "difference is bounded below the reporting resolution"
                    ),
                    "finite_population_w_to_2w_upper_allowance": (
                        selected_analysis.last_doubling.upper_allowance
                    ),
                    "downstream_zero_timestep_population_scope": (
                        "retain_selected_finite_population_zero_timestep_energy; "
                        "no infinite-population central value is claimed"
                    ),
                }
            )
        if interaction is not None:
            payload["timestep_population_interaction_upper_allowance"] = interaction.upper_allowance

    config = {
        "case_id": loaded[0].case_id,
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "energy_reporting_policy": reporting_policy,
        "statistical_method_policy": statistical_method_policy,
        "sampling_design": sampling_design,
        "fit_alpha": fit_alpha,
        "reference_fine_dt": reference_fine_dt,
        "coarse_dt": coarse_dt,
        "selected_dt": selected_timestep,
        "selected_dt_basis": selected_dt_basis,
        "selected_treatment_role": selected_treatment_role,
        "selected_walkers": selected_walkers,
        "selected_walkers_basis": selected_walkers_basis,
        "energy_quality_assessment": energy_quality_assessment,
        "inputs": [
            {
                "summary_path": str(point.summary_path),
                "summary_sha256": point.summary_sha256,
                "manifest_path": str(point.manifest_path),
                "manifest_sha256": point.manifest_sha256,
                "run_id": point.run_id,
                "bundle_sha256": point.bundle_sha256,
                "dt": point.dt,
                "walkers": point.point.walkers,
            }
            for point in loaded
        ],
    }
    artifacts: dict[str, str | None] = {
        "summary": None,
        "point_table": None,
        "comparison_table": None,
        "run_manifest": None,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
    }
    if write_artifacts:
        assert output_dir is not None
        root = ensure_dir(output_dir.resolve())
        summary_path = root / "summary.json"
        point_table_path = _write_point_table(root, loaded)
        comparison_table_path = _write_comparison_table(
            root,
            reference_fine_dt=reference_fine_dt,
            reference_fine_analysis=reference_fine_analysis,
            coarse_dt=coarse_dt,
            coarse_analysis=coarse_analysis,
            selected_dt=selected_timestep,
            selected_dt_basis=selected_dt_basis,
            interaction=interaction,
        )
        payload["artifacts"] = {
            "summary": str(summary_path),
            "point_table": str(point_table_path),
            "comparison_table": str(comparison_table_path),
            "run_manifest": str(root / "run_manifest.json"),
            "output_dir": str(root),
        }
        write_json(summary_path, payload)
        manifest_path = write_run_manifest(
            root,
            run_name=POPULATION_SYSTEMATICS_RUN_NAME,
            config=config,
            artifacts=[summary_path, point_table_path, comparison_table_path],
            schema_version=POPULATION_SYSTEMATICS_SCHEMA_VERSION,
            provenance=build_run_provenance(command),
            status=status,
        )
        artifacts = {
            "summary": str(summary_path),
            "point_table": str(point_table_path),
            "comparison_table": str(comparison_table_path),
            "run_manifest": str(manifest_path),
            "output_dir": str(root),
        }
    else:
        payload["artifacts"] = artifacts
    payload["workflow_artifacts"] = artifacts
    return payload


def _load_population_point(summary_path: Path) -> LoadedPopulationPoint:
    if not summary_path.is_file():
        raise FileNotFoundError(f"population summary does not exist: {summary_path}")
    manifest_path = summary_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
    summary = _load_mapping(summary_path, "summary")
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        summary_path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    run_name = _required_string(manifest, "run_name")
    result_schema = _required_string(manifest, "result_schema_version")
    if (run_name, result_schema) not in SUPPORTED_INPUTS:
        raise ValueError(f"unsupported population input {run_name}/{result_schema}: {summary_path}")
    if summary.get("schema_version") != result_schema:
        raise ValueError(f"summary schema does not match its manifest: {summary_path}")
    if summary.get("status") != manifest.get("status"):
        raise ValueError(f"summary status does not match its manifest: {summary_path}")
    config = _required_mapping(manifest, "config")
    manifest_controls = _required_mapping(config, "controls")
    summary_controls = summary.get("controls")
    if not isinstance(summary_controls, dict):
        raise ValueError(f"summary controls must be a mapping: {summary_path}")
    if summary_controls != manifest_controls:
        raise ValueError(f"summary controls do not match the manifest: {summary_path}")
    controls = summary_controls
    _validate_step_tau_controls(controls, summary_path=summary_path)
    case_id = _case_id(summary, config, run_name=run_name)
    _verify_summary_method_identity(summary, config, summary_path=summary_path)
    stationarity, energy, stderr, energy_status = _energy_fields(
        summary,
        run_name=run_name,
        summary_path=summary_path,
    )
    _verify_stationarity_collective_rn(
        stationarity,
        declared_controls=(
            summary["collective_rn"]
            if "collective_rn" in summary
            else summary["collective_rn_controls"]
        ),
        summary_path=summary_path,
    )
    _verify_case_metadata(
        summary,
        stationarity,
        case_id=case_id,
        run_name=run_name,
        summary_path=summary_path,
    )
    _verify_conservative_stderr(stationarity, stderr, summary_path=summary_path)
    if stationarity.get("base_numerics_valid") is not True:
        raise ValueError(f"population point failed base numerical checks: {summary_path}")
    if stationarity.get("population_weights_controlled") is not True:
        raise ValueError(f"population point has uncontrolled population weights: {summary_path}")
    seeds = _verified_seeds(
        summary,
        config,
        stationarity,
        run_name=run_name,
        summary_path=summary_path,
    )
    seed_energies = _seed_energies(stationarity, seeds=seeds, summary_path=summary_path)
    point = PopulationEnergyPoint(
        walkers=_required_int(controls.get("walkers"), "walkers"),
        energy=energy,
        conservative_stderr=stderr,
        seed_ids=seeds,
        seed_energies=seed_energies,
        label=str(summary_path),
    )
    identity = _scientific_identity(
        summary,
        manifest,
        controls=controls,
        stationarity=stationarity,
        case_id=case_id,
    )
    return LoadedPopulationPoint(
        point=point,
        dt=_required_positive_float(controls.get("dt"), "dt"),
        case_id=case_id,
        identity=identity,
        summary_path=summary_path,
        summary_sha256=file_sha256(summary_path),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        run_name=run_name,
        result_schema_version=result_schema,
        run_id=_required_string(manifest, "run_id"),
        bundle_sha256=_required_string(manifest, "bundle_sha256"),
        run_status=_required_string(manifest, "status"),
        energy_status=energy_status,
        energy_quality=_energy_input_quality(stationarity, reported_status=energy_status),
        energy_quality_assessment=None,
        stationarity=stationarity,
        controls=controls,
        manifest_verification_warnings=tuple(warnings),
        telemetry=_point_telemetry(stationarity),
    )


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
        raise ValueError("population stationarity summary must contain exactly one case")
    case_id = _required_string(cases[0], "case_id")
    if config.get("cases") != [case_id]:
        raise ValueError("stationarity summary case does not match its manifest")
    return case_id


def _verify_case_metadata(
    summary: dict[str, Any],
    stationarity: dict[str, Any],
    *,
    case_id: str,
    run_name: str,
    summary_path: Path,
) -> None:
    case = parse_case(case_id)
    if case.case_id != case_id:
        raise ValueError(f"case id is not in canonical harmonic-oscillator form: {summary_path}")
    expected: dict[str, Any] = {
        "case_id": case_id,
        "case_parameterization": "harmonic_oscillator_units",
        "n_particles": case.n_particles,
        "rod_length_ho": case.rod_length_ho,
        **case.unit_metadata(),
    }
    owners = [("case", stationarity)]
    if run_name == "dmc_benchmark_packet":
        owners.append(("benchmark", summary))
    for owner_name, owner in owners:
        for field, expected_value in expected.items():
            if owner.get(field) != expected_value:
                if field in {"energy_coordinate", "energy_unit", "report_energy_unit"}:
                    raise ValueError(
                        f"{owner_name} energy identity {field} must match "
                        f"{expected_value}: {summary_path}"
                    )
                raise ValueError(
                    f"{owner_name} {field} is not canonical for {case_id}: {summary_path}"
                )


def _verify_summary_method_identity(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    for field in (
        "guide_family",
        "initialization_mode",
        "init_width_log_sigma",
        "breathing_preburn_steps",
        "breathing_preburn_log_step",
    ):
        if summary.get(field) != config.get(field):
            raise ValueError(f"summary {field} does not match its manifest: {summary_path}")
    summary_guide = _required_mapping(summary, "guide_parameters")
    manifest_guide = _manifest_guide_parameters(config)
    if summary_guide != manifest_guide:
        raise ValueError(f"guide identity does not match the manifest: {summary_path}")
    if "collective_rn" not in config:
        raise ValueError(f"manifest collective_rn identity is missing: {summary_path}")
    summary_collective_rn_field = (
        "collective_rn" if "collective_rn" in summary else "collective_rn_controls"
    )
    if summary_collective_rn_field not in summary:
        raise ValueError(f"summary collective RN identity is missing: {summary_path}")
    if summary[summary_collective_rn_field] != config["collective_rn"]:
        raise ValueError(
            f"summary collective RN identity does not match its manifest: {summary_path}"
        )


def _verify_stationarity_collective_rn(
    stationarity: Mapping[str, Any],
    *,
    declared_controls: Any,
    summary_path: Path,
) -> None:
    if "collective_rn_controls" not in stationarity:
        raise ValueError(f"stationarity collective RN controls are missing: {summary_path}")
    if stationarity["collective_rn_controls"] != declared_controls:
        raise ValueError(
            f"stationarity collective RN controls disagree with the declared mode: {summary_path}"
        )
    expected_enabled = declared_controls is not None
    if stationarity.get("collective_rn_enabled") is not expected_enabled:
        raise ValueError(
            "stationarity collective RN enabled flag disagrees with the declared mode: "
            f"{summary_path}"
        )


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
        estimate = _required_mapping(_required_mapping(summary, "estimates"), "energy")
        energy = _required_float(estimate.get("value"), "energy value")
        stderr = _required_positive_float(estimate.get("stderr"), "energy stderr")
        if energy != _required_float(stationarity.get("mixed_energy"), "mixed energy"):
            raise ValueError(f"energy estimate disagrees with stationarity: {summary_path}")
        if stderr != _required_positive_float(
            stationarity.get("mixed_energy_conservative_stderr"),
            "conservative energy stderr",
        ):
            raise ValueError(f"energy stderr disagrees with stationarity: {summary_path}")
        declared_status = _required_string(summary, "energy_validation_status")
        estimate_status = _required_string(estimate, "status")
        recomputed_status = energy_validation_status(stationarity)
        if declared_status != recomputed_status:
            raise ValueError(
                f"benchmark energy validation status is not reproducible: {summary_path}"
            )
        if estimate_status != declared_status:
            raise ValueError(
                f"benchmark energy estimate status disagrees with validation status: {summary_path}"
            )
        return stationarity, energy, stderr, estimate_status
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
    reported = [
        _seed_list(stationarity.get("seeds"), "stationarity seeds"),
        _seed_ids_from_rows(
            stationarity.get("seed_summaries"),
            "stationarity seed_summaries",
        ),
    ]
    if run_name == "dmc_benchmark_packet":
        reported.extend(
            [
                _seed_list(summary.get("seeds"), "benchmark seeds"),
                _seed_ids_from_rows(summary.get("seed_results"), "benchmark seed_results"),
            ]
        )
    if any(seeds != config_seeds for seeds in reported):
        raise ValueError(f"reported seeds do not match manifest seeds: {summary_path}")
    for owner in (summary, stationarity):
        if owner.get("seed_count") not in {None, len(config_seeds)}:
            raise ValueError(f"seed_count does not match manifest seeds: {summary_path}")
    return config_seeds


def _seed_energies(
    stationarity: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    summary_path: Path,
) -> np.ndarray:
    rows = stationarity.get("seed_summaries")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"stationarity seed summaries are invalid: {summary_path}")
    values = np.asarray(
        [_required_float(row.get("mixed_energy"), "seed mixed energy") for row in rows],
        dtype=np.float64,
    )
    if values.shape != (len(seeds),):
        raise ValueError(f"seed energy count does not match manifest seeds: {summary_path}")
    return values


def _scientific_identity(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    *,
    controls: dict[str, Any],
    stationarity: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    implementation = _required_mapping(
        _required_mapping(manifest, "provenance"),
        "implementation",
    )
    cases = summary.get("cases")
    case_summary = (
        cases[0]
        if isinstance(cases, list) and len(cases) == 1 and isinstance(cases[0], dict)
        else summary
    )
    return {
        "case_id": case_id,
        "case_parameterization": _required_string(case_summary, "case_parameterization"),
        "n_particles": _required_int(case_summary.get("n_particles"), "n_particles"),
        "rod_length_ho": _required_float(case_summary.get("rod_length_ho"), "rod_length_ho"),
        "coordinate": _required_string(case_summary, "coordinate"),
        "length_unit": _required_string(case_summary, "length_unit"),
        "time_unit": _required_string(case_summary, "time_unit"),
        "guide_family": summary.get("guide_family"),
        "guide_parameters": _required_mapping(summary, "guide_parameters"),
        "implementation_source_tree_sha256": _required_string(
            implementation,
            "source_tree_sha256",
        ),
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
        "collective_rn": (
            summary["collective_rn"]
            if "collective_rn" in summary
            else summary["collective_rn_controls"]
        ),
    }


def _energy_semantics(
    summary: dict[str, Any],
    stationarity: dict[str, Any],
) -> dict[str, str]:
    labels = [
        value
        for value in (stationarity.get("energy_estimator"), _nested_energy_estimator(summary))
        if isinstance(value, str) and value
    ]
    if not labels or any(
        "mixed" not in label.lower()
        or "local" not in label.lower()
        or "energy" not in label.lower()
        for label in labels
    ):
        raise ValueError("population analysis requires a mixed local-energy estimator")
    return {
        "estimator": "mixed_local_energy",
        "energy_unit": _consistent_semantic_string(summary, stationarity, "energy_unit"),
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
    values = [value for value in (summary.get(field), stationarity.get(field)) if value]
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{field} must be recorded for population analysis")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"summary and stationarity {field} values differ")
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
    finite = [
        float(value)
        for value in components
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if finite and stderr + 1.0e-15 < max(finite):
        raise ValueError(
            f"declared conservative energy stderr is smaller than a component: {summary_path}"
        )


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
    publication_status = (
        "accepted_with_precision_warning"
        if precision_warning
        else "accepted"
        if publication_accepted
        else "unresolved"
    )
    return {
        "validation_passed": validation_passed,
        "method_status": method_status,
        "energy_chain_status": chain_status,
        "precision_status": stationarity.get("precision_status"),
        "energy_uncertainty_status": stationarity.get("mixed_energy_uncertainty_status"),
        "status_basis": "source_summary",
        "source_publication_accepted": publication_accepted,
        "source_publication_status": publication_status,
        "publication_accepted": publication_accepted,
        "publication_status": publication_status,
    }


def _apply_simultaneous_energy_stationarity_assessment(
    points: Sequence[LoadedPopulationPoint],
) -> tuple[list[LoadedPopulationPoint], dict[str, Any]]:
    treatment_ids = [_population_treatment_id(point) for point in points]
    if len(set(treatment_ids)) != len(treatment_ids):
        raise ValueError("population stationarity treatment identities must be unique")
    raw_assessment = assess_matrix_energy_stationarity(
        {
            treatment_id: point.stationarity
            for treatment_id, point in zip(treatment_ids, points, strict=True)
        },
        confidence_level=FIXED_ENERGY_STATIONARITY_CONFIDENCE_LEVEL,
        rhat_limit=FIXED_ENERGY_STATIONARITY_RHAT_LIMIT,
        min_effective_samples=FIXED_ENERGY_STATIONARITY_MIN_EFFECTIVE_SAMPLES,
    )
    assessment = {
        **raw_assessment,
        "assessment_type": "simultaneous_population_energy_stationarity",
        "treatment_count": raw_assessment["case_count"],
        "treatments": raw_assessment["cases"],
    }
    assessment.pop("case_count")
    assessment.pop("cases")
    assessment["policy_timing"] = "retrospective"
    assessment["scope"] = "supplied_population_treatments"
    assessment["threshold_basis"] = (
        "prospective_population_source_rhat_and_effective-sample criteria with "
        "family-wise directional correction"
    )
    treatment_assessments = _required_mapping(assessment, "treatments")
    updated: list[LoadedPopulationPoint] = []
    bindings: list[dict[str, Any]] = []
    for treatment_id, point in zip(treatment_ids, points, strict=True):
        treatment_assessment = _required_mapping(treatment_assessments, treatment_id)
        source_quality = point.energy_quality
        source_already_accepted = source_quality["publication_accepted"]
        override_eligibility = _trace_stationarity_override_eligibility(source_quality)
        publication_accepted = treatment_assessment.get("status") == "accepted" and (
            source_already_accepted or override_eligibility["eligible"]
        )
        if publication_accepted and source_already_accepted:
            publication_status = source_quality["publication_status"]
        elif publication_accepted:
            publication_status = "accepted_with_retrospective_assessment"
        else:
            publication_status = "unresolved"
        assessed_quality = {
            **source_quality,
            "status_basis": "retrospective_population_stationarity_assessment",
            "source_publication_accepted": source_quality["publication_accepted"],
            "source_publication_status": source_quality["publication_status"],
            "retrospective_override_eligibility": override_eligibility,
            "publication_accepted": publication_accepted,
            "publication_status": publication_status,
        }
        binding = {
            "treatment_id": treatment_id,
            "summary_path": str(point.summary_path),
            "summary_sha256": point.summary_sha256,
            "manifest_path": str(point.manifest_path),
            "manifest_sha256": point.manifest_sha256,
            "run_id": point.run_id,
            "bundle_sha256": point.bundle_sha256,
        }
        point_assessment = {
            "assessment_type": assessment["assessment_type"],
            "assessment_method": assessment["method"],
            "assessment_scope": assessment["scope"],
            "policy_timing": assessment["policy_timing"],
            "threshold_basis": assessment["threshold_basis"],
            "confidence_level": assessment["confidence_level"],
            "directional_test_count": assessment["directional_test_count"],
            "simultaneous_two_sided_critical_z": assessment["simultaneous_two_sided_critical_z"],
            "rhat_limit": assessment["rhat_limit"],
            "min_effective_samples": assessment["min_effective_samples"],
            "source_binding": binding,
            "source_override_eligibility": override_eligibility,
            "treatment_assessment": treatment_assessment,
        }
        updated.append(
            replace(
                point,
                energy_quality=assessed_quality,
                energy_quality_assessment=point_assessment,
            )
        )
        bindings.append(binding)
    return updated, {
        "assessment_type": assessment["assessment_type"],
        "verification_scope": "population_workflow_manifest_bound_inputs",
        "policy_timing": assessment["policy_timing"],
        "threshold_basis": assessment["threshold_basis"],
        "assessment_scope": assessment["scope"],
        "assessment_method": assessment["method"],
        "input_bindings": bindings,
        "simultaneous_assessment": assessment,
    }


def _population_treatment_id(point: LoadedPopulationPoint) -> str:
    return f"dt={point.dt:.17g};walkers={point.point.walkers}"


def _trace_stationarity_override_eligibility(
    source_quality: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if source_quality.get("publication_accepted") is True:
        failures.append("source_already_accepted")
    if source_quality.get("method_status") != "trace_nonstationary":
        failures.append("source_method_status_is_not_trace_nonstationary")
    if source_quality.get("energy_chain_status") != "trace_nonstationary":
        failures.append("source_energy_chain_status_is_not_trace_nonstationary")
    if source_quality.get("energy_uncertainty_status") not in (
        ENERGY_UNCERTAINTY_ACCEPTED_STATUSES
    ):
        failures.append("source_energy_uncertainty_is_unresolved")
    return {
        "eligible": not failures,
        "criterion": (
            "only an unresolved source whose method and energy-chain statuses are "
            "both trace_nonstationary and whose conservative energy uncertainty is "
            "available can be reclassified"
        ),
        "failures": failures,
    }


def _apply_energy_quality_assessment(
    points: Sequence[LoadedPopulationPoint],
    *,
    energy_assessment_manifest: Path,
) -> tuple[list[LoadedPopulationPoint], dict[str, Any]]:
    case_ids = {point.case_id for point in points}
    if len(case_ids) != 1:
        raise ValueError("energy assessment requires one shared case identity")
    selection = load_final_matrix_energy_selection(
        energy_assessment_manifest,
        case_id=next(iter(case_ids)),
    )
    assessment = {
        "assessment_type": "final_matrix_energy_selection",
        **selection,
    }
    selected_summary = Path(str(selection["selected_summary_path"])).resolve()
    selected_manifest = Path(str(selection["selected_manifest_path"])).resolve()
    matches = [
        index
        for index, point in enumerate(points)
        if point.summary_path == selected_summary
        and point.summary_sha256 == selection["selected_summary_sha256"]
        and point.manifest_path == selected_manifest
        and point.manifest_sha256 == selection["selected_manifest_sha256"]
        and point.run_id == selection["selected_run_id"]
        and point.bundle_sha256 == selection["selected_bundle_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "energy assessment must select exactly one population input by path and digest"
        )
    index = matches[0]
    selected = points[index]
    source_quality = selected.energy_quality
    assessed_quality = {
        **source_quality,
        "status_basis": selection["energy_status_basis"],
        "source_publication_accepted": source_quality["publication_accepted"],
        "source_publication_status": source_quality["publication_status"],
        "publication_accepted": selection["publication_accepted"],
        "publication_status": selection["publication_status"],
    }
    updated = list(points)
    updated[index] = replace(
        selected,
        energy_quality=assessed_quality,
        energy_quality_assessment=assessment,
    )
    return updated, assessment


def _input_quality(points: Sequence[LoadedPopulationPoint]) -> dict[str, Any]:
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
    warnings = [
        row
        for row in rows
        if row["publication_accepted"] and row["publication_status"] != "accepted"
    ]
    return {
        "status": (
            "unresolved" if unresolved else "accepted_with_warnings" if warnings else "accepted"
        ),
        "publication_accepted": not unresolved,
        "publication_accepted_statuses": [
            "accepted",
            "accepted_with_precision_warning",
            "accepted_with_retrospective_assessment",
        ],
        "points": rows,
        "unresolved_point_count": len(unresolved),
        "warning_point_count": len(warnings),
        "precision_warning_point_count": sum(
            row["publication_status"] == "accepted_with_precision_warning" for row in rows
        ),
        "retrospective_assessment_point_count": sum(
            row["publication_status"] == "accepted_with_retrospective_assessment" for row in rows
        ),
        "interpretation": (
            "Energy-specific method failures remain unresolved unless a manifest-bound "
            "simultaneous population-treatment or final-matrix assessment qualifies "
            "the exact source summaries. Raw run and energy statuses remain recorded "
            "for every input."
        ),
    }


def _validate_shared_identity(points: Sequence[LoadedPopulationPoint]) -> None:
    reference = points[0].identity
    mismatches = [
        (point.summary_path, _different_identity_fields(reference, point.identity))
        for point in points[1:]
        if point.identity != reference
    ]
    if mismatches:
        details = "; ".join(f"{path}: {', '.join(fields)}" for path, fields in mismatches)
        raise ValueError(
            "population summaries do not share exact case, guide, implementation, "
            f"propagator, population-control, initialization, and energy identity: {details}"
        )


def _group_by_timestep(
    points: Sequence[LoadedPopulationPoint],
) -> dict[float, list[LoadedPopulationPoint]]:
    groups: dict[float, list[LoadedPopulationPoint]] = defaultdict(list)
    for point in points:
        groups[point.dt].append(point)
    return {dt: sorted(group, key=lambda item: item.point.walkers) for dt, group in groups.items()}


def _resolve_selected_timestep(
    groups: Mapping[float, Sequence[LoadedPopulationPoint]],
    *,
    selected_dt: float | None,
) -> float:
    if selected_dt is None:
        if len(groups) != 1:
            raise ValueError("selected_dt is required when two timestep treatments are supplied")
        return next(iter(groups))
    if not math.isfinite(selected_dt) or selected_dt <= 0.0:
        raise ValueError("selected_dt must be finite and positive")
    matches = [dt for dt in groups if math.isclose(dt, selected_dt, rel_tol=0.0, abs_tol=1.0e-15)]
    if len(matches) != 1:
        raise ValueError("selected_dt must match exactly one supplied timestep treatment")
    return matches[0]


def _reference_population_point(
    assessment: PopulationLadderAssessment,
) -> PopulationEnergyPoint:
    matches = [
        point for point in assessment.points if point.walkers == assessment.reference_walkers
    ]
    if len(matches) != 1:
        raise ValueError("population ladder does not contain its reference walker point")
    return matches[0]


def _sampling_design(points: Sequence[LoadedPopulationPoint]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    trace_spacings: list[float] = []
    for point in points:
        store_every = _required_int(point.controls.get("store_every"), "store_every")
        if store_every <= 0:
            raise ValueError("store_every must be positive")
        trace_spacing_tau = point.dt * store_every
        trace_spacings.append(trace_spacing_tau)
        records.append(
            {
                "summary_path": str(point.summary_path),
                "dt": point.dt,
                "walkers": point.point.walkers,
                "store_every": store_every,
                "trace_spacing_tau": trace_spacing_tau,
            }
        )
    unique_spacings: list[float] = []
    for spacing in trace_spacings:
        if not any(
            math.isclose(spacing, existing, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for existing in unique_spacings
        ):
            unique_spacings.append(spacing)
    varied = len(unique_spacings) > 1
    return {
        "status": "varied_trace_cadence" if varied else "common_trace_cadence",
        "trace_cadence_varies_across_timesteps": varied,
        "points": records,
        "interpretation": (
            "trace cadence is an advisory observation-control difference; physical "
            "burn and production times are matched and each source supplies its own "
            "autocorrelation-aware conservative standard error"
        ),
    }


def _validate_population_group(
    points: Sequence[LoadedPopulationPoint],
    *,
    name: str,
) -> None:
    if len(points) not in {2, 3}:
        raise ValueError(f"{name} requires W/2W or W/2,W,2W")
    reference_controls = {
        key: value for key, value in points[0].controls.items() if key != "walkers"
    }
    for point in points[1:]:
        candidate = {key: value for key, value in point.controls.items() if key != "walkers"}
        if candidate != reference_controls:
            raise ValueError(f"{name} points may vary only walker count")


def _reference_doubling_pair(
    points: Sequence[LoadedPopulationPoint],
) -> tuple[LoadedPopulationPoint, LoadedPopulationPoint]:
    ordered = sorted(points, key=lambda item: item.point.walkers)
    return (ordered[0], ordered[1]) if len(ordered) == 2 else (ordered[1], ordered[2])


def _validate_cross_timestep_controls(
    fine: Sequence[LoadedPopulationPoint],
    coarse: Sequence[LoadedPopulationPoint],
) -> None:
    fine_reference = _reference_doubling_pair(fine)[0]
    coarse_reference = _reference_doubling_pair(coarse)[0]
    for field in ("burn_tau", "production_tau", "grid_extent", "n_bins"):
        if fine_reference.controls.get(field) != coarse_reference.controls.get(field):
            raise ValueError(f"timestep-population interaction disagrees on {field}")


def _validate_reporting_policy(
    *,
    reporting_resolution: float,
    confidence_level: float,
    fit_alpha: float,
) -> None:
    if not math.isclose(
        reporting_resolution,
        FIXED_ENERGY_REPORTING_RESOLUTION,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "population reporting resolution is fixed prospectively at 0.01 hbar*Omega"
        )
    for value, name in ((confidence_level, "confidence_level"), (fit_alpha, "fit_alpha")):
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie strictly between zero and one")


def _validate_reporting_unit(identity: Mapping[str, Any]) -> None:
    semantics = _required_mapping(identity, "energy_semantics")
    if (
        semantics.get("energy_unit") != FIXED_ENERGY_REPORTING_UNIT
        or semantics.get("report_energy_unit") != FIXED_ENERGY_REPORTING_UNIT
    ):
        raise ValueError(
            "population reporting resolution unit must match hbar*Omega input and report units"
        )


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
            raise ValueError(f"invalid {steps_field}/{tau_field}: {summary_path}")
        expected_steps = max(1, int(round(tau / dt)))
        if steps != expected_steps or (
            tau != 0.0
            and not math.isclose(
                steps * dt,
                tau,
                rel_tol=1.0e-12,
                abs_tol=0.5 * dt + 1.0e-15,
            )
        ):
            raise ValueError(f"{steps_field} * dt is inconsistent with {tau_field}: {summary_path}")


def _point_telemetry(stationarity: dict[str, Any]) -> dict[str, Any]:
    rows_value = stationarity.get("seed_summaries")
    rows = (
        [row for row in rows_value if isinstance(row, dict)] if isinstance(rows_value, list) else []
    )

    def finite_values(field: str) -> list[float]:
        return [
            float(value)
            for row in rows
            if isinstance((value := row.get(field)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]

    def mean(field: str) -> float | None:
        values = finite_values(field)
        return None if not values else float(math.fsum(values) / len(values))

    def maximum(field: str) -> float | None:
        values = finite_values(field)
        return None if not values else max(values)

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


def _qualified_systematics(
    *,
    publication_ready: bool,
    classification: str,
    interaction: TimeStepPopulationInteraction | None,
) -> list[str]:
    if not publication_ready:
        return []
    qualified = [
        (
            "inverse_population_limit_bounded_below_reporting_resolution"
            if classification == "accepted_population_limit"
            else "last_population_doubling_bounded_below_reporting_resolution"
        )
    ]
    if interaction is not None and interaction.bounded_below_reporting_resolution:
        qualified.append("timestep_population_interaction_bounded_below_reporting_resolution")
    return qualified


def _write_point_table(
    output_dir: Path,
    points: Sequence[LoadedPopulationPoint],
) -> Path:
    path = output_dir / "point_table.csv"
    fields = (
        "dt",
        "walkers",
        "energy",
        "conservative_stderr",
        "case_id",
        "run_name",
        "result_schema_version",
        "run_id",
        "bundle_sha256",
        "run_status",
        "energy_status",
        "energy_publication_accepted",
        "energy_publication_status",
        "energy_status_basis",
        "source_energy_publication_accepted",
        "source_energy_publication_status",
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
            row = point.to_dict()
            row["seed_count"] = len(point.point.seed_ids)
            row["seeds"] = ",".join(str(seed) for seed in point.point.seed_ids)
            row.update(
                {
                    "energy_publication_accepted": point.energy_quality.get("publication_accepted"),
                    "energy_publication_status": point.energy_quality.get("publication_status"),
                    "energy_status_basis": point.energy_quality.get("status_basis"),
                    "source_energy_publication_accepted": point.energy_quality.get(
                        "source_publication_accepted"
                    ),
                    "source_energy_publication_status": point.energy_quality.get(
                        "source_publication_status"
                    ),
                    **point.telemetry,
                }
            )
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _write_comparison_table(
    output_dir: Path,
    *,
    reference_fine_dt: float,
    reference_fine_analysis: PopulationLadderAssessment,
    coarse_dt: float | None,
    coarse_analysis: PopulationLadderAssessment | None,
    selected_dt: float,
    selected_dt_basis: str,
    interaction: TimeStepPopulationInteraction | None,
) -> Path:
    path = output_dir / "comparison_table.csv"
    fields = (
        "comparison",
        "treatment_role",
        "timestep",
        "reference_fine_dt",
        "coarse_dt",
        "selected_dt",
        "selected_dt_basis",
        "is_selected_treatment",
        "first_walkers",
        "second_walkers",
        "mean_difference",
        "paired_standard_error",
        "source_run_quadrature_standard_error",
        "worst_case_arbitrary_covariance_standard_error_envelope",
        "conservative_standard_error",
        "upper_allowance",
        "reporting_resolution",
        "bounded_below_reporting_resolution",
        "classification",
    )
    rows: list[dict[str, Any]] = []
    for role, dt, analysis in (
        ("reference_fine", reference_fine_dt, reference_fine_analysis),
        ("coarse", coarse_dt, coarse_analysis),
    ):
        if dt is None or analysis is None:
            continue
        common = {
            "treatment_role": role,
            "timestep": dt,
            "reference_fine_dt": reference_fine_dt,
            "coarse_dt": coarse_dt,
            "selected_dt": selected_dt,
            "selected_dt_basis": selected_dt_basis,
            "is_selected_treatment": dt == selected_dt,
            "classification": analysis.classification,
        }
        last = analysis.last_doubling
        rows.append(
            {
                **common,
                "comparison": f"{role}_last_doubling",
                "first_walkers": last.first_walkers,
                "second_walkers": last.second_walkers,
                "mean_difference": last.mean_difference,
                "paired_standard_error": last.paired_standard_error,
                "source_run_quadrature_standard_error": (last.source_run_quadrature_standard_error),
                "worst_case_arbitrary_covariance_standard_error_envelope": (
                    last.worst_case_arbitrary_covariance_standard_error_envelope
                ),
                "conservative_standard_error": last.conservative_standard_error,
                "upper_allowance": last.upper_allowance,
                "reporting_resolution": last.reporting_resolution,
                "bounded_below_reporting_resolution": (last.bounded_below_reporting_resolution),
            }
        )
        richardson = analysis.richardson_window
        if richardson is not None:
            rows.append(
                {
                    **common,
                    "comparison": f"{role}_richardson_window",
                    "mean_difference": richardson.mean_difference,
                    "paired_standard_error": richardson.paired_standard_error,
                    "source_run_quadrature_standard_error": (
                        richardson.source_run_quadrature_standard_error
                    ),
                    "worst_case_arbitrary_covariance_standard_error_envelope": (
                        richardson.worst_case_arbitrary_covariance_standard_error_envelope
                    ),
                    "conservative_standard_error": richardson.conservative_standard_error,
                    "upper_allowance": richardson.upper_allowance,
                    "reporting_resolution": richardson.reporting_resolution,
                    "bounded_below_reporting_resolution": (
                        richardson.bounded_below_reporting_resolution
                    ),
                }
            )
    if interaction is not None:
        rows.append(
            {
                "comparison": "timestep_population_interaction",
                "treatment_role": "cross_timestep",
                "reference_fine_dt": reference_fine_dt,
                "coarse_dt": coarse_dt,
                "selected_dt": selected_dt,
                "selected_dt_basis": selected_dt_basis,
                "is_selected_treatment": "",
                "first_walkers": interaction.fine_timestep_difference.first_walkers,
                "second_walkers": interaction.fine_timestep_difference.second_walkers,
                "mean_difference": interaction.interaction_difference,
                "paired_standard_error": (interaction.interaction_statistical_standard_error),
                "source_run_quadrature_standard_error": (
                    interaction.interaction_source_run_quadrature_standard_error
                ),
                "worst_case_arbitrary_covariance_standard_error_envelope": (
                    interaction.interaction_worst_case_arbitrary_covariance_standard_error_envelope
                ),
                "conservative_standard_error": interaction.interaction_standard_error,
                "upper_allowance": interaction.upper_allowance,
                "reporting_resolution": interaction.reporting_resolution,
                "bounded_below_reporting_resolution": (
                    interaction.bounded_below_reporting_resolution
                ),
                "classification": (
                    "accepted"
                    if interaction.bounded_below_reporting_resolution
                    else "timestep_population_interaction_unresolved"
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _validate_output_separation(
    output_dir: Path,
    source_artifact_paths: Sequence[Path],
) -> None:
    for source_path in source_artifact_paths:
        run_dir = source_path.parent
        if (
            output_dir == run_dir
            or output_dir.is_relative_to(run_dir)
            or run_dir.is_relative_to(output_dir)
        ):
            raise ValueError(
                f"output_dir must not overlap an input artifact or run directory: {output_dir}"
            )


def _seed_list(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list")
    return tuple(_required_int(seed, description) for seed in value)


def _seed_ids_from_rows(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{description} must contain seed mappings")
    return tuple(_required_int(row.get("seed"), description) for row in value)


def _different_identity_fields(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    return sorted(
        key for key in set(reference) | set(candidate) if reference.get(key) != candidate.get(key)
    )


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a mapping: {path}")
    return payload


def _required_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _required_positive_float(value: Any, description: str) -> float:
    result = _required_float(value, description)
    if result <= 0.0:
        raise ValueError(f"{description} must be positive")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _required_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{description} must be an integer")
    return int(value)
