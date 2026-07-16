from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.analysis.proposal_telemetry import summarize_seed_proposal_telemetry
from hrdmc.artifacts import (
    build_run_provenance,
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    write_json,
    write_run_manifest,
)
from hrdmc.theory import (
    trapped_tg_energy_total,
    trapped_tg_r2_radius,
    trapped_tg_rms_radius,
)
from hrdmc.workflows.dmc.benchmark_packet.matrix_assembly import (
    FINAL_MATRIX_ASSEMBLY_RUN_NAME,
    FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION,
    REQUIRED_CASE_ORDER,
    load_final_matrix_energy_selection,
)
from hrdmc.workflows.dmc.fw_sensitivity import (
    ACCEPTED_FW_STATUSES,
    FW_SENSITIVITY_RUN_NAME,
    FW_SENSITIVITY_SCHEMA_VERSION,
    AnchorSources,
    build_fw_sampling_design,
    load_final_matrix_anchor_sources,
    load_manifest_bound_benchmark_packet,
)
from hrdmc.workflows.dmc.population_systematics import (
    POPULATION_SYSTEMATICS_RUN_NAME,
    POPULATION_SYSTEMATICS_SCHEMA_VERSION,
)
from hrdmc.workflows.dmc.population_systematics import (
    PUBLICATION_READY_STATUSES as POPULATION_READY_STATUSES,
)
from hrdmc.workflows.dmc.timestep_extrapolation import (
    TIMESTEP_EXTRAPOLATION_RUN_NAME,
    TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
)
from hrdmc.workflows.dmc.trapped import parse_case

NUMERICAL_SYSTEMATICS_PACKAGE_SCHEMA_VERSION = "dmc_numerical_systematics_package_v1"
NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME = "dmc_numerical_systematics_package"
SYSTEMATIC_LANES = ("timestep", "population", "forward_walking")
FINITE_CASE_ORDER = tuple(case for case in REQUIRED_CASE_ORDER if not case.endswith("_A0"))
DIRECT_TIMESTEP_STATUSES = {"accepted", "accepted_with_warnings"}

_LANE_CONTRACTS = {
    "timestep": (
        TIMESTEP_EXTRAPOLATION_RUN_NAME,
        TIMESTEP_EXTRAPOLATION_SCHEMA_VERSION,
    ),
    "population": (
        POPULATION_SYSTEMATICS_RUN_NAME,
        POPULATION_SYSTEMATICS_SCHEMA_VERSION,
    ),
    "forward_walking": (
        FW_SENSITIVITY_RUN_NAME,
        FW_SENSITIVITY_SCHEMA_VERSION,
    ),
}

_ARTIFACT_NAMES = {
    "summary": "summary.json",
    "case_status_table": "case_status.csv",
    "thesis_energy_table": "thesis_energy_table.csv",
    "uncertainty_table": "uncertainty_components.csv",
    "source_table": "source_artifacts.csv",
}


@dataclass(frozen=True)
class BoundSystematicAssessment:
    lane: str
    case_id: str
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]

    def reference(self, *, reference_root: Path) -> dict[str, Any]:
        return _artifact_reference(
            self.manifest_path,
            self.summary_path,
            self.manifest,
            reference_root=reference_root,
        )


@dataclass(frozen=True)
class LoadedPackageInputs:
    final_manifest_path: Path
    final_manifest: dict[str, Any]
    final_summary_path: Path
    final_summary: dict[str, Any]
    anchors: dict[str, AnchorSources]
    timestep: dict[str, BoundSystematicAssessment]
    population: dict[str, BoundSystematicAssessment]
    forward_walking: dict[str, BoundSystematicAssessment]
    source_plot_warnings: tuple[str, ...]


def assemble_numerical_systematics_package(
    final_matrix_manifest: Path,
    *,
    timestep_manifests: Mapping[str, Path],
    population_manifests: Mapping[str, Path],
    fw_sensitivity_manifests: Mapping[str, Path],
    output_dir: Path,
    bounded_qualifiers: Mapping[tuple[str, str], str] | None = None,
    command: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Assemble the manifest-bound DMC numerical-systematics result package."""

    root = output_dir.resolve()
    qualifiers = _validate_bounded_qualifiers(bounded_qualifiers or {})
    _validate_input_case_maps(
        timestep=timestep_manifests,
        population=population_manifests,
        forward_walking=fw_sensitivity_manifests,
    )
    input_manifest_paths = [
        final_matrix_manifest.resolve(),
        *(path.resolve() for path in timestep_manifests.values()),
        *(path.resolve() for path in population_manifests.values()),
        *(path.resolve() for path in fw_sensitivity_manifests.values()),
    ]
    _validate_output_directory(root, input_manifest_paths=input_manifest_paths)
    loaded = _load_package_inputs(
        final_matrix_manifest.resolve(),
        timestep_manifests=timestep_manifests,
        population_manifests=population_manifests,
        fw_sensitivity_manifests=fw_sensitivity_manifests,
    )
    _validate_cross_lane_identities(loaded)
    source_config = _source_config(loaded, reference_root=root)
    payload = _build_payload(
        loaded,
        source_config=source_config,
        bounded_qualifiers=qualifiers,
    )

    ensure_dir(root)
    artifact_paths = {name: root / filename for name, filename in _ARTIFACT_NAMES.items()}
    write_json(artifact_paths["summary"], payload)
    _write_csv(artifact_paths["case_status_table"], _case_status_rows(payload))
    _write_csv(artifact_paths["thesis_energy_table"], _thesis_energy_rows(payload))
    _write_csv(artifact_paths["uncertainty_table"], _uncertainty_rows(payload))
    _write_csv(artifact_paths["source_table"], _source_rows(payload))

    config = {
        "case_order": list(REQUIRED_CASE_ORDER),
        "finite_case_order": list(FINITE_CASE_ORDER),
        "source_locator_base": "package_directory",
        "sources": source_config,
        "bounded_qualifiers": _qualifiers_to_payload(qualifiers),
    }
    manifest_path = write_run_manifest(
        root,
        run_name=NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME,
        config=config,
        artifacts=list(artifact_paths.values()),
        schema_version=NUMERICAL_SYSTEMATICS_PACKAGE_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=str(payload["status"]),
    )
    return payload, {**artifact_paths, "run_manifest": manifest_path, "output_dir": root}


def verify_numerical_systematics_package_manifest(path: Path) -> tuple[bool, list[str]]:
    """Rebuild a package from its exact inputs and verify all derived tables."""

    try:
        manifest_path = path.resolve()
        summary_path = manifest_path.parent / _ARTIFACT_NAMES["summary"]
        manifest, _ = load_manifest_bound_artifact(manifest_path, summary_path)
        summary = _load_mapping(summary_path, "numerical-systematics summary")
        _validate_package_manifest_header(manifest, summary)
        config = _required_mapping(manifest, "config")
        root = manifest_path.parent
        source_config = _required_mapping(config, "sources")
        qualifiers = _qualifiers_from_payload(config.get("bounded_qualifiers"))
        loaded = _load_inputs_from_source_config(root, source_config)
        _validate_cross_lane_identities(loaded)
        expected_sources = _source_config(loaded, reference_root=root)
        if expected_sources != source_config:
            raise ValueError("package source identities disagree with current artifacts")
        expected = _build_payload(
            loaded,
            source_config=source_config,
            bounded_qualifiers=qualifiers,
        )
        if summary != expected:
            raise ValueError("package summary disagrees with its exact source artifacts")
        if manifest.get("status") != expected.get("status"):
            raise ValueError("package manifest and derived status disagree")
        expected_tables = {
            "case_status_table": _case_status_rows(expected),
            "thesis_energy_table": _thesis_energy_rows(expected),
            "uncertainty_table": _uncertainty_rows(expected),
            "source_table": _source_rows(expected),
        }
        for name, rows in expected_tables.items():
            table_path = root / _ARTIFACT_NAMES[name]
            if table_path.read_text(encoding="utf-8") != _csv_text(rows):
                raise ValueError(f"{_ARTIFACT_NAMES[name]} disagrees with the package summary")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, [str(exc)]
    return True, []


def _load_package_inputs(
    final_matrix_manifest: Path,
    *,
    timestep_manifests: Mapping[str, Path],
    population_manifests: Mapping[str, Path],
    fw_sensitivity_manifests: Mapping[str, Path],
) -> LoadedPackageInputs:
    final_manifest_path = final_matrix_manifest.resolve()
    final_summary_path = final_manifest_path.parent / "final_matrix_summary.json"
    final_manifest, _ = load_manifest_bound_artifact(final_manifest_path, final_summary_path)
    final_summary = _load_mapping(final_summary_path, "final-matrix summary")
    if final_manifest.get("run_name") != FINAL_MATRIX_ASSEMBLY_RUN_NAME:
        raise ValueError("final-matrix input has the wrong artifact owner")
    if final_manifest.get("result_schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("final-matrix input has the wrong result schema")
    if final_manifest.get("status") != "accepted" or final_summary.get("status") != "accepted":
        raise ValueError("final-matrix input is not accepted")
    if final_summary.get("schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("final-matrix summary has the wrong result schema")
    if final_summary.get("case_order") != list(REQUIRED_CASE_ORDER):
        raise ValueError("final-matrix case order is not canonical")

    # One energy-selection load replays the complete matrix-wide stationarity
    # assessment and all primary-source bindings.  Per-case anchor loads below
    # then verify the density/R2 selections, including composed supplements.
    energy_selection = load_final_matrix_energy_selection(
        final_manifest_path,
        case_id=REQUIRED_CASE_ORDER[0],
    )
    anchors = {
        case_id: load_final_matrix_anchor_sources(final_manifest_path, case_id=case_id)
        for case_id in REQUIRED_CASE_ORDER
    }
    timestep = {
        case_id: _load_systematic_assessment(path, lane="timestep", case_id=case_id)
        for case_id, path in timestep_manifests.items()
    }
    population = {
        case_id: _load_systematic_assessment(path, lane="population", case_id=case_id)
        for case_id, path in population_manifests.items()
    }
    forward_walking = {
        case_id: _load_systematic_assessment(
            path,
            lane="forward_walking",
            case_id=case_id,
        )
        for case_id, path in fw_sensitivity_manifests.items()
    }
    return LoadedPackageInputs(
        final_manifest_path=final_manifest_path,
        final_manifest=final_manifest,
        final_summary_path=final_summary_path,
        final_summary=final_summary,
        anchors=anchors,
        timestep=timestep,
        population=population,
        forward_walking=forward_walking,
        source_plot_warnings=tuple(energy_selection.get("source_plot_artifact_warnings", [])),
    )


def _load_systematic_assessment(
    manifest_path: Path,
    *,
    lane: str,
    case_id: str,
) -> BoundSystematicAssessment:
    path = manifest_path.resolve()
    if path.name != "run_manifest.json":
        raise ValueError(f"{case_id} {lane}: input must be a run_manifest.json")
    summary_path = path.parent / "summary.json"
    manifest, _ = load_manifest_bound_artifact(path, summary_path)
    summary = _load_mapping(summary_path, f"{case_id} {lane} summary")
    expected_run_name, expected_schema = _LANE_CONTRACTS[lane]
    if manifest.get("run_name") != expected_run_name:
        raise ValueError(f"{case_id} {lane}: artifact owner is invalid")
    if manifest.get("result_schema_version") != expected_schema:
        raise ValueError(f"{case_id} {lane}: manifest result schema is invalid")
    if summary.get("schema_version") != expected_schema:
        raise ValueError(f"{case_id} {lane}: summary result schema is invalid")
    if manifest.get("status") != summary.get("status"):
        raise ValueError(f"{case_id} {lane}: manifest and summary statuses disagree")
    config = _required_mapping(manifest, "config")
    if config.get("case_id") != case_id or summary.get("case_id") != case_id:
        raise ValueError(f"{case_id} {lane}: case identity mismatch")
    if summary.get("identity") != config.get("identity"):
        raise ValueError(f"{case_id} {lane}: scientific identity disagrees with its manifest")
    if summary.get("identity_fingerprint") != config.get("identity_fingerprint"):
        raise ValueError(f"{case_id} {lane}: identity fingerprint disagrees")
    if lane == "timestep":
        _validate_timestep_result_semantics(summary, config, case_id=case_id)
    elif lane == "population":
        _validate_population_selected_treatment(summary, config, case_id=case_id)
    else:
        _validate_fw_result_semantics(summary, config, case_id=case_id)
    return BoundSystematicAssessment(
        lane=lane,
        case_id=case_id,
        manifest_path=path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
    )


def _validate_timestep_result_semantics(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    _timestep_input_records(summary, config, case_id=case_id)
    extrapolation = _required_mapping(summary, "extrapolation")
    candidate = _required_float(
        extrapolation.get("candidate_zero_step_energy"),
        "candidate zero-step energy",
    )
    statistical = _required_float(
        extrapolation.get("candidate_zero_step_energy_statistical_stderr"),
        "candidate zero-step statistical stderr",
    )
    status = str(summary.get("status", "unresolved"))
    publication_ready = (
        summary.get("publication_ready_within_fixed_population_timestep_scope") is True
    )
    if not publication_ready:
        return
    if summary.get("extrapolated_energy") != candidate:
        raise ValueError(f"{case_id} timestep: promoted zero-step energy disagrees")
    if summary.get("extrapolated_energy_statistical_stderr") != statistical:
        raise ValueError(f"{case_id} timestep: promoted statistical stderr disagrees")
    unresolved = summary.get("unresolved_reasons")
    if unresolved != []:
        raise ValueError(f"{case_id} timestep: publication-ready result remains unresolved")
    classification = summary.get("classification")
    if status in DIRECT_TIMESTEP_STATUSES:
        if classification != "model_consistent":
            raise ValueError(f"{case_id} timestep: direct acceptance is not model-consistent")
        return
    if status != "accepted_with_model_bound" or classification != "model_sensitive":
        raise ValueError(f"{case_id} timestep: non-direct publication status is invalid")
    practical = _required_mapping(summary, "practical_resolution_assessment")
    if practical.get("accepted_with_model_bound") is not True:
        raise ValueError(f"{case_id} timestep: model-bound acceptance is not established")
    if practical.get("failed_checks") != []:
        raise ValueError(f"{case_id} timestep: model-bound checks are unresolved")
    policy = _required_mapping(practical, "policy")
    resolution = _required_positive_float(policy.get("resolution"), "reporting resolution")
    model_order = _required_mapping(practical, "model_order")
    fit_window = _required_mapping(practical, "fit_window")
    model_allowance = _required_float(model_order.get("upper_allowance"), "model-order bound")
    window_allowance = _required_float(fit_window.get("upper_allowance"), "fit-window bound")
    if model_allowance > resolution or window_allowance > resolution:
        raise ValueError(f"{case_id} timestep: model-bound allowance exceeds its resolution")
    if summary.get("extrapolated_energy_model_order_upper_allowance") != model_allowance:
        raise ValueError(f"{case_id} timestep: promoted model-order bound disagrees")
    if summary.get("extrapolated_energy_fit_window_upper_allowance") != window_allowance:
        raise ValueError(f"{case_id} timestep: promoted fit-window bound disagrees")


def _validate_fw_result_semantics(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    _validate_fw_sampling_design(summary, config, case_id=case_id)
    if summary.get("publication_ready_within_fw_sensitivity_scope") is not True:
        return
    if summary.get("status") not in ACCEPTED_FW_STATUSES:
        raise ValueError(f"{case_id} forward_walking: publication status is invalid")
    if summary.get("unresolved_reasons") != []:
        raise ValueError(f"{case_id} forward_walking: publication-ready result is unresolved")
    checks = (
        (_required_mapping(summary, "input_quality").get("status") == "accepted"),
        (_required_mapping(summary, "density_grid").get("compatible") is True),
        (_required_mapping(summary, "plateau_assessment").get("resolved") is True),
        (_required_mapping(summary, "genealogy_assessment").get("supported") is True),
        (_required_mapping(summary, "observable_comparison").get("equivalent") is True),
    )
    if not all(checks):
        raise ValueError(f"{case_id} forward_walking: publication checks do not reproduce")
    treatments = _required_mapping(summary, "treatments")
    anchor = _required_mapping(treatments, "anchor_density")
    candidate = _required_mapping(treatments, "candidate")
    if anchor.get("dt") == candidate.get("dt") and anchor.get("walkers") == candidate.get(
        "walkers"
    ):
        raise ValueError(f"{case_id} forward_walking: candidate treatment equals the anchor")


def _validate_fw_sampling_design(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    design = _required_mapping(summary, "sampling_design")
    manifest_design = _required_mapping(config, "sampling_design")
    if design != manifest_design:
        raise ValueError(f"{case_id} forward_walking: sampling design disagrees with its manifest")
    if design.get("status") not in {"common_cadence", "varied_cadence"}:
        raise ValueError(f"{case_id} forward_walking: sampling cadence status is invalid")
    if design.get("phase_safe") is not True:
        raise ValueError(f"{case_id} forward_walking: sampling cadence is not phase safe")
    if design.get("source_phase") != "production_event_index_mod_stride_zero":
        raise ValueError(f"{case_id} forward_walking: source-window phase is invalid")
    phase_policy = design.get("phase_policy")
    if not isinstance(phase_policy, str) or not phase_policy:
        raise ValueError(f"{case_id} forward_walking: cadence phase policy is missing")
    scheduled = _required_mapping(design, "scheduled_collective_move_enabled")
    if set(scheduled) != {"anchor_density", "anchor_r2", "candidate"} or any(
        not isinstance(value, bool) for value in scheduled.values()
    ):
        raise ValueError(f"{case_id} forward_walking: scheduled-move cadence record is invalid")
    all_local = not any(scheduled.values())
    if design.get("all_treatments_use_ordinary_local_dmc") is not all_local:
        raise ValueError(f"{case_id} forward_walking: local-DMC cadence record is inconsistent")
    if design.get("status") == "varied_cadence" and not all_local:
        raise ValueError(
            f"{case_id} forward_walking: varied cadence is invalid with a scheduled move"
        )
    cadence_statuses: list[str] = []
    for observable in ("r2", "density"):
        comparison = _required_mapping(design, observable)
        observable_status = comparison.get("status")
        if observable_status not in {"common_cadence", "varied_cadence"}:
            raise ValueError(
                f"{case_id} forward_walking: {observable} cadence comparison is invalid"
            )
        cadence_statuses.append(str(observable_status))
        records: dict[str, Mapping[str, Any]] = {}
        for treatment in ("anchor", "candidate"):
            record = _required_mapping(comparison, treatment)
            records[treatment] = record
            _required_positive_int(record.get("stride_steps"), f"{observable} stride steps")
            _required_positive_float(
                record.get("physical_stride_tau"),
                f"{observable} physical stride",
            )
        expected_ratio = _required_positive_float(
            records["candidate"].get("physical_stride_tau"),
            f"{observable} candidate physical stride",
        ) / _required_positive_float(
            records["anchor"].get("physical_stride_tau"),
            f"{observable} anchor physical stride",
        )
        declared_ratio = _required_positive_float(
            comparison.get("candidate_to_anchor_physical_stride_ratio"),
            f"{observable} cadence ratio",
        )
        if not math.isclose(declared_ratio, expected_ratio, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(f"{case_id} forward_walking: {observable} cadence ratio disagrees")

    anchor_composition_common = design.get("anchor_r2_composition_common_cadence")
    if not isinstance(anchor_composition_common, bool):
        raise ValueError(f"{case_id} forward_walking: anchor cadence-composition record is invalid")
    expected_status = (
        "varied_cadence"
        if not anchor_composition_common or "varied_cadence" in cadence_statuses
        else "common_cadence"
    )
    if design.get("status") != expected_status:
        raise ValueError(f"{case_id} forward_walking: aggregate cadence status disagrees")


def _load_inputs_from_source_config(
    root: Path,
    source_config: Mapping[str, Any],
) -> LoadedPackageInputs:
    final_reference = _required_mapping(source_config, "final_matrix")
    final_path = _resolve_reference(root, final_reference.get("manifest_path"))
    lane_paths: dict[str, dict[str, Path]] = {}
    for lane in SYSTEMATIC_LANES:
        records = _required_mapping(source_config, lane)
        lane_paths[lane] = {
            case_id: _resolve_reference(
                root, _required_mapping(records, case_id).get("manifest_path")
            )
            for case_id in records
        }
    loaded = _load_package_inputs(
        final_path,
        timestep_manifests=lane_paths["timestep"],
        population_manifests=lane_paths["population"],
        fw_sensitivity_manifests=lane_paths["forward_walking"],
    )
    return loaded


def _validate_cross_lane_identities(loaded: LoadedPackageInputs) -> None:
    rows = _final_rows_by_case(loaded.final_summary)
    assembly_reference = _artifact_reference(
        loaded.final_manifest_path,
        loaded.final_summary_path,
        loaded.final_manifest,
        reference_root=loaded.final_manifest_path.parent,
    )
    for case_id in REQUIRED_CASE_ORDER:
        _validate_anchor_row(rows[case_id], loaded.anchors[case_id])
    for case_id in FINITE_CASE_ORDER:
        anchor = loaded.anchors[case_id]
        for assessment in (
            loaded.timestep.get(case_id),
            loaded.population.get(case_id),
            loaded.forward_walking.get(case_id),
        ):
            if assessment is not None:
                _validate_assessment_identity(anchor, assessment)
        timestep = loaded.timestep.get(case_id)
        if timestep is not None:
            _validate_timestep_contains_anchor(timestep, anchor)
        population = loaded.population.get(case_id)
        if timestep is not None and population is not None:
            _validate_selected_timestep_population_treatment(timestep, population)
        fw = loaded.forward_walking.get(case_id)
        if fw is not None:
            declared = _required_mapping(fw.manifest, "config").get("final_matrix_manifest")
            if not isinstance(declared, dict):
                raise ValueError(f"{case_id} forward_walking: final-matrix binding is missing")
            for key in ("run_id", "bundle_sha256"):
                if declared.get(key) != assembly_reference.get(key):
                    raise ValueError(
                        f"{case_id} forward_walking: final-matrix {key} binding disagrees"
                    )
            if declared.get("sha256") != file_sha256(loaded.final_manifest_path):
                raise ValueError(
                    f"{case_id} forward_walking: final-matrix manifest digest disagrees"
                )
            _validate_fw_proposal_telemetry(fw, anchor)
        if population is not None and fw is not None:
            _validate_selected_population_fw_treatment(population, fw)


def _validate_selected_timestep_population_treatment(
    timestep: BoundSystematicAssessment,
    population: BoundSystematicAssessment,
) -> None:
    case_id = timestep.case_id
    selected_dt = _required_positive_float(
        population.summary.get("selected_dt"),
        "selected population timestep",
    )
    selected_walkers = _required_positive_int(
        population.summary.get("selected_walkers"),
        "selected population walkers",
    )
    timestep_identity = _required_mapping(timestep.summary, "identity")
    if timestep_identity.get("walkers") != selected_walkers:
        raise ValueError(f"{case_id}: zero-step walker population differs from selected treatment")
    config = _required_mapping(timestep.manifest, "config")
    inputs = _timestep_input_records(timestep.summary, config, case_id=case_id)
    matches = [
        record
        for record in inputs
        if math.isclose(
            _required_positive_float(record.get("dt"), "timestep input dt"),
            selected_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{case_id}: selected population timestep is absent from the accepted "
            "timestep analysis window"
        )
    controls = _required_mapping(matches[0], "controls")
    if controls.get("walkers") != selected_walkers:
        raise ValueError(
            f"{case_id}: selected timestep input walkers differ from selected treatment"
        )


def _timestep_input_records(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    case_id: str,
) -> list[Mapping[str, Any]]:
    summary_inputs = summary.get("input_summaries")
    config_inputs = config.get("inputs")
    if (
        not isinstance(summary_inputs, list)
        or not all(isinstance(record, dict) for record in summary_inputs)
        or not isinstance(config_inputs, list)
        or not all(isinstance(record, dict) for record in config_inputs)
        or len(summary_inputs) != len(config_inputs)
        or not summary_inputs
    ):
        raise ValueError(f"{case_id} timestep: input declarations are invalid")
    identity_fields = (
        "summary_path",
        "summary_sha256",
        "manifest_path",
        "manifest_sha256",
        "run_id",
        "bundle_sha256",
        "dt",
    )
    for summary_record, config_record in zip(summary_inputs, config_inputs, strict=True):
        for field in identity_fields:
            if summary_record.get(field) != config_record.get(field):
                raise ValueError(f"{case_id} timestep: input {field} disagrees with its manifest")
    return summary_inputs


def _validate_population_selected_treatment(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    selected_dt = _required_positive_float(summary.get("selected_dt"), "selected_dt")
    reference_dt = _required_positive_float(
        summary.get("reference_fine_dt"),
        "reference_fine_dt",
    )
    coarse_value = summary.get("coarse_dt")
    coarse_dt = (
        None if coarse_value is None else _required_positive_float(coarse_value, "coarse_dt")
    )
    for field in (
        "reference_fine_dt",
        "coarse_dt",
        "selected_dt",
        "selected_dt_basis",
        "selected_treatment_role",
        "selected_walkers",
        "selected_walkers_basis",
    ):
        if summary.get(field) != config.get(field):
            raise ValueError(f"{case_id} population: {field} disagrees with its manifest")
    if coarse_dt is None:
        if (
            selected_dt != reference_dt
            or summary.get("selected_dt_basis") != "only_supplied_timestep"
        ):
            raise ValueError(f"{case_id} population: single-treatment selection is invalid")
        expected_role = "reference_fine"
    else:
        if not reference_dt < coarse_dt:
            raise ValueError(f"{case_id} population: reference/coarse timestep order is invalid")
        if summary.get("selected_dt_basis") != "explicit_selected_dt":
            raise ValueError(f"{case_id} population: two-treatment selection is not explicit")
        if selected_dt not in {reference_dt, coarse_dt}:
            raise ValueError(f"{case_id} population: selected timestep is not supplied")
        expected_role = "reference_fine" if selected_dt == reference_dt else "coarse"
    if summary.get("selected_treatment_role") != expected_role:
        raise ValueError(f"{case_id} population: selected treatment role is invalid")

    reference_ladder = _required_mapping(summary, "reference_fine_population_ladder")
    coarse_ladder_value = summary.get("coarse_population_ladder")
    if coarse_dt is None:
        if coarse_ladder_value is not None:
            raise ValueError(f"{case_id} population: unexpected coarse ladder")
        expected_selected = reference_ladder
    else:
        if not isinstance(coarse_ladder_value, dict):
            raise ValueError(f"{case_id} population: coarse ladder is missing")
        expected_selected = reference_ladder if selected_dt == reference_dt else coarse_ladder_value
    selected_ladder = _required_mapping(summary, "selected_population_ladder")
    if selected_ladder != expected_selected:
        raise ValueError(f"{case_id} population: selected ladder does not match selected_dt")
    if selected_ladder.get("classification") != summary.get("classification"):
        raise ValueError(f"{case_id} population: selected classification disagrees")
    selected_walkers = selected_ladder.get("reference_walkers")
    if summary.get("selected_walkers") != selected_walkers:
        raise ValueError(f"{case_id} population: selected walkers disagree with selected ladder")
    if summary.get("selected_walkers_basis") != (
        "reference_population_of_selected_timestep_ladder"
    ):
        raise ValueError(f"{case_id} population: selected walker basis is invalid")
    last_doubling = _required_mapping(selected_ladder, "last_doubling")
    selected_allowance = _required_float(
        last_doubling.get("upper_allowance"),
        "selected population upper allowance",
    )
    bounds = _required_mapping(summary, "population_bounds")
    if bounds.get("selected_last_doubling_upper_allowance") != selected_allowance:
        raise ValueError(f"{case_id} population: selected bound is not reproducible")

    inputs = summary.get("input_summaries")
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise ValueError(f"{case_id} population: input summaries are invalid")
    supplied_dts = {
        _required_positive_float(item.get("dt"), "population input dt") for item in inputs
    }
    expected_dts = {reference_dt} if coarse_dt is None else {reference_dt, coarse_dt}
    if supplied_dts != expected_dts:
        raise ValueError(f"{case_id} population: selected timestep is not bound to inputs")

    publication_ready = summary.get("publication_ready_within_population_systematic_scope") is True
    if not publication_ready:
        return
    if summary.get("selected_population_last_doubling_upper_allowance") != selected_allowance:
        raise ValueError(f"{case_id} population: promoted selected bound disagrees")
    interaction = summary.get("timestep_population_interaction")
    if (
        not isinstance(interaction, dict)
        or interaction.get("bounded_below_reporting_resolution") is not True
    ):
        raise ValueError(f"{case_id} population: accepted selection lacks bounded interaction")
    interaction_allowance = interaction.get("upper_allowance")
    if summary.get("timestep_population_interaction_upper_allowance") != interaction_allowance:
        raise ValueError(f"{case_id} population: promoted interaction bound disagrees")
    if bounds.get("timestep_population_interaction_upper_allowance") != interaction_allowance:
        raise ValueError(f"{case_id} population: interaction bound record disagrees")

    classification = summary.get("classification")
    if classification == "accepted_population_limit":
        inverse_fit = _required_mapping(selected_ladder, "inverse_population_fit")
        richardson = _required_mapping(selected_ladder, "richardson_window")
        correction = _required_mapping(selected_ladder, "population_limit_correction")
        if correction.get("reference_walkers") != selected_walkers:
            raise ValueError(f"{case_id} population: correction reference walkers disagree")
        if summary.get("population_limit_energy_at_selected_timestep") != inverse_fit.get(
            "intercept"
        ):
            raise ValueError(f"{case_id} population: promoted population-limit energy disagrees")
        if summary.get("population_limit_energy_statistical_stderr") != inverse_fit.get(
            "intercept_stderr"
        ):
            raise ValueError(f"{case_id} population: population-limit stderr disagrees")
        if summary.get("population_limit_model_window_upper_allowance") != richardson.get(
            "upper_allowance"
        ):
            raise ValueError(f"{case_id} population: population-limit window disagrees")
        correction_fields = {
            "population_limit_correction_at_selected_timestep": "value",
            "population_limit_correction_statistical_stderr": "conservative_standard_error",
            "population_limit_correction_matched_seed_standard_error": (
                "matched_seed_standard_error"
            ),
            "population_limit_correction_source_run_quadrature_standard_error": (
                "source_run_quadrature_standard_error"
            ),
            "population_limit_correction_worst_case_arbitrary_covariance_standard_error_envelope": (
                "worst_case_arbitrary_covariance_standard_error_envelope"
            ),
        }
        for field, correction_field in correction_fields.items():
            if summary.get(field) != correction.get(correction_field):
                raise ValueError(f"{case_id} population: promoted {field} disagrees")
        if summary.get("selected_energy_population_basis") != (
            "population_limit_at_selected_timestep"
        ):
            raise ValueError(f"{case_id} population: population-limit basis is invalid")
        if summary.get("population_limit_correction_basis") != (
            "E_infinity(selected_dt) - E(selected_dt, selected_walkers)"
        ):
            raise ValueError(f"{case_id} population: population correction basis is invalid")
        if summary.get("downstream_zero_timestep_population_scope") != (
            "apply_population_limit_correction_to_selected_finite_population_zero_timestep_energy"
        ):
            raise ValueError(f"{case_id} population: zero-timestep correction scope is invalid")
        points = selected_ladder.get("points")
        if not isinstance(points, list) or not all(isinstance(point, dict) for point in points):
            raise ValueError(f"{case_id} population: selected points are invalid")
        reference_points = [point for point in points if point.get("walkers") == selected_walkers]
        if len(reference_points) != 1:
            raise ValueError(f"{case_id} population: selected reference point is not unique")
        limit_energy = _required_float(inverse_fit.get("intercept"), "population-limit energy")
        reference_energy = _required_float(
            reference_points[0].get("energy"),
            "selected finite-population energy",
        )
        correction_value = _required_float(correction.get("value"), "population correction")
        if not math.isclose(
            limit_energy - reference_energy,
            correction_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{case_id} population: correction does not reproduce the limit")
    elif classification == "accepted_finite_population_bound":
        reference_walkers = selected_ladder.get("reference_walkers")
        points = selected_ladder.get("points")
        if not isinstance(points, list) or not all(isinstance(point, dict) for point in points):
            raise ValueError(f"{case_id} population: selected points are invalid")
        matches = [point for point in points if point.get("walkers") == reference_walkers]
        if len(matches) != 1:
            raise ValueError(f"{case_id} population: selected reference point is not unique")
        selected_point = matches[0]
        expected_fields = {
            "finite_population_energy_at_selected_timestep": selected_point.get("energy"),
            "finite_population_energy_statistical_stderr": selected_point.get(
                "conservative_stderr"
            ),
            "finite_population_walkers": reference_walkers,
        }
        for field, expected in expected_fields.items():
            if summary.get(field) != expected:
                raise ValueError(f"{case_id} population: promoted {field} disagrees")
        if summary.get("selected_energy_population_basis") != (
            "finite_population_at_selected_walkers"
        ):
            raise ValueError(f"{case_id} population: finite-population basis is invalid")
        if summary.get("finite_population_w_to_2w_upper_allowance") != selected_allowance:
            raise ValueError(f"{case_id} population: finite-population W-to-2W bound disagrees")
        if summary.get("downstream_zero_timestep_population_scope") != (
            "retain_selected_finite_population_zero_timestep_energy; "
            "no infinite-population central value is claimed"
        ):
            raise ValueError(f"{case_id} population: finite-population scope is invalid")
    else:
        raise ValueError(f"{case_id} population: publication status has invalid classification")


def _validate_selected_population_fw_treatment(
    population: BoundSystematicAssessment,
    fw: BoundSystematicAssessment,
) -> None:
    case_id = population.case_id
    selected_dt = _required_positive_float(population.summary.get("selected_dt"), "selected_dt")
    treatments = _required_mapping(fw.summary, "treatments")
    candidate = _required_mapping(treatments, "candidate")
    candidate_dt = _required_positive_float(candidate.get("dt"), "FW candidate dt")
    if not math.isclose(selected_dt, candidate_dt, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(
            f"{case_id}: FW candidate timestep differs from selected population treatment"
        )
    selected_walkers = population.summary.get("selected_walkers")
    if candidate.get("walkers") != selected_walkers:
        raise ValueError(
            f"{case_id}: FW candidate walkers differ from selected population treatment"
        )


def _validate_fw_proposal_telemetry(
    fw: BoundSystematicAssessment,
    anchor: AnchorSources,
) -> None:
    treatments = _required_mapping(fw.summary, "treatments")
    anchor_treatment = _required_mapping(treatments, "anchor_density")
    candidate_treatment = _required_mapping(treatments, "candidate")
    anchor_telemetry = _required_mapping(anchor_treatment, "proposal_telemetry")
    candidate_telemetry = _required_mapping(candidate_treatment, "proposal_telemetry")
    expected_anchor = _proposal_telemetry(anchor.density.summary)
    if anchor_telemetry != expected_anchor:
        raise ValueError(f"{fw.case_id} forward_walking: anchor proposal telemetry disagrees")
    config = _required_mapping(fw.manifest, "config")
    candidate_reference = _required_mapping(config, "candidate")
    candidate_summary_path = Path(str(candidate_reference.get("summary_path", ""))).resolve()
    candidate_packet = load_manifest_bound_benchmark_packet(candidate_summary_path)
    exact_candidate_reference = candidate_packet.reference()
    for field in (
        "summary_sha256",
        "manifest_sha256",
        "run_id",
        "bundle_sha256",
        "source_tree_sha256",
        "dt",
        "walkers",
    ):
        if candidate_reference.get(field) != exact_candidate_reference.get(field):
            raise ValueError(f"{fw.case_id} forward_walking: candidate {field} binding disagrees")
    expected_candidate = _proposal_telemetry(candidate_packet.summary)
    if candidate_telemetry != expected_candidate:
        raise ValueError(f"{fw.case_id} forward_walking: candidate proposal telemetry disagrees")
    expected_sampling_design = build_fw_sampling_design(anchor, candidate_packet)
    if fw.summary.get("sampling_design") != expected_sampling_design:
        raise ValueError(
            f"{fw.case_id} forward_walking: sampling design disagrees with bound treatments"
        )
    for label, telemetry in (
        ("anchor", anchor_telemetry),
        ("candidate", candidate_telemetry),
    ):
        if telemetry.get("status") != "available":
            raise ValueError(
                f"{fw.case_id} forward_walking: {label} proposal telemetry is unavailable"
            )


def _validate_anchor_row(row: Mapping[str, Any], anchor: AnchorSources) -> None:
    case_id = str(row.get("case"))
    primary_estimates = _required_mapping(anchor.density.summary, "estimates")
    r2_estimates = _required_mapping(anchor.r2.summary, "estimates")
    energy = _required_mapping(primary_estimates, "energy")
    r2 = _required_mapping(r2_estimates, "r2")
    rms = _required_mapping(r2_estimates, "rms")
    expected = {
        "energy": energy.get("value"),
        "energy_stderr": energy.get("stderr"),
        "r2": r2.get("value"),
        "r2_stderr": r2.get("stderr"),
        "rms_radius": rms.get("value"),
        "rms_mc_statistical_stderr": rms.get("mc_statistical_stderr"),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{case_id}: final-matrix {field} disagrees with its source")


def _validate_assessment_identity(
    anchor: AnchorSources,
    assessment: BoundSystematicAssessment,
) -> None:
    identity = _required_mapping(assessment.summary, "identity")
    case_id = assessment.case_id
    if assessment.lane == "forward_walking":
        source_tree = identity.get("source_tree_sha256")
    else:
        source_tree = identity.get("implementation_source_tree_sha256")
    if source_tree != anchor.density.source_tree_sha256:
        raise ValueError(f"{case_id} {assessment.lane}: implementation identity differs")
    if identity.get("guide_family") != anchor.density.summary.get("guide_family"):
        raise ValueError(f"{case_id} {assessment.lane}: guide family differs")
    anchor_guide = _guide_binding(_required_mapping(anchor.density.summary, "guide_parameters"))
    assessment_guide = _guide_binding(_required_mapping(identity, "guide_parameters"))
    if assessment_guide != anchor_guide:
        raise ValueError(f"{case_id} {assessment.lane}: guide parameter identity differs")
    case = parse_case(case_id)
    if identity.get("n_particles") not in (None, case.n_particles):
        raise ValueError(f"{case_id} {assessment.lane}: particle number differs")
    if identity.get("rod_length_ho") not in (None, case.rod_length_ho):
        raise ValueError(f"{case_id} {assessment.lane}: rod length differs")
    controls = anchor.density.controls
    for field in ("local_step_method", "drift_limiter"):
        if identity.get(field) not in (None, controls.get(field)):
            raise ValueError(f"{case_id} {assessment.lane}: {field} differs")


def _validate_timestep_contains_anchor(
    assessment: BoundSystematicAssessment,
    anchor: AnchorSources,
) -> None:
    anchor_reference = anchor.density.reference()
    inputs = assessment.summary.get("input_summaries")
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise ValueError(f"{assessment.case_id} timestep: input summaries are invalid")
    matches = [
        item
        for item in inputs
        if item.get("summary_sha256") == anchor_reference.get("summary_sha256")
        and item.get("manifest_sha256") == anchor_reference.get("manifest_sha256")
        and item.get("run_id") == anchor_reference.get("run_id")
        and item.get("bundle_sha256") == anchor_reference.get("bundle_sha256")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{assessment.case_id} timestep: exact fine-time-step matrix anchor is not bound"
        )


def _build_payload(
    loaded: LoadedPackageInputs,
    *,
    source_config: Mapping[str, Any],
    bounded_qualifiers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    rows_by_case = _final_rows_by_case(loaded.final_summary)
    rows: list[dict[str, Any]] = []
    used_qualifiers: set[tuple[str, str]] = set()
    for case_id in REQUIRED_CASE_ORDER:
        case = parse_case(case_id)
        raw = rows_by_case[case_id]
        anchor = loaded.anchors[case_id]
        telemetry = _proposal_telemetry(anchor.density.summary)
        if case.rod_length_ho == 0.0:
            row = _exact_tg_row(case_id, raw=raw, anchor=anchor, telemetry=telemetry)
        else:
            row, row_qualifiers = _finite_case_row(
                case_id,
                raw=raw,
                anchor=anchor,
                timestep=loaded.timestep.get(case_id),
                population=loaded.population.get(case_id),
                fw_sensitivity=loaded.forward_walking.get(case_id),
                telemetry=telemetry,
                bounded_qualifiers=bounded_qualifiers,
            )
            used_qualifiers.update(row_qualifiers)
        if row["accepted_fw_sources"] is not None:
            row["accepted_fw_sources"] = {
                "density": _required_mapping(
                    _required_mapping(source_config, "raw_density"),
                    case_id,
                ),
                "r2": _required_mapping(
                    _required_mapping(source_config, "raw_r2"),
                    case_id,
                ),
            }
        rows.append(row)
    unused = set(bounded_qualifiers) - used_qualifiers
    if unused:
        formatted = ", ".join(f"{case_id}:{lane}" for case_id, lane in sorted(unused))
        raise ValueError(f"bounded qualifiers are unused or ineligible: {formatted}")

    missing = {
        lane: [case for case in FINITE_CASE_ORDER if case not in getattr(loaded, lane)]
        for lane in ("timestep", "population")
    }
    missing["forward_walking"] = [
        case for case in FINITE_CASE_ORDER if case not in loaded.forward_walking
    ]
    unresolved = [row["case"] for row in rows if not row["publication_ready"]]
    status = "publication_ready" if not unresolved else "systematics_incomplete"
    return {
        "schema_version": NUMERICAL_SYSTEMATICS_PACKAGE_SCHEMA_VERSION,
        "status": status,
        "case_order": list(REQUIRED_CASE_ORDER),
        "finite_case_order": list(FINITE_CASE_ORDER),
        "source_locator_base": "package_directory",
        "sources": source_config,
        "source_plot_artifact_warnings": list(loaded.source_plot_warnings),
        "bounded_qualifiers": _qualifiers_to_payload(bounded_qualifiers),
        "missing_inputs": missing,
        "unresolved_cases": unresolved,
        "publication_ready_case_count": sum(row["publication_ready"] for row in rows),
        "rows": rows,
        "thesis_energy_rows": [row for row in rows if row["publication_ready"]],
        "uncertainty_component_policy": (
            "Statistical, timestep model-order, timestep fit-window, walker-population, "
            "population-limit correction, population model-window, timestep-population "
            "interaction, forward-walking lag, and forward-walking treatment-sensitivity "
            "components are reported separately and are not combined in quadrature. "
            "Cross-lane covariance between the zero-step fit and population correction "
            "has not been estimated."
        ),
        "energy_population_scope": (
            "A zero-time-step fit at selected finite W is E(0,W), not E(0,infinity). "
            "An infinite-population central value is formed only when a manifest-bound "
            "selected-treatment population correction is available; that correction's "
            "statistical and model-window terms remain separate."
        ),
        "density_extrapolation_policy": (
            "No binwise zero-timestep or infinite-population density extrapolation is "
            "performed; accepted fine-treatment FW density/R2 sources remain the "
            "reported coordinate estimators after sensitivity qualification."
        ),
        "artifacts": dict(_ARTIFACT_NAMES),
    }


def _exact_tg_row(
    case_id: str,
    *,
    raw: Mapping[str, Any],
    anchor: AnchorSources,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    case = parse_case(case_id)
    exact_energy = trapped_tg_energy_total(case.n_particles, case.omega)
    exact_r2 = trapped_tg_r2_radius(case.n_particles, case.omega)
    exact_rms = trapped_tg_rms_radius(case.n_particles, case.omega)
    raw_energy = _optional_float(raw.get("energy"))
    r2 = _optional_float(raw.get("r2"))
    rms = _optional_float(raw.get("rms_radius"))
    energy_exact = raw_energy is not None and math.isclose(
        raw_energy,
        exact_energy,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )
    source_accepted = all(
        raw.get(field) == "accepted"
        for field in ("status", "energy_status", "r2_status", "density_status")
    )
    publication_ready = source_accepted and energy_exact
    return {
        "case": case_id,
        "n_particles": case.n_particles,
        "rod_length_ho": case.rod_length_ho,
        "status": "accepted_exact_tg_anchor" if publication_ready else "exact_tg_unresolved",
        "publication_ready": publication_ready,
        "energy_result_status": "exact_tg_anchor" if publication_ready else "unresolved",
        "raw_finite_dt_energy": raw_energy,
        "raw_finite_dt_energy_stderr": _optional_float(raw.get("energy_stderr")),
        "raw_dt": _optional_float(raw.get("dt")),
        "raw_walkers": raw.get("walkers"),
        "candidate_zero_timestep_energy_at_selected_walkers": None,
        "accepted_zero_timestep_energy_at_selected_walkers": None,
        "bounded_zero_timestep_energy_at_selected_walkers": None,
        "population_limit_correction_at_selected_timestep": None,
        "zero_timestep_population_limit_energy": None,
        "thesis_energy": exact_energy if publication_ready else None,
        "energy_lda": _optional_float(raw.get("energy_lda")),
        "energy_relative_delta_vs_lda": _relative_delta(
            exact_energy if publication_ready else None,
            _optional_float(raw.get("energy_lda")),
        ),
        "exact_tg": {
            "status": "accepted" if publication_ready else "unresolved",
            "energy": exact_energy,
            "raw_energy_absolute_error": (
                None if raw_energy is None else abs(raw_energy - exact_energy)
            ),
            "r2": exact_r2,
            "raw_fw_r2": r2,
            "raw_fw_r2_relative_error": _relative_error(r2, exact_r2),
            "rms_radius": exact_rms,
            "raw_fw_rms_radius": rms,
            "raw_fw_rms_relative_error": _relative_error(rms, exact_rms),
            "reference_scope": "zero-length hard rods in a harmonic trap",
        },
        "lane_status": {
            "final_matrix": str(raw.get("status")),
            "timestep": "not_required_exact_tg",
            "population": "not_required_exact_tg",
            "forward_walking": "accepted_exact_tg_anchor",
        },
        "unresolved_reasons": [] if publication_ready else ["exact_tg_anchor_unresolved"],
        "bounded_qualifiers": {},
        "selected_population_treatment": None,
        "population_energy_semantics": {"status": "exact_tg_anchor"},
        "uncertainty_components": _uncertainty_components(
            raw,
            timestep=None,
            population=None,
            fw_sensitivity=None,
        ),
        "proposal_telemetry": _treatment_proposal_telemetry(telemetry, None),
        "accepted_fw_sources": {
            "density": _packet_reference(anchor.density),
            "r2": _packet_reference(anchor.r2),
        }
        if publication_ready
        else None,
    }


def _finite_case_row(
    case_id: str,
    *,
    raw: Mapping[str, Any],
    anchor: AnchorSources,
    timestep: BoundSystematicAssessment | None,
    population: BoundSystematicAssessment | None,
    fw_sensitivity: BoundSystematicAssessment | None,
    telemetry: dict[str, Any],
    bounded_qualifiers: Mapping[tuple[str, str], str],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    used: set[tuple[str, str]] = set()
    lane_results: dict[str, dict[str, Any]] = {}
    for lane, assessment in (
        ("timestep", timestep),
        ("population", population),
        ("forward_walking", fw_sensitivity),
    ):
        result = _lane_result(
            lane,
            assessment,
            qualifier=bounded_qualifiers.get((case_id, lane)),
        )
        lane_results[lane] = result
        if result["qualifier_used"]:
            used.add((case_id, lane))

    final_accepted = all(
        raw.get(field) == "accepted"
        for field in ("status", "energy_status", "r2_status", "density_status")
    )
    treatment_telemetry = _treatment_proposal_telemetry(telemetry, fw_sensitivity)
    telemetry_ready = treatment_telemetry["status"] == "available"
    population_energy = _population_energy_semantics(population)
    population_energy_ready = population_energy["status"] in {
        "selected_finite_population",
        "population_limit_correction",
    }
    publication_ready = (
        final_accepted
        and telemetry_ready
        and population_energy_ready
        and all(result["publication_ready"] for result in lane_results.values())
    )
    unresolved = []
    if not final_accepted:
        unresolved.append("final_matrix_row_unresolved")
    if not telemetry_ready:
        unresolved.append("selected_treatment_proposal_telemetry_unavailable")
    if not population_energy_ready:
        unresolved.append(str(population_energy["status"]))
    unresolved.extend(
        f"{lane}:{reason}"
        for lane, result in lane_results.items()
        for reason in result["unresolved_reasons"]
    )

    ts_summary = None if timestep is None else timestep.summary
    candidate_energy = _candidate_zero_step_energy(ts_summary)
    ts_disposition = lane_results["timestep"]["disposition"]
    accepted_energy = candidate_energy if ts_disposition == "accepted" else None
    bounded_energy = candidate_energy if ts_disposition == "bounded_qualified" else None
    selected_finite_zero_step = accepted_energy if accepted_energy is not None else bounded_energy
    population_correction = _optional_float(population_energy.get("correction"))
    zero_step_population_limit = (
        None
        if selected_finite_zero_step is None or population_correction is None
        else selected_finite_zero_step + population_correction
    )
    thesis_energy = (
        zero_step_population_limit
        if population_energy["status"] == "population_limit_correction"
        else selected_finite_zero_step
    )
    if not publication_ready:
        thesis_energy = None
    raw_lda = _optional_float(raw.get("energy_lda"))
    case = parse_case(case_id)
    status = (
        "accepted_with_bounded_qualifier"
        if publication_ready and any(result["qualifier_used"] for result in lane_results.values())
        else "accepted"
        if publication_ready
        else "systematics_incomplete"
    )
    population_limit = population_energy["status"] == "population_limit_correction"
    energy_status = (
        (
            "accepted_zero_timestep_population_limit"
            if population_limit
            else "accepted_zero_timestep_at_selected_finite_population"
        )
        if publication_ready and accepted_energy is not None
        else (
            "bounded_model_sensitive_zero_timestep_population_limit"
            if population_limit
            else "bounded_model_sensitive_zero_timestep_at_selected_finite_population"
        )
        if publication_ready and bounded_energy is not None
        else "candidate_zero_timestep_estimate_unresolved"
        if candidate_energy is not None
        else "missing_zero_timestep_estimate"
    )
    return (
        {
            "case": case_id,
            "n_particles": case.n_particles,
            "rod_length_ho": case.rod_length_ho,
            "status": status,
            "publication_ready": publication_ready,
            "energy_result_status": energy_status,
            "raw_finite_dt_energy": _optional_float(raw.get("energy")),
            "raw_finite_dt_energy_stderr": _optional_float(raw.get("energy_stderr")),
            "raw_dt": _optional_float(raw.get("dt")),
            "raw_walkers": raw.get("walkers"),
            "candidate_zero_timestep_energy_at_selected_walkers": candidate_energy,
            "accepted_zero_timestep_energy_at_selected_walkers": accepted_energy,
            "bounded_zero_timestep_energy_at_selected_walkers": bounded_energy,
            "population_limit_correction_at_selected_timestep": population_correction,
            "zero_timestep_population_limit_energy": zero_step_population_limit,
            "thesis_energy": thesis_energy,
            "energy_lda": raw_lda,
            "energy_relative_delta_vs_lda": _relative_delta(thesis_energy, raw_lda),
            "exact_tg": None,
            "lane_status": {
                "final_matrix": str(raw.get("status")),
                **{lane: result["disposition"] for lane, result in lane_results.items()},
            },
            "source_lane_status": {
                lane: result["source_status"] for lane, result in lane_results.items()
            },
            "selected_population_treatment": _selected_population_treatment(population),
            "population_energy_semantics": population_energy,
            "unresolved_reasons": unresolved,
            "bounded_qualifiers": {
                lane: result["qualifier"]
                for lane, result in lane_results.items()
                if result["qualifier_used"]
            },
            "uncertainty_components": _uncertainty_components(
                raw,
                timestep=timestep,
                population=population,
                fw_sensitivity=fw_sensitivity,
            ),
            "proposal_telemetry": treatment_telemetry,
            "accepted_fw_sources": {
                "density": _packet_reference(anchor.density),
                "r2": _packet_reference(anchor.r2),
            }
            if publication_ready
            else None,
        },
        used,
    )


def _lane_result(
    lane: str,
    assessment: BoundSystematicAssessment | None,
    *,
    qualifier: str | None,
) -> dict[str, Any]:
    if assessment is None:
        return {
            "source_status": "missing",
            "disposition": "missing",
            "publication_ready": False,
            "qualifier": qualifier,
            "qualifier_used": False,
            "unresolved_reasons": ["input_missing"],
        }
    summary = assessment.summary
    source_status = str(summary.get("status", "unresolved"))
    publication_flag = _lane_publication_flag(lane, summary)
    direct = _lane_directly_accepted(lane, source_status) and publication_flag
    bounded = _lane_bounded_eligible(lane, summary)
    if direct:
        return {
            "source_status": source_status,
            "disposition": "accepted",
            "publication_ready": True,
            "qualifier": qualifier,
            "qualifier_used": False,
            "unresolved_reasons": [],
        }
    if bounded and qualifier is not None:
        return {
            "source_status": source_status,
            "disposition": "bounded_qualified",
            "publication_ready": True,
            "qualifier": qualifier,
            "qualifier_used": True,
            "unresolved_reasons": [],
        }
    reasons = summary.get("unresolved_reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        reasons = [source_status]
    if bounded and qualifier is None:
        reasons = [*reasons, "explicit_bounded_thesis_qualifier_required"]
    return {
        "source_status": source_status,
        "disposition": "bounded_qualifier_required" if bounded else "unresolved",
        "publication_ready": False,
        "qualifier": qualifier,
        "qualifier_used": False,
        "unresolved_reasons": list(dict.fromkeys(reasons)),
    }


def _lane_publication_flag(lane: str, summary: Mapping[str, Any]) -> bool:
    field = {
        "timestep": "publication_ready_within_fixed_population_timestep_scope",
        "population": "publication_ready_within_population_systematic_scope",
        "forward_walking": "publication_ready_within_fw_sensitivity_scope",
    }[lane]
    return summary.get(field) is True


def _lane_directly_accepted(lane: str, status: str) -> bool:
    if lane == "timestep":
        return status in DIRECT_TIMESTEP_STATUSES
    if lane == "population":
        return status in POPULATION_READY_STATUSES
    return status in ACCEPTED_FW_STATUSES


def _lane_bounded_eligible(lane: str, summary: Mapping[str, Any]) -> bool:
    if lane == "timestep":
        assessment = summary.get("practical_resolution_assessment")
        return isinstance(assessment, dict) and assessment.get("accepted_with_model_bound") is True
    return _lane_publication_flag(lane, summary)


def _uncertainty_components(
    raw: Mapping[str, Any],
    *,
    timestep: BoundSystematicAssessment | None,
    population: BoundSystematicAssessment | None,
    fw_sensitivity: BoundSystematicAssessment | None,
) -> dict[str, Any]:
    ts = {} if timestep is None else timestep.summary
    pop = {} if population is None else population.summary
    fw = {} if fw_sensitivity is None else fw_sensitivity.summary
    practical = ts.get("practical_resolution_assessment")
    practical = practical if isinstance(practical, dict) else {}
    model_order = practical.get("model_order")
    model_order = model_order if isinstance(model_order, dict) else {}
    fit_window = practical.get("fit_window")
    fit_window = fit_window if isinstance(fit_window, dict) else {}
    comparison = fw.get("observable_comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    r2_sensitivity = comparison.get("r2")
    r2_sensitivity = r2_sensitivity if isinstance(r2_sensitivity, dict) else {}
    rms_sensitivity = comparison.get("rms_radius")
    rms_sensitivity = rms_sensitivity if isinstance(rms_sensitivity, dict) else {}
    density_sensitivity = comparison.get("density")
    density_sensitivity = density_sensitivity if isinstance(density_sensitivity, dict) else {}
    return {
        "energy_statistical_stderr": _optional_float(
            ts.get(
                "extrapolated_energy_statistical_stderr",
                _mapping(ts.get("extrapolation")).get(
                    "candidate_zero_step_energy_statistical_stderr"
                ),
            )
        )
        if timestep is not None
        else _optional_float(raw.get("energy_stderr")),
        "timestep_model_order_upper_allowance": _optional_float(model_order.get("upper_allowance")),
        "timestep_fit_window_upper_allowance": _optional_float(fit_window.get("upper_allowance")),
        "population_selected_last_doubling_upper_allowance": _optional_float(
            pop.get("selected_population_last_doubling_upper_allowance")
        ),
        "population_limit_model_window_upper_allowance": _optional_float(
            pop.get("population_limit_model_window_upper_allowance")
        ),
        "population_limit_correction_statistical_stderr": _optional_float(
            pop.get("population_limit_correction_statistical_stderr")
        ),
        "population_limit_correction_matched_seed_standard_error": _optional_float(
            pop.get("population_limit_correction_matched_seed_standard_error")
        ),
        "population_limit_correction_source_run_quadrature_standard_error": (
            _optional_float(
                pop.get("population_limit_correction_source_run_quadrature_standard_error")
            )
        ),
        "population_limit_correction_worst_case_arbitrary_covariance_"
        "standard_error_envelope": _optional_float(
            pop.get(
                "population_limit_correction_worst_case_arbitrary_covariance_"
                "standard_error_envelope"
            )
        ),
        "timestep_population_interaction_upper_allowance": _optional_float(
            pop.get("timestep_population_interaction_upper_allowance")
        ),
        "fw_anchor_rms_lag_relative_upper_bound": _optional_float(
            raw.get("rms_fw_lag_systematic_relative_upper_bound")
        ),
        "fw_anchor_density_lag_relative_l2_upper_bound": _optional_float(
            raw.get("density_fw_lag_systematic_relative_l2_upper_bound")
        ),
        "fw_treatment_r2_relative_upper_bound": _optional_float(
            r2_sensitivity.get("simultaneous_relative_upper_bound")
        ),
        "fw_treatment_rms_relative_upper_bound": _optional_float(
            rms_sensitivity.get("simultaneous_relative_upper_bound")
        ),
        "fw_treatment_density_relative_l2_upper_bound": _optional_float(
            density_sensitivity.get("simultaneous_upper_bound")
        ),
        "combination_rule": "reported separately; no cross-component quadrature sum",
    }


def _proposal_telemetry(summary: Mapping[str, Any]) -> dict[str, Any]:
    seeds = summary.get("seeds")
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("benchmark proposal telemetry has invalid declared seeds")
    return summarize_seed_proposal_telemetry(
        summary.get("seed_results"),
        expected_seed_ids=seeds,
    )


def _treatment_proposal_telemetry(
    anchor: Mapping[str, Any],
    fw_sensitivity: BoundSystematicAssessment | None,
) -> dict[str, Any]:
    if fw_sensitivity is None:
        return {
            "status": "anchor_only",
            "anchor": dict(anchor),
            "selected_candidate": None,
            "comparison": None,
            "source": "final-matrix fine-treatment benchmark packet",
        }
    treatments = _required_mapping(fw_sensitivity.summary, "treatments")
    anchor_treatment = _required_mapping(treatments, "anchor_density")
    candidate_treatment = _required_mapping(treatments, "candidate")
    declared_anchor = _required_mapping(anchor_treatment, "proposal_telemetry")
    candidate = _required_mapping(candidate_treatment, "proposal_telemetry")
    status = (
        "available"
        if declared_anchor.get("status") == candidate.get("status") == "available"
        else "telemetry_unavailable"
    )
    return {
        "status": status,
        "anchor": declared_anchor,
        "selected_candidate": candidate,
        "comparison": _proposal_telemetry_comparison(declared_anchor, candidate),
        "source": "manifest-bound FW sensitivity anchor and selected candidate treatments",
    }


def _proposal_telemetry_comparison(
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = (
        "local_acceptance_fraction_mean",
        "configuration_esjd_mean",
        "r2_esjd_mean",
        "weighted_free_gap_esjd_mean",
        "invalid_proposal_fraction_max",
        "metropolis_rejection_fraction_max",
    )
    comparison: dict[str, Any] = {}
    for metric in metrics:
        first = _optional_float(anchor.get(metric))
        second = _optional_float(candidate.get(metric))
        comparison[metric] = {
            "anchor": first,
            "selected_candidate": second,
            "delta": None if first is None or second is None else second - first,
            "ratio": (None if first is None or second is None or first == 0.0 else second / first),
        }
    return comparison


def _source_config(
    loaded: LoadedPackageInputs,
    *,
    reference_root: Path,
) -> dict[str, Any]:
    return {
        "final_matrix": _artifact_reference(
            loaded.final_manifest_path,
            loaded.final_summary_path,
            loaded.final_manifest,
            reference_root=reference_root,
        ),
        "raw_density": {
            case_id: _packet_reference(anchor.density, reference_root=reference_root)
            for case_id, anchor in loaded.anchors.items()
        },
        "raw_r2": {
            case_id: _packet_reference(anchor.r2, reference_root=reference_root)
            for case_id, anchor in loaded.anchors.items()
        },
        "timestep": {
            case_id: loaded.timestep[case_id].reference(reference_root=reference_root)
            for case_id in FINITE_CASE_ORDER
            if case_id in loaded.timestep
        },
        "population": {
            case_id: loaded.population[case_id].reference(reference_root=reference_root)
            for case_id in FINITE_CASE_ORDER
            if case_id in loaded.population
        },
        "forward_walking": {
            case_id: loaded.forward_walking[case_id].reference(reference_root=reference_root)
            for case_id in FINITE_CASE_ORDER
            if case_id in loaded.forward_walking
        },
        "fw_candidate": {
            case_id: _fw_candidate_reference(
                loaded.forward_walking[case_id],
                reference_root=reference_root,
            )
            for case_id in FINITE_CASE_ORDER
            if case_id in loaded.forward_walking
        },
    }


def _fw_candidate_reference(
    assessment: BoundSystematicAssessment,
    *,
    reference_root: Path,
) -> dict[str, Any]:
    config = _required_mapping(assessment.manifest, "config")
    declared = _required_mapping(config, "candidate")
    manifest_path = Path(str(declared.get("manifest_path", ""))).resolve()
    summary_path = Path(str(declared.get("summary_path", ""))).resolve()
    packet = load_manifest_bound_benchmark_packet(summary_path)
    reference = _packet_reference(packet, reference_root=reference_root)
    if manifest_path != packet.manifest_path:
        raise ValueError(f"{assessment.case_id}: FW candidate manifest path disagrees")
    return reference


def _artifact_reference(
    manifest_path: Path,
    summary_path: Path,
    manifest: Mapping[str, Any],
    *,
    reference_root: Path,
) -> dict[str, Any]:
    return {
        "manifest_path": _relative_locator(manifest_path, reference_root),
        "manifest_sha256": file_sha256(manifest_path),
        "summary_path": _relative_locator(summary_path, reference_root),
        "summary_sha256": file_sha256(summary_path),
        "run_name": manifest.get("run_name"),
        "result_schema_version": manifest.get("result_schema_version"),
        "run_id": manifest.get("run_id"),
        "bundle_sha256": manifest.get("bundle_sha256"),
        "status": manifest.get("status"),
    }


def _packet_reference(packet: Any, *, reference_root: Path | None = None) -> dict[str, Any]:
    reference = packet.reference()
    if reference_root is not None:
        reference["manifest_path"] = _relative_locator(packet.manifest_path, reference_root)
        reference["summary_path"] = _relative_locator(packet.summary_path, reference_root)
    reference["status"] = packet.summary.get("status")
    reference["run_name"] = packet.manifest.get("run_name")
    reference["result_schema_version"] = packet.manifest.get("result_schema_version")
    reference["verification_warnings"] = list(packet.verification_warnings)
    return reference


def _case_status_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _payload_rows(payload):
        telemetry = _mapping(row.get("proposal_telemetry"))
        anchor_telemetry = _mapping(telemetry.get("anchor"))
        candidate_telemetry = _mapping(telemetry.get("selected_candidate"))
        telemetry_comparison = _mapping(telemetry.get("comparison"))
        acceptance_comparison = _mapping(telemetry_comparison.get("local_acceptance_fraction_mean"))
        configuration_esjd_comparison = _mapping(
            telemetry_comparison.get("configuration_esjd_mean")
        )
        selected_population = _mapping(row.get("selected_population_treatment"))
        rows.append(
            {
                "case": row.get("case"),
                "status": row.get("status"),
                "publication_ready": row.get("publication_ready"),
                "energy_result_status": row.get("energy_result_status"),
                "raw_dt": row.get("raw_dt"),
                "raw_walkers": row.get("raw_walkers"),
                "raw_finite_dt_energy": row.get("raw_finite_dt_energy"),
                "candidate_zero_timestep_energy_at_selected_walkers": row.get(
                    "candidate_zero_timestep_energy_at_selected_walkers"
                ),
                "accepted_zero_timestep_energy_at_selected_walkers": row.get(
                    "accepted_zero_timestep_energy_at_selected_walkers"
                ),
                "bounded_zero_timestep_energy_at_selected_walkers": row.get(
                    "bounded_zero_timestep_energy_at_selected_walkers"
                ),
                "population_limit_correction": row.get(
                    "population_limit_correction_at_selected_timestep"
                ),
                "zero_timestep_population_limit_energy": row.get(
                    "zero_timestep_population_limit_energy"
                ),
                "thesis_energy": row.get("thesis_energy"),
                "timestep_status": _mapping(row.get("lane_status")).get("timestep"),
                "population_status": _mapping(row.get("lane_status")).get("population"),
                "forward_walking_status": _mapping(row.get("lane_status")).get("forward_walking"),
                "selected_population_dt": selected_population.get("selected_dt"),
                "selected_population_walkers": selected_population.get("walkers"),
                "selected_population_energy": selected_population.get("energy"),
                "anchor_acceptance_mean": anchor_telemetry.get("local_acceptance_fraction_mean"),
                "selected_candidate_acceptance_mean": candidate_telemetry.get(
                    "local_acceptance_fraction_mean"
                ),
                "acceptance_delta": acceptance_comparison.get("delta"),
                "acceptance_ratio": acceptance_comparison.get("ratio"),
                "anchor_configuration_esjd_mean": anchor_telemetry.get("configuration_esjd_mean"),
                "selected_candidate_configuration_esjd_mean": candidate_telemetry.get(
                    "configuration_esjd_mean"
                ),
                "configuration_esjd_ratio": configuration_esjd_comparison.get("ratio"),
                "configuration_esjd_delta": configuration_esjd_comparison.get("delta"),
                "anchor_r2_esjd_mean": anchor_telemetry.get("r2_esjd_mean"),
                "selected_candidate_r2_esjd_mean": candidate_telemetry.get("r2_esjd_mean"),
                "unresolved_reasons": ";".join(row.get("unresolved_reasons", [])),
            }
        )
    return rows


def _thesis_energy_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _payload_rows(payload):
        if row.get("publication_ready") is not True:
            continue
        uncertainty = _mapping(row.get("uncertainty_components"))
        rows.append(
            {
                "case": row.get("case"),
                "energy_status": row.get("energy_result_status"),
                "energy": row.get("thesis_energy"),
                "zero_timestep_energy_at_selected_walkers": (
                    row.get("accepted_zero_timestep_energy_at_selected_walkers")
                    if row.get("accepted_zero_timestep_energy_at_selected_walkers") is not None
                    else row.get("bounded_zero_timestep_energy_at_selected_walkers")
                ),
                "population_limit_correction_at_selected_timestep": row.get(
                    "population_limit_correction_at_selected_timestep"
                ),
                "energy_statistical_stderr": uncertainty.get("energy_statistical_stderr"),
                "population_limit_correction_statistical_stderr": uncertainty.get(
                    "population_limit_correction_statistical_stderr"
                ),
                "timestep_model_order_upper_allowance": uncertainty.get(
                    "timestep_model_order_upper_allowance"
                ),
                "timestep_fit_window_upper_allowance": uncertainty.get(
                    "timestep_fit_window_upper_allowance"
                ),
                "population_upper_allowance": uncertainty.get(
                    "population_selected_last_doubling_upper_allowance"
                ),
                "population_limit_model_window_upper_allowance": uncertainty.get(
                    "population_limit_model_window_upper_allowance"
                ),
                "timestep_population_interaction_upper_allowance": uncertainty.get(
                    "timestep_population_interaction_upper_allowance"
                ),
                "energy_lda": row.get("energy_lda"),
                "relative_delta_vs_lda": row.get("energy_relative_delta_vs_lda"),
                "bounded_qualifier": json.dumps(
                    row.get("bounded_qualifiers", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _uncertainty_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _payload_rows(payload):
        components = _mapping(row.get("uncertainty_components"))
        rows.append({"case": row.get("case"), **components})
    return rows


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = _required_mapping(payload, "sources")
    rows = [_source_table_record("", "final_matrix", _required_mapping(sources, "final_matrix"))]
    for lane in ("raw_density", "raw_r2", *SYSTEMATIC_LANES, "fw_candidate"):
        lane_sources = _required_mapping(sources, lane)
        rows.extend(
            _source_table_record(case_id, lane, _required_mapping(lane_sources, case_id))
            for case_id in REQUIRED_CASE_ORDER
            if case_id in lane_sources
        )
    return rows


def _source_table_record(case_id: str, lane: str, reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": case_id,
        "lane": lane,
        "manifest_path": reference.get("manifest_path"),
        "manifest_sha256": reference.get("manifest_sha256"),
        "summary_path": reference.get("summary_path"),
        "summary_sha256": reference.get("summary_sha256"),
        "run_name": reference.get("run_name"),
        "result_schema_version": reference.get("result_schema_version"),
        "run_id": reference.get("run_id"),
        "bundle_sha256": reference.get("bundle_sha256"),
        "status": reference.get("status"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(_csv_text(rows), encoding="utf-8")
    return path


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _validate_package_manifest_header(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if manifest.get("run_name") != NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME:
        raise ValueError("numerical-systematics package has the wrong artifact owner")
    if manifest.get("result_schema_version") != NUMERICAL_SYSTEMATICS_PACKAGE_SCHEMA_VERSION:
        raise ValueError("numerical-systematics package has the wrong result schema")
    if summary.get("schema_version") != NUMERICAL_SYSTEMATICS_PACKAGE_SCHEMA_VERSION:
        raise ValueError("numerical-systematics summary has the wrong result schema")
    expected_artifacts = set(_ARTIFACT_NAMES.values())
    actual = {
        entry.get("path") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    if actual != expected_artifacts:
        raise ValueError("numerical-systematics package has the wrong artifact set")
    config = _required_mapping(manifest, "config")
    if config.get("case_order") != list(REQUIRED_CASE_ORDER):
        raise ValueError("numerical-systematics package case order is invalid")
    if config.get("finite_case_order") != list(FINITE_CASE_ORDER):
        raise ValueError("numerical-systematics package finite-case order is invalid")
    if config.get("source_locator_base") != "package_directory":
        raise ValueError("numerical-systematics package source locator base is invalid")


def _validate_input_case_maps(**lanes: Mapping[str, Path]) -> None:
    finite = set(FINITE_CASE_ORDER)
    for lane, mapping in lanes.items():
        unknown = set(mapping) - finite
        if unknown:
            raise ValueError(
                f"{lane} inputs contain unsupported cases: {', '.join(sorted(unknown))}"
            )


def _validate_bounded_qualifiers(
    qualifiers: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for key, reason in qualifiers.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in FINITE_CASE_ORDER
            or key[1] not in SYSTEMATIC_LANES
        ):
            raise ValueError("bounded qualifier keys must be finite CASE/lane pairs")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"bounded qualifier for {key[0]}:{key[1]} is empty")
        result[key] = reason.strip()
    return result


def _validate_output_directory(root: Path, *, input_manifest_paths: Sequence[Path]) -> None:
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"numerical-systematics output directory is not empty: {root}")
    for path in input_manifest_paths:
        input_root = path.resolve().parent
        if root == input_root or root.is_relative_to(input_root) or input_root.is_relative_to(root):
            raise ValueError(
                "numerical-systematics output and input run directories must not overlap"
            )


def _final_rows_by_case(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("final-matrix rows are invalid")
    if [row.get("case") for row in rows] != list(REQUIRED_CASE_ORDER):
        raise ValueError("final-matrix rows are not in canonical case order")
    return {str(row["case"]): row for row in rows}


def _candidate_zero_step_energy(summary: Mapping[str, Any] | None) -> float | None:
    if summary is None:
        return None
    value = summary.get("extrapolated_energy")
    if value is None:
        value = _mapping(summary.get("extrapolation")).get("candidate_zero_step_energy")
    return _optional_float(value)


def _selected_population_treatment(
    assessment: BoundSystematicAssessment | None,
) -> dict[str, Any] | None:
    if assessment is None:
        return None
    summary = assessment.summary
    population_energy = summary.get("population_limit_energy_at_selected_timestep")
    energy_basis = "inverse_population_limit"
    statistical_stderr = summary.get("population_limit_energy_statistical_stderr")
    if population_energy is None:
        population_energy = summary.get("finite_population_energy_at_selected_timestep")
        statistical_stderr = summary.get("finite_population_energy_statistical_stderr")
        energy_basis = "finite_population_with_bounded_last_doubling"
    return {
        "classification": summary.get("classification"),
        "selected_dt": summary.get("selected_dt"),
        "selected_dt_basis": summary.get("selected_dt_basis"),
        "selected_treatment_role": summary.get("selected_treatment_role"),
        "selected_walkers": summary.get("selected_walkers"),
        "selected_walkers_basis": summary.get("selected_walkers_basis"),
        "energy": population_energy,
        "energy_statistical_stderr": statistical_stderr,
        "walkers": summary.get("selected_walkers"),
        "energy_basis": energy_basis if population_energy is not None else None,
        "population_limit_correction": summary.get(
            "population_limit_correction_at_selected_timestep"
        ),
        "population_limit_correction_statistical_stderr": summary.get(
            "population_limit_correction_statistical_stderr"
        ),
    }


def _population_energy_semantics(
    assessment: BoundSystematicAssessment | None,
) -> dict[str, Any]:
    if assessment is None:
        return {"status": "population_assessment_missing", "correction": None}
    summary = assessment.summary
    if summary.get("publication_ready_within_population_systematic_scope") is not True:
        return {"status": "population_assessment_unresolved", "correction": None}
    classification = summary.get("classification")
    if classification == "accepted_finite_population_bound":
        return {
            "status": "selected_finite_population",
            "selected_walkers": summary.get("selected_walkers"),
            "correction": None,
            "scope": (
                "zero-time-step energy remains at the selected finite walker population; "
                "the W-to-2W allowance is separate"
            ),
        }
    if classification == "accepted_population_limit":
        correction = _optional_float(
            summary.get("population_limit_correction_at_selected_timestep")
        )
        if correction is None:
            return {
                "status": "population_limit_correction_missing",
                "correction": None,
            }
        return {
            "status": "population_limit_correction",
            "selected_walkers": summary.get("selected_walkers"),
            "correction": correction,
            "correction_basis": summary.get("population_limit_correction_basis"),
            "scope": (
                "E(0,infinity) = E(0,selected_walkers) + "
                "[E(infinity,selected_dt)-E(selected_walkers,selected_dt)]"
            ),
        }
    return {"status": "population_classification_unresolved", "correction": None}


def _guide_binding(guide: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: guide.get(key)
        for key in (
            "relative_alpha",
            "contact_beta",
            "source_sha256",
            "source_manifest_sha256",
            "source_identity_fingerprint",
        )
    }


def _qualifiers_to_payload(
    qualifiers: Mapping[tuple[str, str], str],
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for (case_id, lane), reason in sorted(qualifiers.items()):
        payload.setdefault(case_id, {})[lane] = reason
    return payload


def _qualifiers_from_payload(value: object) -> dict[tuple[str, str], str]:
    if not isinstance(value, dict):
        raise ValueError("bounded qualifier declaration is invalid")
    result: dict[tuple[str, str], str] = {}
    for case_id, lanes in value.items():
        if not isinstance(case_id, str) or not isinstance(lanes, dict):
            raise ValueError("bounded qualifier declaration is invalid")
        for lane, reason in lanes.items():
            if not isinstance(lane, str) or not isinstance(reason, str):
                raise ValueError("bounded qualifier declaration is invalid")
            result[(case_id, lane)] = reason
    return _validate_bounded_qualifiers(result)


def _payload_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("package rows are invalid")
    return rows


def _relative_locator(path: Path, reference_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=reference_root.resolve())).as_posix()


def _resolve_reference(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("source artifact path is invalid")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("source artifact paths must be relative to the package directory")
    return (root / path).resolve()


def _relative_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return (value - reference) / reference


def _relative_error(value: float | None, reference: float) -> float | None:
    if value is None:
        return None
    return abs(value - reference) / abs(reference)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _required_float(value: object, description: str) -> float:
    result = _optional_float(value)
    if result is None:
        raise ValueError(f"{description} must be finite")
    return result


def _required_positive_float(value: object, description: str) -> float:
    result = _required_float(value, description)
    if result <= 0.0:
        raise ValueError(f"{description} must be positive")
    return result


def _required_positive_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value
