from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.analysis.fw_sensitivity import (
    ForwardWalkingSensitivityResult,
    analyze_fw_observable_sensitivity,
    classify_fw_sensitivity_status,
)
from hrdmc.analysis.proposal_telemetry import summarize_seed_proposal_telemetry
from hrdmc.artifacts import (
    build_run_provenance,
    config_fingerprint,
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.workflows.dmc.benchmark_packet.matrix_assembly import (
    FINAL_MATRIX_ASSEMBLY_RUN_NAME,
    FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION,
)

FloatArray = NDArray[np.float64]

FW_SENSITIVITY_SCHEMA_VERSION = "dmc_fw_sensitivity_v2"
FW_SENSITIVITY_RUN_NAME = "dmc_fw_sensitivity"
BENCHMARK_PACKET_SCHEMA_VERSION = "dmc_benchmark_packet_v3"
ACCEPTED_FW_STATUSES = {"accepted", "accepted_with_warnings"}


@dataclass(frozen=True)
class LoadedBenchmarkPacket:
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    verification_warnings: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return str(self.summary["case_id"])

    @property
    def controls(self) -> dict[str, Any]:
        return _required_mapping(self.summary, "controls")

    @property
    def pure_config(self) -> dict[str, Any]:
        return _required_mapping(self.summary, "pure_config")

    @property
    def seeds(self) -> tuple[int, ...]:
        values = self.summary.get("seeds")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"invalid seed identities: {self.summary_path}")
        return tuple(values)

    @property
    def dt(self) -> float:
        return _positive_float(self.controls.get("dt"), "dt")

    @property
    def walkers(self) -> int:
        value = self.controls.get("walkers")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid walker population: {self.summary_path}")
        return value

    @property
    def source_tree_sha256(self) -> str:
        provenance = _required_mapping(self.manifest, "provenance")
        implementation = _required_mapping(provenance, "implementation")
        return _required_string(implementation, "source_tree_sha256")

    def reference(self) -> dict[str, Any]:
        return {
            "summary_path": str(self.summary_path),
            "summary_sha256": file_sha256(self.summary_path),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "run_id": _required_string(self.manifest, "run_id"),
            "bundle_sha256": _required_string(self.manifest, "bundle_sha256"),
            "source_tree_sha256": self.source_tree_sha256,
            "dt": self.dt,
            "walkers": self.walkers,
        }


@dataclass(frozen=True)
class AnchorSources:
    assembly_manifest_path: Path
    assembly_manifest: dict[str, Any]
    density: LoadedBenchmarkPacket
    r2: LoadedBenchmarkPacket


def load_final_matrix_anchor_sources(path: Path, *, case_id: str) -> AnchorSources:
    """Load one exactly declared density/R2 pair from a final-matrix assembly."""

    return _load_anchor_sources(path, case_id=case_id)


def load_manifest_bound_benchmark_packet(summary_path: Path) -> LoadedBenchmarkPacket:
    """Load one benchmark packet through its exact summary/manifest binding."""

    return _load_packet(summary_path.resolve())


def run_fw_sensitivity_workflow(
    final_matrix_manifest_path: Path,
    candidate_summary_path: Path,
    *,
    case_id: str,
    output_dir: Path | None,
    command: list[str] | None = None,
    write_artifacts: bool = True,
    rms_relative_margin: float = 0.001,
    density_relative_l2_margin: float = 0.03,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Assess pure-FW timestep/population sensitivity against the final anchor."""

    _validate_analysis_controls(
        rms_relative_margin=rms_relative_margin,
        density_relative_l2_margin=density_relative_l2_margin,
        confidence_level=confidence_level,
    )
    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when write_artifacts is true")

    anchors = _load_anchor_sources(final_matrix_manifest_path, case_id=case_id)
    candidate = _load_packet(candidate_summary_path.resolve())
    if candidate.case_id != case_id:
        raise ValueError("candidate summary has the wrong case identity")
    _validate_anchor_analysis_policy(
        anchors,
        rms_relative_margin=rms_relative_margin,
        density_relative_l2_margin=density_relative_l2_margin,
        confidence_level=confidence_level,
    )

    sampling_design = build_fw_sampling_design(anchors, candidate)
    input_reasons, input_checks = _assess_input_quality(
        anchors,
        candidate,
        sampling_design=sampling_design,
    )
    grid_assessment = _assess_density_grid(anchors.density, candidate)
    plateau_assessment = _assess_plateaus(anchors, candidate)
    genealogy_assessment = _assess_genealogy(anchors, candidate)

    comparison: ForwardWalkingSensitivityResult | None = None
    comparison_error: str | None = None
    if (
        not input_reasons
        and grid_assessment["compatible"]
        and plateau_assessment["resolved"]
        and genealogy_assessment["supported"]
    ):
        try:
            comparison = _analyze_observables(
                anchors,
                candidate,
                rms_relative_margin=rms_relative_margin,
                density_relative_l2_margin=density_relative_l2_margin,
                confidence_level=confidence_level,
            )
        except ValueError as exc:
            comparison_error = str(exc)
            input_reasons.append(f"paired observable payload is invalid: {exc}")

    input_checks["paired_observable_payload_valid"] = (
        None if comparison is None and comparison_error is None else comparison_error is None
    )
    input_checks["input_quality_requirements_met"] = not input_reasons

    warnings = _verification_warnings(anchors, candidate)
    input_quality_accepted = not input_reasons
    plateau_resolved = bool(plateau_assessment["resolved"])
    genealogy_supported = bool(genealogy_assessment["supported"])
    observables_equivalent = comparison is not None and comparison.equivalent
    status = classify_fw_sensitivity_status(
        input_quality_accepted=input_quality_accepted,
        density_grid_compatible=bool(grid_assessment["compatible"]),
        plateau_resolved=plateau_resolved,
        genealogy_supported=genealogy_supported,
        observables_equivalent=observables_equivalent,
        has_warnings=bool(warnings),
    )
    publication_ready = status in ACCEPTED_FW_STATUSES
    unresolved_reasons = _unresolved_reasons(
        status=status,
        input_reasons=input_reasons,
        grid_assessment=grid_assessment,
        plateau_assessment=plateau_assessment,
        genealogy_assessment=genealogy_assessment,
        comparison_error=comparison_error,
    )

    identity = _scientific_identity(anchors, candidate)
    identity_fingerprint = config_fingerprint(identity)
    comparison_payload = None if comparison is None else comparison.to_dict()
    if comparison_payload is not None:
        comparison_payload["held_physical_lag_windows"] = _held_lag_windows(
            anchors,
            candidate,
        )
    payload: dict[str, Any] = {
        "schema_version": FW_SENSITIVITY_SCHEMA_VERSION,
        "status": status,
        "case_id": case_id,
        "diagnostic": "DMC transported forward-walking timestep/population sensitivity",
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "treatments": {
            "anchor_density": _treatment_record(anchors.density),
            "anchor_r2": _treatment_record(anchors.r2),
            "candidate": _treatment_record(candidate),
        },
        "sampling_design": sampling_design,
        "input_quality": {
            "status": "accepted" if input_quality_accepted else "unresolved",
            "reasons": input_reasons,
            "checks": input_checks,
            "manifest_verification_warnings": warnings,
            "warning_policy": "only unrelated artifacts under plots/ may drift",
        },
        "density_grid": grid_assessment,
        "plateau_assessment": plateau_assessment,
        "genealogy_assessment": genealogy_assessment,
        "observable_comparison": comparison_payload,
        "unresolved_reasons": unresolved_reasons,
        "publication_ready_within_fw_sensitivity_scope": publication_ready,
        "qualified_systematics": {
            "forward_walking_timestep_population_sensitivity": (
                "accepted" if publication_ready else "unresolved"
            )
        },
        "scientific_scope": (
            "Paired seed-level sensitivity of transported pure R2/RMS and density "
            "to the candidate timestep/population treatment. Density is compared on "
            "one exact common grid. The fixed anchor shell-cell envelope and shell "
            "peaks are descriptive diagnostics; no binwise zero-timestep or "
            "infinite-population extrapolation is performed."
        ),
    }

    config = {
        "case_id": case_id,
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "rms_relative_margin": rms_relative_margin,
        "density_relative_l2_margin": density_relative_l2_margin,
        "confidence_level": confidence_level,
        "sampling_design": sampling_design,
        "final_matrix_manifest": {
            "path": str(anchors.assembly_manifest_path),
            "sha256": file_sha256(anchors.assembly_manifest_path),
            "run_id": _required_string(anchors.assembly_manifest, "run_id"),
            "bundle_sha256": _required_string(anchors.assembly_manifest, "bundle_sha256"),
        },
        "anchor_density": anchors.density.reference(),
        "anchor_r2": anchors.r2.reference(),
        "candidate": candidate.reference(),
    }
    artifacts: dict[str, str | None] = {
        "summary": None,
        "observable_table": None,
        "shell_table": None,
        "run_manifest": None,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
    }
    if write_artifacts:
        assert output_dir is not None
        root = output_dir.resolve()
        _validate_output_directory(root, anchors=anchors, candidate=candidate)
        ensure_dir(root)
        summary_path = root / "summary.json"
        observable_table_path = root / "observable_comparison.csv"
        shell_table_path = root / "shell_comparison.csv"
        payload["artifacts"] = {
            "summary": str(summary_path),
            "observable_table": str(observable_table_path),
            "shell_table": str(shell_table_path),
            "run_manifest": str(root / "run_manifest.json"),
            "output_dir": str(root),
        }
        write_json(summary_path, payload)
        _write_observable_table(observable_table_path, comparison)
        _write_shell_table(shell_table_path, comparison)
        manifest_path = write_run_manifest(
            root,
            run_name=FW_SENSITIVITY_RUN_NAME,
            config=config,
            artifacts=[summary_path, observable_table_path, shell_table_path],
            schema_version=FW_SENSITIVITY_SCHEMA_VERSION,
            provenance=build_run_provenance(command),
            status=status,
        )
        artifacts = {
            "summary": str(summary_path),
            "observable_table": str(observable_table_path),
            "shell_table": str(shell_table_path),
            "run_manifest": str(manifest_path),
            "output_dir": str(root),
        }
    else:
        payload["artifacts"] = artifacts
    payload["workflow_artifacts"] = artifacts
    return payload


def _load_anchor_sources(path: Path, *, case_id: str) -> AnchorSources:
    manifest_path = path.resolve()
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError("final matrix manifest verification failed: " + "; ".join(errors))
    manifest = _load_json_mapping(manifest_path, "final matrix manifest")
    if manifest.get("run_name") != FINAL_MATRIX_ASSEMBLY_RUN_NAME:
        raise ValueError("final matrix manifest has the wrong owner")
    if manifest.get("result_schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("final matrix manifest has the wrong result schema")
    if manifest.get("status") != "accepted":
        raise ValueError("final matrix manifest is not accepted")
    config = _required_mapping(manifest, "config")
    sources = _required_mapping(config, "sources")
    case_sources = _required_mapping(sources, case_id)
    assembly_summary_path = manifest_path.parent / "final_matrix_summary.json"
    assembly_summary = _load_json_mapping(assembly_summary_path, "final matrix summary")
    if assembly_summary.get("schema_version") != FINAL_MATRIX_ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("final matrix summary has the wrong result schema")
    if assembly_summary.get("status") != "accepted":
        raise ValueError("final matrix summary is not accepted")
    rows = assembly_summary.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("final matrix summary rows are invalid")
    selected_rows = [row for row in rows if row.get("case") == case_id]
    if len(selected_rows) != 1:
        raise ValueError(f"final matrix summary does not contain one row for {case_id}")
    selected_row = selected_rows[0]
    if selected_row.get("status") != "accepted":
        raise ValueError(f"final matrix row is not accepted for {case_id}")
    if selected_row.get("density_status") != "accepted":
        raise ValueError(f"final matrix density source is not accepted for {case_id}")
    if selected_row.get("r2_status") != "accepted":
        raise ValueError(f"final matrix R2 source is not accepted for {case_id}")
    if selected_row.get("primary_source") != case_sources.get("primary"):
        raise ValueError(f"final matrix primary source selection disagrees for {case_id}")
    if selected_row.get("r2_source") != case_sources.get("r2"):
        raise ValueError(f"final matrix R2 source selection disagrees for {case_id}")
    density = _load_declared_packet(
        manifest_path.parent,
        case_id=case_id,
        reference=_required_mapping(case_sources, "primary"),
    )
    r2 = _load_declared_packet(
        manifest_path.parent,
        case_id=case_id,
        reference=_required_mapping(case_sources, "r2"),
    )
    return AnchorSources(
        assembly_manifest_path=manifest_path,
        assembly_manifest=manifest,
        density=density,
        r2=r2,
    )


def _load_declared_packet(
    reference_root: Path,
    *,
    case_id: str,
    reference: Mapping[str, Any],
) -> LoadedBenchmarkPacket:
    directory = _resolve_reference(reference_root, reference.get("directory"))
    manifest_path = _resolve_reference(reference_root, reference.get("manifest_path"))
    summary_path = _resolve_reference(reference_root, reference.get("summary_path"))
    if manifest_path != directory / "run_manifest.json":
        raise ValueError(f"{case_id}: declared manifest path is invalid")
    if summary_path != directory / "summary.json":
        raise ValueError(f"{case_id}: declared summary path is invalid")
    if not manifest_path.is_file() or file_sha256(manifest_path) != reference.get(
        "manifest_sha256"
    ):
        raise ValueError(f"{case_id}: declared manifest identity mismatch")
    if not summary_path.is_file() or file_sha256(summary_path) != reference.get("summary_sha256"):
        raise ValueError(f"{case_id}: declared summary identity mismatch")
    loaded = _load_packet(summary_path)
    if loaded.case_id != case_id:
        raise ValueError(f"{case_id}: declared packet has the wrong case")
    if loaded.manifest.get("run_id") != reference.get("run_id"):
        raise ValueError(f"{case_id}: declared run identity mismatch")
    if loaded.manifest.get("bundle_sha256") != reference.get("bundle_sha256"):
        raise ValueError(f"{case_id}: declared bundle identity mismatch")
    return loaded


def _load_packet(summary_path: Path) -> LoadedBenchmarkPacket:
    if not summary_path.is_file():
        raise FileNotFoundError(f"benchmark summary does not exist: {summary_path}")
    manifest_path = summary_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest does not exist: {manifest_path}")
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        summary_path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    summary = _load_json_mapping(summary_path, "benchmark summary")
    if manifest.get("run_name") != "dmc_benchmark_packet":
        raise ValueError(f"benchmark packet has the wrong owner: {summary_path}")
    if manifest.get("result_schema_version") != BENCHMARK_PACKET_SCHEMA_VERSION:
        raise ValueError(f"benchmark packet has an unsupported schema: {summary_path}")
    if summary.get("schema_version") != BENCHMARK_PACKET_SCHEMA_VERSION:
        raise ValueError(f"benchmark summary has an unsupported schema: {summary_path}")
    if summary.get("status") != manifest.get("status"):
        raise ValueError(f"benchmark summary and manifest statuses disagree: {summary_path}")
    config = _required_mapping(manifest, "config")
    if config.get("case") != summary.get("case_id"):
        raise ValueError(f"benchmark case identity disagrees with its manifest: {summary_path}")
    for field in ("controls", "pure_config"):
        if summary.get(field) != config.get(field):
            raise ValueError(f"benchmark {field} disagrees with its manifest: {summary_path}")
    if summary.get("collective_rn_controls") != config.get("collective_rn"):
        raise ValueError(
            f"benchmark collective-RN controls disagree with its manifest: {summary_path}"
        )
    if summary.get("seeds") != config.get("seeds"):
        raise ValueError(f"benchmark seed identities disagree with its manifest: {summary_path}")
    return LoadedBenchmarkPacket(
        summary_path=summary_path,
        manifest_path=manifest_path,
        summary=summary,
        manifest=manifest,
        verification_warnings=tuple(warnings),
    )


def _assess_input_quality(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
    *,
    sampling_design: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    packets = {
        "anchor_density": anchors.density,
        "anchor_r2": anchors.r2,
        "candidate": candidate,
    }
    for label, packet in packets.items():
        reasons.extend(f"{label}: {reason}" for reason in _internal_control_reasons(packet))

    reference = anchors.density
    if anchors.r2.controls != reference.controls:
        reasons.append("anchor sources: fine-timestep treatment controls differ")
    if candidate.summary.get("status") != "accepted":
        reasons.append("candidate: benchmark packet status is not accepted")
    if candidate.summary.get("energy_validation_status") != "accepted":
        reasons.append("candidate: mixed-energy validation status is not accepted")
    if candidate.summary.get("pure_fw_validation_status") != "accepted":
        reasons.append("candidate: pure-FW validation status is not accepted")
    if candidate.dt == reference.dt and candidate.walkers == reference.walkers:
        reasons.append("candidate: timestep and walker treatment is identical to the anchor")
    for label, packet in (("anchor_r2", anchors.r2), ("candidate", candidate)):
        if packet.seeds != reference.seeds:
            reasons.append(f"{label}: seed identities differ from the density anchor")
        for field in (
            "case_id",
            "n_particles",
            "rod_length",
            "rod_length_ho",
            "guide_family",
            "coordinate",
            "energy_coordinate",
            "length_unit",
            "time_unit",
            "energy_unit",
            "report_energy_unit",
            "initialization_mode",
            "init_width_log_sigma",
            "breathing_preburn_steps",
            "breathing_preburn_log_step",
            "collective_rn_controls",
        ):
            if packet.summary.get(field) != reference.summary.get(field):
                reasons.append(f"{label}: {field} differs from the density anchor")
        if _guide_identity(packet) != _guide_identity(reference):
            reasons.append(f"{label}: guide identity differs from the density anchor")
        if packet.source_tree_sha256 != reference.source_tree_sha256:
            reasons.append(f"{label}: implementation source tree differs from the anchor")
        for control in ("local_step_method", "drift_limiter"):
            if packet.controls.get(control) != reference.controls.get(control):
                reasons.append(f"{label}: {control} differs from the density anchor")
        for physical_control in ("burn_tau", "production_tau"):
            if not _same_float(
                packet.controls.get(physical_control),
                reference.controls.get(physical_control),
            ):
                reasons.append(f"{label}: physical {physical_control} differs from the anchor")

    candidate_observables = candidate.pure_config.get("observables")
    if not isinstance(candidate_observables, list) or not {"r2", "density"}.issubset(
        candidate_observables
    ):
        reasons.append("candidate: both r2 and density FW observables are required")
    if _density_estimator(candidate) != _density_estimator(reference):
        reasons.append("candidate: density estimator identity differs from the anchor")
    if _r2_estimator(candidate) != _r2_estimator(anchors.r2):
        reasons.append("candidate: R2 estimator identity differs from the anchor")

    reasons.extend(
        _pure_config_identity_reasons(
            anchors,
            candidate,
            sampling_design=sampling_design,
        )
    )
    checks = {
        "same_seed_identities": all(packet.seeds == reference.seeds for packet in packets.values()),
        "same_composed_anchor_treatment": anchors.r2.controls == reference.controls,
        "same_guide_identity": all(
            _guide_identity(packet) == _guide_identity(reference) for packet in packets.values()
        ),
        "same_source_tree": all(
            packet.source_tree_sha256 == reference.source_tree_sha256 for packet in packets.values()
        ),
        "same_physical_burn_and_production": all(
            _same_float(packet.controls.get(name), reference.controls.get(name))
            for packet in packets.values()
            for name in ("burn_tau", "production_tau")
        ),
        "block_size_policy": (
            "sliding-window block_size_steps is held at the exact one-event setting; "
            "collection strides are recorded as an estimator sampling design"
        ),
        "sampling_cadence_phase_safe": sampling_design.get("phase_safe") is True,
        "physical_and_estimator_identity_held": not reasons,
    }
    return reasons, checks


def _pure_config_identity_reasons(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
    *,
    sampling_design: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    shared_fields = (
        "center",
        "block_size_steps",
        "collection_mode",
        "transport_mode",
        "transport_invariant_tests_passed",
        "min_block_count",
        "min_source_ancestor_ess",
        "max_source_family_fraction",
        "min_walker_weight_ess",
        "plateau_abs_tolerance",
        "plateau_sigma_threshold",
        "plateau_equivalence_confidence_level",
    )
    for field in shared_fields:
        if anchors.density.pure_config.get(field) != anchors.r2.pure_config.get(field):
            reasons.append(f"anchor sources: shared FW {field} differs")
        if candidate.pure_config.get(field) != anchors.r2.pure_config.get(field):
            reasons.append(f"candidate: FW {field} differs from the R2 anchor")
    for field in (
        "observable_source",
        "r2_rb_com_variance",
        "plateau_window_lag_count",
        "rms_plateau_relative_tolerance",
    ):
        if candidate.pure_config.get(field) != anchors.r2.pure_config.get(field):
            reasons.append(f"candidate: R2 {field} differs from the R2 anchor")
    for field in (
        "density_source",
        "density_com_variance",
        "density_parity_average",
        "density_expected_particles",
        "density_accounting_abs_tolerance",
        "density_plateau_window_lag_count",
        "density_plateau_relative_l2_tolerance",
    ):
        if candidate.pure_config.get(field) != anchors.density.pure_config.get(field):
            reasons.append(f"candidate: density {field} differs from the density anchor")
    if sampling_design.get("phase_safe") is not True:
        reasons.append(
            "candidate: collection cadence differs while a scheduled collective move is active"
        )
    return reasons


def _internal_control_reasons(packet: LoadedBenchmarkPacket) -> list[str]:
    reasons: list[str] = []
    dt = packet.dt
    for steps_name, tau_name in (
        ("burn_in_steps", "burn_tau"),
        ("production_steps", "production_tau"),
    ):
        steps = packet.controls.get(steps_name)
        tau = packet.controls.get(tau_name)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            reasons.append(f"{steps_name} is not a positive integer")
        elif not _same_float(float(steps) * dt, tau):
            reasons.append(f"{steps_name}*dt does not reproduce {tau_name}")
    if packet.pure_config.get("lag_unit") != "dmc_steps":
        reasons.append("FW lag unit is not dmc_steps")
    center = packet.pure_config.get("center")
    if (
        isinstance(center, bool)
        or not isinstance(center, (int, float))
        or not math.isfinite(float(center))
    ):
        reasons.append("FW center is not finite")
    block_size = packet.pure_config.get("block_size_steps")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        reasons.append("FW block_size_steps is not a positive integer")
    fields = ["collection_stride_steps"]
    observables = packet.pure_config.get("observables")
    if isinstance(observables, list) and "density" in observables:
        fields.append("density_collection_stride_steps")
    for field in fields:
        value = packet.pure_config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            reasons.append(f"{field} is not a positive integer")
    return reasons


def _assess_density_grid(
    anchor: LoadedBenchmarkPacket,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    anchor_edges, anchor_seed_edges = _density_edges(anchor)
    candidate_edges, candidate_seed_edges = _density_edges(candidate)
    reasons: list[str] = []
    if not np.array_equal(anchor_edges, candidate_edges):
        reasons.append("aggregate density bin edges are not byte-value identical")
    for label, rows, aggregate in (
        ("anchor", anchor_seed_edges, anchor_edges),
        ("candidate", candidate_seed_edges, candidate_edges),
    ):
        for seed_index, edges in enumerate(rows):
            if not np.array_equal(edges, aggregate):
                reasons.append(f"{label} seed index {seed_index} uses different bin edges")
    compatible = not reasons
    return {
        "status": "accepted" if compatible else "incompatible",
        "compatible": compatible,
        "reasons": reasons,
        "bin_count": int(anchor_edges.size - 1),
        "anchor_edge_fingerprint": config_fingerprint(anchor_edges.tolist()),
        "candidate_edge_fingerprint": config_fingerprint(candidate_edges.tolist()),
        "comparison_rule": "exact float-array equality; no interpolation or rebinning",
    }


def _assess_plateaus(anchors: AnchorSources, candidate: LoadedBenchmarkPacket) -> dict[str, Any]:
    records = {
        "anchor_density": _plateau_record(anchors.density, "density"),
        "anchor_r2": _plateau_record(anchors.r2, "r2"),
        "candidate_density": _plateau_record(candidate, "density"),
        "candidate_r2": _plateau_record(candidate, "r2"),
    }
    reasons = [
        f"{label}: {reason}" for label, record in records.items() for reason in record["reasons"]
    ]
    for observable, anchor_label in (("density", "anchor_density"), ("r2", "anchor_r2")):
        anchor_record = records[anchor_label]
        candidate_record = records[f"candidate_{observable}"]
        anchor_lags = anchor_record["selected_physical_lags"]
        candidate_requested = candidate_record["requested_physical_lags"]
        if not _contains_physical_lags(candidate_requested, anchor_lags):
            reasons.append(
                f"candidate_{observable}: requested lags do not contain the anchor plateau window"
            )
        if not _same_physical_lag_sequence(
            candidate_record["selected_physical_lags"],
            anchor_lags,
        ):
            reasons.append(
                f"candidate_{observable}: selected physical plateau window differs from anchor"
            )
        anchor_terminal = anchor_record["selected_terminal_physical_lag"]
        candidate_terminal = candidate_record["selected_terminal_physical_lag"]
        if (
            not isinstance(anchor_terminal, float)
            or not isinstance(candidate_terminal, float)
            or candidate_terminal + 1.0e-12 < anchor_terminal
        ):
            reasons.append(
                f"candidate_{observable}: selected terminal physical lag is shorter than anchor"
            )
    resolved = not reasons
    return {
        "status": "accepted" if resolved else "unresolved",
        "resolved": resolved,
        "reasons": reasons,
        "observables": records,
        "lag_comparison_unit": "1/Omega",
    }


def _plateau_record(packet: LoadedBenchmarkPacket, observable: str) -> dict[str, Any]:
    result = _observable_result(packet, observable)
    diagnostics = _required_mapping(result, "aggregate_plateau_diagnostics")
    reasons: list[str] = []
    estimate = _estimate(packet, observable)
    if estimate.get("status") != "accepted":
        reasons.append("reported observable estimate is not accepted")
    if result.get("aggregate_schema_status") != "schema_valid":
        reasons.append("aggregate FW schema is invalid")
    if result.get("aggregate_plateau_status") != "plateau_resolved":
        reasons.append("aggregate FW plateau is unresolved")
    if diagnostics.get("decision") != "plateau_resolved":
        reasons.append("aggregate plateau decision is unresolved")
    if diagnostics.get("equivalence_pass") is not True:
        reasons.append("within-run plateau equivalence did not pass")
    selected_value = diagnostics.get("selected_window_lags")
    selected = _optional_integer_list(selected_value)
    if selected is None:
        selected = []
        reasons.append("aggregate plateau selected-window lags are unavailable")
    requested_field = "density_lag_steps" if observable == "density" else "lag_steps"
    requested = _integer_list(packet.pure_config.get(requested_field), requested_field)
    selected_physical = [float(lag) * packet.dt for lag in selected]
    requested_physical = [float(lag) * packet.dt for lag in requested]
    terminal = max(selected_physical) if selected_physical else None
    if not selected_physical or any(lag <= 0 for lag in selected):
        reasons.append("aggregate plateau selected no nonzero lag window")
    return {
        "status": "accepted" if not reasons else "unresolved",
        "reasons": reasons,
        "selected_step_lags": selected,
        "selected_physical_lags": selected_physical,
        "selected_terminal_physical_lag": terminal,
        "requested_step_lags": requested,
        "requested_physical_lags": requested_physical,
        "aggregate_plateau_status": result.get("aggregate_plateau_status"),
        "aggregate_schema_status": result.get("aggregate_schema_status"),
        "decision_level": result.get("decision_level"),
    }


def _assess_genealogy(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    anchor_density_physical = _selected_physical_lags(anchors.density, "density")
    anchor_r2_physical = _selected_physical_lags(anchors.r2, "r2")
    records = {
        "anchor_density": _genealogy_record(anchors.density, "density"),
        "anchor_r2": _genealogy_record(anchors.r2, "r2"),
        "candidate_density": _genealogy_record(
            candidate,
            "density",
            comparison_step_lags=_matching_step_lags(
                candidate,
                "density",
                anchor_density_physical,
                require_all=False,
            ),
        ),
        "candidate_r2": _genealogy_record(
            candidate,
            "r2",
            comparison_step_lags=_matching_step_lags(
                candidate,
                "r2",
                anchor_r2_physical,
                require_all=False,
            ),
        ),
    }
    reasons = [
        f"{label}: {reason}" for label, record in records.items() for reason in record["reasons"]
    ]
    supported = not reasons
    return {
        "status": "accepted" if supported else "unresolved",
        "supported": supported,
        "reasons": reasons,
        "observables": records,
    }


def _genealogy_record(
    packet: LoadedBenchmarkPacket,
    observable: str,
    *,
    comparison_step_lags: list[int] | None = None,
) -> dict[str, Any]:
    result = _observable_result(packet, observable)
    diagnostics = _required_mapping(result, "aggregate_plateau_diagnostics")
    lag_support = _required_mapping(diagnostics, "lag_support")
    reasons: list[str] = []
    selected = _optional_integer_list(diagnostics.get("selected_window_lags"))
    if selected is None:
        selected = []
        reasons.append("aggregate genealogy selected-window lags are unavailable")
    checked_lags = list(dict.fromkeys([*selected, *(comparison_step_lags or [])]))
    if result.get("aggregate_genealogy_status") != "genealogy_support_accepted":
        reasons.append("aggregate genealogy status is unresolved")
    pooled_ancestor_values: list[float] = []
    pooled_family_values: list[float] = []
    walker_ess_values: list[float] = []
    block_counts: list[int] = []
    for lag in checked_lags:
        support_value = lag_support.get(str(lag))
        if not isinstance(support_value, dict):
            reasons.append(f"selected lag {lag} has no support record")
            continue
        support = support_value
        if support.get("supported") is not True:
            reasons.append(f"selected lag {lag} is not supported")
        ancestor = _finite_float(
            support.get("pooled_ancestor_ess_lower_bound"),
            "pooled_ancestor_ess_lower_bound",
        )
        required_ancestor = _finite_float(
            support.get("required_pooled_ancestor_ess"),
            "required_pooled_ancestor_ess",
        )
        family = _finite_float(
            support.get("pooled_family_fraction_upper_bound"),
            "pooled_family_fraction_upper_bound",
        )
        maximum_family = _finite_float(
            support.get("maximum_pooled_family_fraction"),
            "maximum_pooled_family_fraction",
        )
        configured_ancestor = _finite_float(
            packet.pure_config.get("min_source_ancestor_ess"),
            "min_source_ancestor_ess",
        )
        configured_family = _finite_float(
            packet.pure_config.get("max_source_family_fraction"),
            "max_source_family_fraction",
        )
        walker_ess = _finite_float(
            support.get("min_walker_weight_ess"),
            "min_walker_weight_ess",
        )
        min_walker_ess = _finite_float(
            packet.pure_config.get("min_walker_weight_ess"),
            "min_walker_weight_ess",
        )
        block_count = support.get("min_block_count")
        required_blocks = packet.pure_config.get("min_block_count")
        if isinstance(block_count, bool) or not isinstance(block_count, int):
            raise ValueError("genealogy support block count is invalid")
        if isinstance(required_blocks, bool) or not isinstance(required_blocks, int):
            raise ValueError("configured minimum block count is invalid")
        if ancestor < required_ancestor:
            reasons.append(f"selected lag {lag} has insufficient pooled ancestor ESS")
        if not _same_float(required_ancestor, configured_ancestor):
            reasons.append(f"selected lag {lag} uses an unbound ancestor-ESS threshold")
        if family > maximum_family:
            reasons.append(f"selected lag {lag} exceeds pooled family concentration")
        if not _same_float(maximum_family, configured_family):
            reasons.append(f"selected lag {lag} uses an unbound family-fraction threshold")
        if walker_ess < min_walker_ess:
            reasons.append(f"selected lag {lag} has insufficient walker-weight ESS")
        if block_count < required_blocks:
            reasons.append(f"selected lag {lag} has insufficient collected source windows")
        pooled_ancestor_values.append(ancestor)
        pooled_family_values.append(family)
        walker_ess_values.append(walker_ess)
        block_counts.append(block_count)
    return {
        "status": "accepted" if not reasons else "unresolved",
        "reasons": reasons,
        "aggregate_genealogy_status": result.get("aggregate_genealogy_status"),
        "selected_step_lags": selected,
        "selected_physical_lags": [float(lag) * packet.dt for lag in selected],
        "comparison_step_lags": comparison_step_lags or [],
        "comparison_physical_lags": [
            float(lag) * packet.dt for lag in (comparison_step_lags or [])
        ],
        "checked_step_lags": checked_lags,
        "minimum_pooled_ancestor_ess_lower_bound": (
            min(pooled_ancestor_values) if pooled_ancestor_values else None
        ),
        "maximum_pooled_family_fraction_upper_bound": (
            max(pooled_family_values) if pooled_family_values else None
        ),
        "minimum_walker_weight_ess": min(walker_ess_values) if walker_ess_values else None,
        "minimum_block_count": min(block_counts) if block_counts else None,
    }


def _analyze_observables(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
    *,
    rms_relative_margin: float,
    density_relative_l2_margin: float,
    confidence_level: float,
) -> ForwardWalkingSensitivityResult:
    anchor_r2_selected = _selected_window_lags(anchors.r2, "r2")
    anchor_density_selected = _selected_window_lags(anchors.density, "density")
    candidate_r2_selected = _selected_window_lags(candidate, "r2")
    candidate_density_selected = _selected_window_lags(candidate, "density")
    _verify_aggregate_matches_seed_mean(
        anchors.r2,
        "r2",
        _seed_r2_values(anchors.r2, step_lags=anchor_r2_selected),
    )
    _verify_aggregate_matches_seed_mean(
        candidate,
        "r2",
        _seed_r2_values(candidate, step_lags=candidate_r2_selected),
    )
    _verify_aggregate_matches_seed_mean(
        anchors.density,
        "density",
        _seed_density_values(anchors.density, step_lags=anchor_density_selected),
    )
    _verify_aggregate_matches_seed_mean(
        candidate,
        "density",
        _seed_density_values(candidate, step_lags=candidate_density_selected),
    )

    anchor_r2_physical = [float(lag) * anchors.r2.dt for lag in anchor_r2_selected]
    anchor_density_physical = [float(lag) * anchors.density.dt for lag in anchor_density_selected]
    candidate_r2_comparison = _matching_step_lags(
        candidate,
        "r2",
        anchor_r2_physical,
    )
    candidate_density_comparison = _matching_step_lags(
        candidate,
        "density",
        anchor_density_physical,
    )
    anchor_r2 = _seed_r2_values(anchors.r2, step_lags=anchor_r2_selected)
    candidate_r2 = _seed_r2_values(candidate, step_lags=candidate_r2_comparison)
    anchor_density = _seed_density_values(
        anchors.density,
        step_lags=anchor_density_selected,
    )
    candidate_density = _seed_density_values(
        candidate,
        step_lags=candidate_density_comparison,
    )
    edges, _ = _density_edges(anchors.density)
    n_particles = anchors.density.summary.get("n_particles")
    if isinstance(n_particles, bool) or not isinstance(n_particles, int):
        raise ValueError("anchor particle count is invalid")
    density_tolerance = _finite_float(
        anchors.density.pure_config.get("density_accounting_abs_tolerance"),
        "density_accounting_abs_tolerance",
    )
    return analyze_fw_observable_sensitivity(
        anchor_r2_by_seed=anchor_r2,
        candidate_r2_by_seed=candidate_r2,
        bin_edges=edges,
        anchor_density_by_seed=anchor_density,
        candidate_density_by_seed=candidate_density,
        particle_count=n_particles,
        rms_relative_margin=rms_relative_margin,
        density_relative_l2_margin=density_relative_l2_margin,
        confidence_level=confidence_level,
        density_normalization_atol=density_tolerance,
    )


def _seed_r2_values(
    packet: LoadedBenchmarkPacket,
    *,
    step_lags: list[int] | None = None,
) -> FloatArray:
    selected_lags = _selected_window_lags(packet, "r2") if step_lags is None else step_lags
    values = []
    for result in _ordered_seed_results(packet):
        observable = _required_mapping(
            _required_mapping(_required_mapping(result, "pure_walking"), "observable_results"),
            "r2",
        )
        values_by_lag = _required_mapping(observable, "values_by_lag")
        selected_values = [
            _positive_float(values_by_lag.get(str(lag)), f"R2 value at lag {lag}")
            for lag in selected_lags
        ]
        values.append(float(np.mean(selected_values)))
    return np.asarray(values, dtype=np.float64)


def _seed_density_values(
    packet: LoadedBenchmarkPacket,
    *,
    step_lags: list[int] | None = None,
) -> FloatArray:
    selected_lags = _selected_window_lags(packet, "density") if step_lags is None else step_lags
    rows = []
    for result in _ordered_seed_results(packet):
        observable = _required_mapping(
            _required_mapping(_required_mapping(result, "pure_walking"), "observable_results"),
            "density",
        )
        values_by_lag = _required_mapping(observable, "values_by_lag")
        selected_rows = []
        for lag in selected_lags:
            row = np.asarray(values_by_lag.get(str(lag)), dtype=np.float64)
            if row.ndim != 1 or not np.all(np.isfinite(row)) or np.any(row < 0.0):
                raise ValueError(f"seed density value at lag {lag} is invalid")
            selected_rows.append(row)
        rows.append(np.mean(np.asarray(selected_rows, dtype=np.float64), axis=0))
    return np.asarray(rows, dtype=np.float64)


def _selected_window_lags(packet: LoadedBenchmarkPacket, observable: str) -> list[int]:
    result = _observable_result(packet, observable)
    diagnostics = _required_mapping(result, "aggregate_plateau_diagnostics")
    lags = _optional_integer_list(diagnostics.get("selected_window_lags"))
    if not lags:
        raise ValueError(f"aggregate {observable} selected-window lags are unavailable")
    return lags


def _selected_physical_lags(
    packet: LoadedBenchmarkPacket,
    observable: str,
) -> list[float]:
    return [float(lag) * packet.dt for lag in _selected_window_lags(packet, observable)]


def _matching_step_lags(
    packet: LoadedBenchmarkPacket,
    observable: str,
    physical_lags: list[float],
    *,
    require_all: bool = True,
) -> list[int]:
    field = "density_lag_steps" if observable == "density" else "lag_steps"
    requested = _integer_list(packet.pure_config.get(field), field)
    matches: list[int] = []
    for physical_lag in physical_lags:
        candidates = [
            lag
            for lag in requested
            if math.isclose(
                float(lag) * packet.dt,
                physical_lag,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ]
        if len(candidates) != 1:
            if require_all:
                raise ValueError(
                    f"candidate {observable} lags do not map uniquely to "
                    f"physical lag {physical_lag}"
                )
            return []
        matches.append(candidates[0])
    return matches


def _held_lag_windows(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for observable, anchor in (
        ("r2", anchors.r2),
        ("density", anchors.density),
    ):
        anchor_steps = _selected_window_lags(anchor, observable)
        physical_lags = [float(lag) * anchor.dt for lag in anchor_steps]
        records[observable] = {
            "physical_lags": physical_lags,
            "anchor_step_lags": anchor_steps,
            "candidate_step_lags": _matching_step_lags(
                candidate,
                observable,
                physical_lags,
            ),
            "unit": "1/Omega",
            "comparison_rule": ("paired values are evaluated at the anchor physical lag window"),
        }
    return records


def _density_edges(packet: LoadedBenchmarkPacket) -> tuple[FloatArray, list[FloatArray]]:
    estimate = _estimate(packet, "density")
    aggregate = np.asarray(estimate.get("bin_edges"), dtype=np.float64)
    if (
        aggregate.ndim != 1
        or aggregate.size < 2
        or not np.all(np.isfinite(aggregate))
        or not np.all(np.diff(aggregate) > 0.0)
    ):
        raise ValueError("aggregate density bin edges are invalid")
    seed_edges = []
    for result in _ordered_seed_results(packet):
        observable = _required_mapping(
            _required_mapping(_required_mapping(result, "pure_walking"), "observable_results"),
            "density",
        )
        metadata = _required_mapping(observable, "metadata")
        edges = np.asarray(metadata.get("bin_edges"), dtype=np.float64)
        if edges.shape != aggregate.shape or not np.all(np.isfinite(edges)):
            raise ValueError("seed density bin edges are invalid")
        seed_edges.append(edges)
    return aggregate, seed_edges


def _verify_aggregate_matches_seed_mean(
    packet: LoadedBenchmarkPacket,
    observable: str,
    seed_values: FloatArray,
) -> None:
    reported = np.asarray(_estimate(packet, observable).get("value"), dtype=np.float64)
    expected = np.mean(seed_values, axis=0)
    if reported.shape != expected.shape or not np.allclose(
        reported,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(f"reported {observable} estimate does not equal its seed aggregate")


def _ordered_seed_results(packet: LoadedBenchmarkPacket) -> list[dict[str, Any]]:
    raw = packet.summary.get("seed_results")
    if not isinstance(raw, list) or not all(isinstance(value, dict) for value in raw):
        raise ValueError("benchmark packet has invalid seed results")
    by_seed = {value.get("seed"): value for value in raw}
    if len(by_seed) != len(raw) or set(by_seed) != set(packet.seeds):
        raise ValueError("benchmark seed payload identities disagree")
    return [by_seed[seed] for seed in packet.seeds]


def _observable_result(packet: LoadedBenchmarkPacket, observable: str) -> dict[str, Any]:
    pure = _required_mapping(packet.summary, "pure_walking")
    observables = _required_mapping(pure, "observables")
    return _required_mapping(observables, observable)


def _estimate(packet: LoadedBenchmarkPacket, observable: str) -> dict[str, Any]:
    estimates = _required_mapping(packet.summary, "estimates")
    return _required_mapping(estimates, observable)


def _density_estimator(packet: LoadedBenchmarkPacket) -> object:
    return _estimate(packet, "density").get("estimator")


def _r2_estimator(packet: LoadedBenchmarkPacket) -> object:
    return _estimate(packet, "r2").get("estimator")


def _guide_identity(packet: LoadedBenchmarkPacket) -> dict[str, Any]:
    guide = _required_mapping(packet.summary, "guide_parameters")
    return {
        "relative_alpha": guide.get("relative_alpha"),
        "contact_beta": guide.get("contact_beta"),
        "source_sha256": guide.get("source_sha256"),
        "source_manifest_sha256": guide.get("source_manifest_sha256"),
        "source_identity_fingerprint": guide.get("source_identity_fingerprint"),
    }


def _scientific_identity(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    summary = anchors.density.summary
    return {
        "case_id": anchors.density.case_id,
        "n_particles": summary.get("n_particles"),
        "rod_length_ho": summary.get("rod_length_ho"),
        "coordinate": summary.get("coordinate"),
        "length_unit": summary.get("length_unit"),
        "time_unit": summary.get("time_unit"),
        "guide_family": summary.get("guide_family"),
        "guide_parameters": _guide_identity(anchors.density),
        "source_tree_sha256": anchors.density.source_tree_sha256,
        "seed_ids": list(anchors.density.seeds),
        "density_estimator": _density_estimator(anchors.density),
        "r2_estimator": _r2_estimator(anchors.r2),
        "density_source": anchors.density.pure_config.get("density_source"),
        "density_parity_average": anchors.density.pure_config.get("density_parity_average"),
        "r2_source": anchors.r2.pure_config.get("observable_source"),
        "candidate_source_tree_sha256": candidate.source_tree_sha256,
    }


def build_fw_sampling_design(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    """Derive the collection-cadence design from bound FW treatments."""

    packets = {
        "anchor_density": anchors.density,
        "anchor_r2": anchors.r2,
        "candidate": candidate,
    }
    scheduled_move_enabled = {
        label: packet.summary.get("collective_rn_controls") is not None
        for label, packet in packets.items()
    }
    any_scheduled_move = any(scheduled_move_enabled.values())
    r2 = _cadence_comparison(anchors.r2, candidate, observable="r2")
    density = _cadence_comparison(anchors.density, candidate, observable="density")
    anchor_r2_composition_common = _same_float(
        _stride_tau(anchors.density, "collection_stride_steps"),
        _stride_tau(anchors.r2, "collection_stride_steps"),
    )
    cadence_varied = bool(
        r2["status"] == "varied_cadence"
        or density["status"] == "varied_cadence"
        or not anchor_r2_composition_common
    )
    phase_safe = not any_scheduled_move or not cadence_varied
    status = (
        "scheduled_move_phase_unsafe"
        if not phase_safe
        else "varied_cadence"
        if cadence_varied
        else "common_cadence"
    )
    return {
        "status": status,
        "phase_safe": phase_safe,
        "phase_policy": (
            "scheduled collective moves require exact physical collection cadence"
            if any_scheduled_move
            else "ordinary local DMC permits deterministic source-window subsampling"
        ),
        "source_phase": "production_event_index_mod_stride_zero",
        "scheduled_collective_move_enabled": scheduled_move_enabled,
        "all_treatments_use_ordinary_local_dmc": not any_scheduled_move,
        "anchor_r2_composition_common_cadence": anchor_r2_composition_common,
        "r2": r2,
        "density": density,
        "interpretation": (
            "Collection cadence selects transported source windows and changes sampling cost "
            "and correlation, not the fixed physical-lag estimator. Publication acceptance "
            "still requires the declared collected-window, plateau, genealogy, and "
            "independent-seed equivalence checks."
        ),
    }


def _cadence_comparison(
    anchor: LoadedBenchmarkPacket,
    candidate: LoadedBenchmarkPacket,
    *,
    observable: str,
) -> dict[str, Any]:
    anchor_record = _cadence_record(anchor, observable=observable)
    candidate_record = _cadence_record(candidate, observable=observable)
    anchor_tau = float(anchor_record["physical_stride_tau"])
    candidate_tau = float(candidate_record["physical_stride_tau"])
    common = _same_float(anchor_tau, candidate_tau)
    return {
        "status": "common_cadence" if common else "varied_cadence",
        "candidate_to_anchor_physical_stride_ratio": candidate_tau / anchor_tau,
        "anchor": anchor_record,
        "candidate": candidate_record,
    }


def _cadence_record(packet: LoadedBenchmarkPacket, *, observable: str) -> dict[str, Any]:
    stride_steps = _observable_stride_steps(packet, observable=observable)
    try:
        selected_lags = _selected_window_lags(packet, observable)
    except ValueError:
        selected_lags = []
    support_counts = _source_window_support_counts(
        packet,
        observable=observable,
        step_lags=selected_lags,
    )
    return {
        "stride_steps": stride_steps,
        "physical_stride_tau": float(stride_steps) * packet.dt,
        "selected_step_lags": selected_lags,
        "selected_physical_lags": [float(lag) * packet.dt for lag in selected_lags],
        "source_window_support_counts_by_step_lag": support_counts,
        "minimum_selected_source_window_count": min(support_counts.values(), default=0),
        "required_minimum_source_window_count": packet.pure_config.get("min_block_count"),
    }


def _source_window_support_counts(
    packet: LoadedBenchmarkPacket,
    *,
    observable: str,
    step_lags: list[int],
) -> dict[str, int]:
    if not step_lags:
        return {}
    result = _observable_result(packet, observable)
    diagnostics = _required_mapping(result, "aggregate_plateau_diagnostics")
    support = _required_mapping(diagnostics, "lag_support")
    counts: dict[str, int] = {}
    for lag in step_lags:
        lag_support = _required_mapping(support, str(lag))
        value = lag_support.get("min_block_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{observable} lag {lag} source-window count is invalid")
        counts[str(lag)] = value
    return counts


def _observable_stride_steps(packet: LoadedBenchmarkPacket, *, observable: str) -> int:
    field = (
        "density_collection_stride_steps" if observable == "density" else "collection_stride_steps"
    )
    value = packet.pure_config.get(field)
    if value is None and observable == "density":
        value = packet.pure_config.get("collection_stride_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must resolve to a positive integer")
    return value


def _treatment_record(packet: LoadedBenchmarkPacket) -> dict[str, Any]:
    return {
        "dt": packet.dt,
        "walkers": packet.walkers,
        "burn_tau": packet.controls.get("burn_tau"),
        "production_tau": packet.controls.get("production_tau"),
        "r2_collection_stride_tau": _stride_tau(packet, "collection_stride_steps"),
        "density_collection_stride_tau": _optional_stride_tau(
            packet,
            "density_collection_stride_steps",
        ),
        "proposal_telemetry": summarize_seed_proposal_telemetry(
            packet.summary.get("seed_results"),
            expected_seed_ids=packet.seeds,
        ),
    }


def _verification_warnings(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> list[str]:
    records: list[str] = []
    seen: set[tuple[Path, str]] = set()
    for label, packet in (
        ("anchor_density", anchors.density),
        ("anchor_r2", anchors.r2),
        ("candidate", candidate),
    ):
        for warning in packet.verification_warnings:
            key = (packet.manifest_path, warning)
            if key not in seen:
                records.append(f"{label}: {warning}")
                seen.add(key)
    return records


def _unresolved_reasons(
    *,
    status: str,
    input_reasons: list[str],
    grid_assessment: Mapping[str, Any],
    plateau_assessment: Mapping[str, Any],
    genealogy_assessment: Mapping[str, Any],
    comparison_error: str | None,
) -> list[str]:
    if status in ACCEPTED_FW_STATUSES:
        return []
    if status == "input_quality_unresolved":
        return input_reasons or ([comparison_error] if comparison_error else [status])
    if status == "density_grid_incompatible":
        return list(grid_assessment.get("reasons", []))
    if status == "plateau_unresolved":
        return list(plateau_assessment.get("reasons", []))
    if status == "genealogy_unresolved":
        return list(genealogy_assessment.get("reasons", []))
    return ["paired R2/RMS or exact-grid density sensitivity exceeds its margin"]


def _write_observable_table(
    path: Path,
    comparison: ForwardWalkingSensitivityResult | None,
) -> Path:
    fields = [
        "observable",
        "metric",
        "observed",
        "simultaneous_upper_bound",
        "equivalence_margin",
        "equivalent",
    ]
    rows: list[dict[str, Any]] = []
    if comparison is not None:
        rows = [
            {
                "observable": "r2",
                "metric": "paired_absolute_difference",
                "observed": comparison.r2_equivalence.observed_max_difference,
                "simultaneous_upper_bound": comparison.r2_equivalence.simultaneous_upper_bound,
                "equivalence_margin": comparison.r2_equivalence.equivalence_margin,
                "equivalent": comparison.r2_equivalence.equivalent,
            },
            {
                "observable": "rms_radius",
                "metric": "paired_absolute_difference",
                "observed": comparison.rms_equivalence.observed_max_difference,
                "simultaneous_upper_bound": comparison.rms_equivalence.simultaneous_upper_bound,
                "equivalence_margin": comparison.rms_equivalence.equivalence_margin,
                "equivalent": comparison.rms_equivalence.equivalent,
            },
            {
                "observable": "density",
                "metric": "paired_bin_width_weighted_relative_l2",
                "observed": comparison.density_equivalence.observed_max_relative_norm,
                "simultaneous_upper_bound": (
                    comparison.density_equivalence.simultaneous_upper_bound
                ),
                "equivalence_margin": comparison.density_equivalence.equivalence_margin,
                "equivalent": comparison.density_equivalence.equivalent,
            },
            {
                "observable": "fixed_cell_envelope",
                "metric": "descriptive_aggregate_cell_width_weighted_relative_l2",
                "observed": comparison.aggregate_envelope_relative_l2,
                "simultaneous_upper_bound": None,
                "equivalence_margin": None,
                "equivalent": None,
            },
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_shell_table(
    path: Path,
    comparison: ForwardWalkingSensitivityResult | None,
) -> Path:
    fields = [
        "shell_cell",
        "left",
        "right",
        "center",
        "anchor_envelope",
        "candidate_envelope",
        "anchor_peak_position",
        "candidate_peak_position",
        "peak_position_shift_in_cell_widths",
        "anchor_peak_amplitude",
        "candidate_peak_amplitude",
        "peak_amplitude_relative_difference",
    ]
    rows: list[dict[str, Any]] = []
    if comparison is not None:
        shell = comparison.shell_comparison
        for index in range(shell.centers.size):
            rows.append(
                {
                    "shell_cell": index + 1,
                    "left": shell.boundaries[index],
                    "right": shell.boundaries[index + 1],
                    "center": shell.centers[index],
                    "anchor_envelope": shell.anchor_envelope[index],
                    "candidate_envelope": shell.candidate_envelope[index],
                    "anchor_peak_position": shell.anchor_peak_positions[index],
                    "candidate_peak_position": shell.candidate_peak_positions[index],
                    "peak_position_shift_in_cell_widths": (
                        shell.peak_position_shift_in_cell_widths[index]
                    ),
                    "anchor_peak_amplitude": shell.anchor_peak_amplitudes[index],
                    "candidate_peak_amplitude": shell.candidate_peak_amplitudes[index],
                    "peak_amplitude_relative_difference": (
                        shell.peak_amplitude_relative_difference[index]
                    ),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_output_directory(
    root: Path,
    *,
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> None:
    input_directories = {
        anchors.assembly_manifest_path.parent,
        anchors.density.summary_path.parent,
        anchors.r2.summary_path.parent,
        candidate.summary_path.parent,
    }
    if root in input_directories:
        raise ValueError("FW sensitivity output directory must not contain an input artifact")
    for name in (
        "summary.json",
        "observable_comparison.csv",
        "shell_comparison.csv",
        "run_manifest.json",
    ):
        if (root / name).exists():
            raise FileExistsError(f"FW sensitivity output artifact already exists: {root / name}")


def _validate_analysis_controls(
    *,
    rms_relative_margin: float,
    density_relative_l2_margin: float,
    confidence_level: float,
) -> None:
    for name, value in (
        ("rms_relative_margin", rms_relative_margin),
        ("density_relative_l2_margin", density_relative_l2_margin),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if rms_relative_margin >= 1.0:
        raise ValueError("rms_relative_margin must be smaller than one")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")


def _validate_anchor_analysis_policy(
    anchors: AnchorSources,
    *,
    rms_relative_margin: float,
    density_relative_l2_margin: float,
    confidence_level: float,
) -> None:
    expected = (
        (
            "rms_relative_margin",
            rms_relative_margin,
            anchors.r2.pure_config.get("rms_plateau_relative_tolerance"),
        ),
        (
            "density_relative_l2_margin",
            density_relative_l2_margin,
            anchors.density.pure_config.get("density_plateau_relative_l2_tolerance"),
        ),
        (
            "confidence_level",
            confidence_level,
            anchors.r2.pure_config.get("plateau_equivalence_confidence_level"),
        ),
    )
    for name, supplied, reference in expected:
        if not _same_float(supplied, reference):
            raise ValueError(f"{name} must match the established final-matrix FW policy")


def _resolve_reference(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("source reference path is invalid")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("source reference paths must be relative to the assembly directory")
    return (root / path).resolve()


def _stride_tau(packet: LoadedBenchmarkPacket, field: str) -> float:
    value = packet.pure_config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return float(value) * packet.dt


def _optional_stride_tau(packet: LoadedBenchmarkPacket, field: str) -> float | None:
    value = packet.pure_config.get(field)
    if value is None:
        return None
    return _stride_tau(packet, field)


def _contains_physical_lags(available: object, required: object) -> bool:
    if not isinstance(available, list) or not isinstance(required, list):
        return False
    available_values = [float(value) for value in available]
    return all(
        any(
            math.isclose(float(value), other, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for other in available_values
        )
        for value in required
    )


def _same_physical_lag_sequence(first: object, second: object) -> bool:
    if not isinstance(first, list) or not isinstance(second, list) or len(first) != len(second):
        return False
    return all(
        _same_float(first_value, second_value)
        for first_value, second_value in zip(first, second, strict=True)
    )


def _same_float(first: object, second: object) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return False
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return False
    return math.isclose(float(first), float(second), rel_tol=1.0e-12, abs_tol=1.0e-12)


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{name} must be a list of non-negative integer step lags")
    return list(value)


def _optional_integer_list(value: object) -> list[int] | None:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        return None
    return list(value)


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_mapping(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value
