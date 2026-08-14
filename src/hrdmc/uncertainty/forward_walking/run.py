from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.artifacts.manifest import config_fingerprint
from hrdmc.statistics.equivalence import (
    simultaneous_pairwise_equivalence,
    simultaneous_pairwise_norm_equivalence,
)
from hrdmc.statistics.fw_sensitivity import (
    ForwardWalkingSensitivityResult,
    analyze_fw_observable_sensitivity,
    classify_fw_sensitivity_status,
)
from hrdmc.uncertainty.forward_walking.outputs import (
    empty_artifacts as _empty_artifacts,
)
from hrdmc.uncertainty.forward_walking.outputs import (
    write_fw_artifacts as _write_fw_artifacts,
)
from hrdmc.uncertainty.forward_walking.sources import (
    AnchorSources,
    LoadedBenchmarkPacket,
    load_final_matrix_anchor_sources,
    load_manifest_bound_benchmark_packet,
)
from hrdmc.uncertainty.forward_walking.sources import (
    mapping as _mapping,
)
from hrdmc.uncertainty.forward_walking.sources import (
    positive as _positive,
)
from hrdmc.uncertainty.forward_walking.sources import (
    positive_int as _positive_int,
)
from hrdmc.uncertainty.forward_walking.sources import (
    positive_or_zero as _positive_or_zero,
)
from hrdmc.uncertainty.stationarity import (
    load_energy_stationarity_selection,
)

FloatArray = NDArray[np.float64]
ACCEPTED_FW_STATUSES = {"accepted", "accepted_with_warnings"}
def run_fw_sensitivity_workflow(
    final_matrix_manifest_path: Path,
    candidate_summary_path: Path,
    *,
    case_id: str,
    output_dir: Path | None,
    energy_assessment_manifest: Path | None = None,
    command: list[str] | None = None,
    write_artifacts: bool = True,
    rms_relative_margin: float = 0.001,
    density_relative_l2_margin: float = 0.03,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    _validate_controls(rms_relative_margin, density_relative_l2_margin, confidence_level)
    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when artifacts are written")
    anchors = load_final_matrix_anchor_sources(final_matrix_manifest_path, case_id=case_id)
    candidate = load_manifest_bound_benchmark_packet(candidate_summary_path)
    if candidate.case_id != case_id:
        raise ValueError("candidate has the wrong case")
    _require_shared_physics(anchors, candidate)
    _validate_controls(
        rms_relative_margin,
        density_relative_l2_margin,
        confidence_level,
        anchor_treatment=(anchors.density.dt, anchors.density.walkers),
        candidate_treatment=(candidate.dt, candidate.walkers),
    )
    energy_selection = (
        None
        if energy_assessment_manifest is None
        else load_energy_stationarity_selection(
            energy_assessment_manifest,
            case_id=case_id,
            selected_summary_path=candidate.summary_path,
        )
    )
    input_reasons = _packet_quality_reasons((anchors.density, anchors.r2, candidate))
    if candidate.summary.get("energy_validation_status") != "accepted" and not (
        isinstance(energy_selection, dict) and energy_selection.get("publication_accepted") is True
    ):
        input_reasons.append("candidate_energy_not_accepted")
    grid_reasons = _grid_reasons(anchors.density, candidate)
    plateau_assessment = _assess_plateaus(anchors, candidate)
    plateau_reasons = plateau_assessment["reasons"]
    genealogy_assessment = _assess_genealogy(anchors, candidate)
    genealogy_reasons = genealogy_assessment["reasons"]
    comparison: ForwardWalkingSensitivityResult | None = None
    comparison_error: str | None = None
    if not (input_reasons or grid_reasons or plateau_reasons or genealogy_reasons):
        try:
            comparison = _compare(
                anchors,
                candidate,
                rms_relative_margin,
                density_relative_l2_margin,
                confidence_level,
            )
        except ValueError as exc:
            comparison_error = str(exc)
            input_reasons.append(f"observable payload: {exc}")
    warnings = [
        *anchors.density.verification_warnings,
        *anchors.r2.verification_warnings,
        *candidate.verification_warnings,
    ]
    status = classify_fw_sensitivity_status(
        input_quality_accepted=not input_reasons,
        density_grid_compatible=not grid_reasons,
        plateau_resolved=not plateau_reasons,
        genealogy_supported=not genealogy_reasons,
        observables_equivalent=comparison is not None and comparison.equivalent,
        has_warnings=bool(warnings),
    )
    identity = _identity(anchors, candidate)
    payload: dict[str, Any] = {
        "schema_version": "dmc_fw_sensitivity_v2",
        "status": status,
        "case_id": case_id,
        "identity": identity,
        "identity_fingerprint": config_fingerprint(identity),
        "rms_relative_margin": rms_relative_margin,
        "density_relative_l2_margin": density_relative_l2_margin,
        "confidence_level": confidence_level,
        "treatments": {
            "anchor_density": anchors.density.reference(),
            "anchor_r2": anchors.r2.reference(),
            "candidate": candidate.reference(),
        },
        "proposal_telemetry": {
            "anchor": _proposal_telemetry(anchors.density),
            "selected_candidate": _proposal_telemetry(candidate),
        },
        "sampling_design": build_fw_sampling_design(anchors, candidate),
        "input_quality": {
            "status": "accepted" if not input_reasons else "unresolved",
            "reasons": input_reasons,
            "checks": {
                "source_energy_accepted": candidate.summary.get("energy_validation_status")
                == "accepted",
                "candidate_family_energy_assessment_accepted": isinstance(energy_selection, dict)
                and energy_selection.get("publication_accepted") is True,
                "input_quality_requirements_met": not input_reasons,
            },
            "manifest_verification_warnings": warnings,
        },
        "density_grid": {"compatible": not grid_reasons, "reasons": grid_reasons},
        "plateau_assessment": plateau_assessment,
        "genealogy_assessment": genealogy_assessment,
        "observable_comparison": None if comparison is None else comparison.to_dict(),
        "comparison_error": comparison_error,
        "publication_ready_within_fw_sensitivity_scope": status in ACCEPTED_FW_STATUSES,
        "qualified_systematics": {
            "forward_walking_timestep_population_sensitivity": (
                "accepted" if status in ACCEPTED_FW_STATUSES else "unresolved"
            )
        },
        "unresolved_reasons": [
            *input_reasons,
            *grid_reasons,
            *plateau_reasons,
            *genealogy_reasons,
            *([] if comparison_error is None else [comparison_error]),
        ],
        "candidate_energy_assessment": energy_selection,
    }
    if write_artifacts:
        assert output_dir is not None
        artifacts = _write_fw_artifacts(output_dir, payload, command)
    else:
        artifacts = _empty_artifacts(output_dir)
    payload["workflow_artifacts"] = artifacts
    return payload
def build_fw_sampling_design(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
) -> dict[str, Any]:
    scheduled = any(
        packet.summary.get("collective_rn_controls") is not None
        for packet in (anchors.density, anchors.r2, candidate)
    )
    r2 = _cadence_comparison(anchors.r2, candidate, "r2")
    density = _cadence_comparison(anchors.density, candidate, "density")
    varied = r2["status"] == "varied_cadence" or density["status"] == "varied_cadence"
    return {
        "status": "scheduled_move_phase_unsafe"
        if scheduled and varied
        else ("varied_cadence" if varied else "common_cadence"),
        "phase_safe": not (scheduled and varied),
        "paired_seed_ids": list(candidate.seeds),
        "all_treatments_use_ordinary_local_dmc": not scheduled,
        "r2": r2,
        "density": density,
    }


def _compare(
    anchors: AnchorSources,
    candidate: LoadedBenchmarkPacket,
    rms_margin: float,
    density_margin: float,
    confidence: float,
) -> ForwardWalkingSensitivityResult:
    anchor_r2 = _seed_scalar(anchors.r2, "r2")
    candidate_r2 = _seed_scalar(candidate, "r2")
    anchor_density = _seed_density(anchors.density)
    candidate_density = _seed_density(candidate)
    edges = _density_bin_edges(anchors.density)
    return analyze_fw_observable_sensitivity(
        anchor_r2_by_seed=anchor_r2,
        candidate_r2_by_seed=candidate_r2,
        bin_edges=edges,
        anchor_density_by_seed=anchor_density,
        candidate_density_by_seed=candidate_density,
        particle_count=_positive_int(anchors.density.summary.get("n_particles"), "particles"),
        rms_relative_margin=rms_margin,
        density_relative_l2_margin=density_margin,
        confidence_level=confidence,
    )


def _seed_scalar(packet: LoadedBenchmarkPacket, observable: str) -> FloatArray:
    rows = _seed_results(packet)
    values = [_positive(_seed_plateau(packet, row, observable), observable) for row in rows]
    return np.asarray(values, dtype=np.float64)


def _seed_density(packet: LoadedBenchmarkPacket) -> FloatArray:
    return np.asarray(
        [_seed_plateau(packet, row, "density") for row in _seed_results(packet)],
        dtype=np.float64,
    )


def _seed_plateau(packet: LoadedBenchmarkPacket, row: dict[str, Any], name: str) -> Any:
    result = _seed_observable(row, name)
    if result.get("plateau_value") is not None:
        return result["plateau_value"]
    values = _mapping(result.get("values_by_lag"), f"{name} values_by_lag")
    diagnostics = _mapping(
        _observable(packet, name).get("aggregate_plateau_diagnostics"),
        f"{name} plateau diagnostics",
    )
    lags = diagnostics.get("selected_window_lags")
    if not isinstance(lags, list) or not lags:
        raise ValueError(f"{name} has no selected plateau window")
    selected = [values.get(lag, values.get(str(lag))) for lag in lags]
    if any(value is None for value in selected):
        raise ValueError(f"{name} plateau window is incomplete")
    return np.mean(np.asarray(selected, dtype=np.float64), axis=0)


def _seed_results(packet: LoadedBenchmarkPacket) -> list[dict[str, Any]]:
    rows = packet.summary.get("seed_results")
    if not isinstance(rows, list):
        raise ValueError(f"seed results are missing: {packet.summary_path}")
    by_seed = {row.get("seed"): row for row in rows if isinstance(row, dict)}
    try:
        return [by_seed[seed] for seed in packet.seeds]
    except KeyError as exc:
        raise ValueError("seed result identities are incomplete") from exc


def _proposal_telemetry(packet: LoadedBenchmarkPacket) -> dict[str, Any]:
    rows = _seed_results(packet)
    metadata = []
    for row in rows:
        dmc = row.get("dmc_summary")
        value = dmc.get("metadata") if isinstance(dmc, dict) else None
        if not isinstance(value, dict):
            return {"status": "unavailable", "seed_count": len(rows)}
        metadata.append(value)
    mean_fields = ("local_acceptance_fraction_mean", "configuration_esjd_mean")
    max_fields = ("invalid_proposal_fraction_max", "metropolis_rejection_fraction_max")
    result: dict[str, Any] = {"status": "available", "seed_count": len(rows)}
    for field in mean_fields:
        values = [_positive_or_zero(row.get(field), field) for row in metadata]
        result[field] = float(np.mean(values))
    for field in max_fields:
        values = [_positive_or_zero(row.get(field), field) for row in metadata]
        result[field] = float(np.max(values))
    return result


def _seed_observable(row: dict[str, Any], name: str) -> dict[str, Any]:
    pure = _mapping(row.get("pure_walking"), "seed pure_walking")
    results = _mapping(pure.get("observable_results"), "observable_results")
    return _mapping(results.get(name), name)


def _observable(packet: LoadedBenchmarkPacket, name: str) -> dict[str, Any]:
    pure = _mapping(packet.summary.get("pure_walking"), "pure_walking")
    return _mapping(_mapping(pure.get("observables"), "observables").get(name), name)


def _packet_quality_reasons(packets: tuple[LoadedBenchmarkPacket, ...]) -> list[str]:
    selected = (
        (packets[0], "density"),
        (packets[1], "r2"),
        (packets[2], "density"),
        (packets[2], "r2"),
    )
    return [
        f"{packet.summary_path}:{name}_not_accepted"
        for packet, name in selected
        if _observable(packet, name).get("status") not in {None, "accepted"}
    ]


def _grid_reasons(anchor: LoadedBenchmarkPacket, candidate: LoadedBenchmarkPacket) -> list[str]:
    first = _density_bin_edges(anchor)
    second = _density_bin_edges(candidate)
    return (
        []
        if first.shape == second.shape and np.array_equal(first, second)
        else ["density_grid_mismatch"]
    )


def _density_bin_edges(packet: LoadedBenchmarkPacket) -> FloatArray:
    estimates = _mapping(packet.summary.get("estimates"), "estimates")
    density = _mapping(estimates.get("density"), "density estimate")
    edges = np.asarray(density.get("bin_edges"), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("density bin edges are invalid")
    return edges


def _plateau_reasons(packets: tuple[LoadedBenchmarkPacket, ...]) -> list[str]:
    selected = (
        (packets[0], "density"),
        (packets[1], "r2"),
        (packets[2], "density"),
        (packets[2], "r2"),
    )
    return [
        f"{packet.summary_path}:{name}_plateau"
        for packet, name in selected
        if _observable(packet, name).get("aggregate_plateau_status") != "plateau_resolved"
    ]


def _assess_plateaus(anchors: AnchorSources, candidate: LoadedBenchmarkPacket) -> dict[str, Any]:
    reasons = _plateau_reasons((anchors.density, anchors.r2, candidate))
    bridges: dict[str, Any] = {}
    for name, anchor in (("r2", anchors.r2), ("density", anchors.density)):
        anchor_lags = _selected_lags(anchor, name)
        candidate_lags = _selected_lags(candidate, name)
        comparison = [
            lag
            for lag in _requested_lags(candidate, name)
            if any(math.isclose(lag * candidate.dt, target * anchor.dt) for target in anchor_lags)
        ]
        if len(comparison) != len(anchor_lags):
            bridge = {"status": "anchor_window_unavailable", "equivalent": False}
        elif comparison == candidate_lags:
            bridge = {"status": "not_required", "equivalent": True}
        else:
            bridge = _bridge(candidate, name, comparison, candidate_lags)
        bridge.update(
            anchor_window_step_lags=comparison,
            candidate_selected_step_lags=candidate_lags,
        )
        bridges[name] = bridge
        if bridge["equivalent"] is not True:
            reasons.append(f"candidate_{name}: plateau bridge unresolved")
    return {
        "status": "accepted" if not reasons else "unresolved",
        "resolved": not reasons,
        "reasons": reasons,
        "bridge_equivalence": bridges,
    }


def _bridge(
    packet: LoadedBenchmarkPacket, name: str, anchor_lags: list[int], selected: list[int]
) -> dict[str, Any]:
    combined = sorted(set(anchor_lags + selected))
    values = _seed_ladder(packet, name, combined)
    confidence = float(packet.pure_config["plateau_equivalence_confidence_level"])
    if name == "r2":
        rms = np.sqrt(values)
        scale = float(np.sqrt(np.mean(values)))
        margin = float(packet.pure_config["rms_plateau_relative_tolerance"]) * scale
        result = simultaneous_pairwise_equivalence(
            rms, equivalence_margin=margin, confidence_level=confidence
        )
    else:
        edges = _density_bin_edges(packet)
        result = simultaneous_pairwise_norm_equivalence(
            values,
            feature_weights=np.diff(edges),
            scale_vector=np.mean(values, axis=(0, 1)),
            equivalence_margin=float(packet.pure_config["density_plateau_relative_l2_tolerance"]),
            confidence_level=confidence,
        )
    return {
        "status": "accepted" if result.equivalent else "unresolved",
        "equivalent": bool(result.equivalent),
    }


def _assess_genealogy(anchors: AnchorSources, candidate: LoadedBenchmarkPacket) -> dict[str, Any]:
    reasons: list[str] = []
    for label, packet, name in (
        ("anchor_density", anchors.density, "density"),
        ("anchor_r2", anchors.r2, "r2"),
        ("candidate_density", candidate, "density"),
        ("candidate_r2", candidate, "r2"),
    ):
        diagnostics = _mapping(
            _observable(packet, name).get("aggregate_plateau_diagnostics"), "diagnostics"
        )
        if _observable(packet, name).get("aggregate_genealogy_status") != (
            "genealogy_support_accepted"
        ):
            reasons.append(f"{label}: aggregate genealogy status is unresolved")
        support = _mapping(diagnostics.get("lag_support"), "lag_support")
        for lag in _selected_lags(packet, name):
            row = _mapping(support.get(str(lag)), "lag support")
            checks = (
                (
                    _positive(row.get("required_pooled_ancestor_ess"), "required ESS"),
                    _positive(packet.pure_config.get("min_source_ancestor_ess"), "configured ESS"),
                    "unbound ancestor-ESS threshold",
                ),
                (
                    _positive(row.get("maximum_pooled_family_fraction"), "maximum family"),
                    _positive(
                        packet.pure_config.get("max_source_family_fraction"), "configured family"
                    ),
                    "unbound family-fraction threshold",
                ),
            )
            for observed, configured, message in checks:
                if not math.isclose(float(observed), float(configured)):
                    reasons.append(f"{label}: selected lag {lag} uses an {message}")
            if _positive(row.get("pooled_ancestor_ess_lower_bound"), "ancestor ESS") < checks[0][0]:
                reasons.append(f"{label}: selected lag {lag} has insufficient pooled ancestor ESS")
            if (
                _positive_or_zero(row.get("pooled_family_fraction_upper_bound"), "family")
                > checks[1][0]
            ):
                reasons.append(f"{label}: selected lag {lag} exceeds pooled family concentration")
            if _positive(row.get("min_walker_weight_ess"), "walker ESS") < _positive(
                packet.pure_config.get("min_walker_weight_ess"), "configured walker ESS"
            ):
                reasons.append(f"{label}: selected lag {lag} has insufficient walker-weight ESS")
            if _positive_int(row.get("min_block_count"), "block count") < _positive_int(
                packet.pure_config.get("min_block_count"), "configured block count"
            ):
                reasons.append(
                    f"{label}: selected lag {lag} has insufficient collected source windows"
                )
    return {
        "status": "accepted" if not reasons else "unresolved",
        "supported": not reasons,
        "reasons": reasons,
    }


def _selected_lags(packet: LoadedBenchmarkPacket, name: str) -> list[int]:
    diagnostics = _mapping(
        _observable(packet, name).get("aggregate_plateau_diagnostics"), "diagnostics"
    )
    value = diagnostics.get("selected_window_lags")
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{name} selected lags are invalid")
    return value


def _requested_lags(packet: LoadedBenchmarkPacket, name: str) -> list[int]:
    key = "density_lag_steps" if name == "density" else "lag_steps"
    value = packet.pure_config.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{name} requested lags are invalid")
    return value


def _seed_ladder(packet: LoadedBenchmarkPacket, name: str, lags: list[int]) -> FloatArray:
    return np.asarray(
        [
            [
                _mapping(_seed_observable(row, name).get("values_by_lag"), "values_by_lag")[
                    str(lag)
                ]
                for lag in lags
            ]
            for row in _seed_results(packet)
        ],
        dtype=np.float64,
    )


def _require_shared_physics(anchors: AnchorSources, candidate: LoadedBenchmarkPacket) -> None:
    packets = (anchors.density, anchors.r2, candidate)
    if any(packet.seeds != packets[0].seeds for packet in packets[1:]):
        raise ValueError("FW treatments must use matched seed identities")
    for field in ("n_particles", "rod_length", "guide_family"):
        if any(
            packet.summary.get(field) != packets[0].summary.get(field) for packet in packets[1:]
        ):
            raise ValueError(f"FW treatment {field} mismatch")


def _identity(anchors: AnchorSources, candidate: LoadedBenchmarkPacket) -> dict[str, Any]:
    return {
        "case_id": candidate.case_id,
        "n_particles": candidate.summary.get("n_particles"),
        "rod_length_ho": candidate.summary.get(
            "rod_length_ho", candidate.summary.get("rod_length")
        ),
        "guide_family": candidate.summary.get("guide_family"),
        "guide_parameters": {
            "relative_alpha": candidate.controls.get("relative_alpha"),
        },
        "seed_ids": list(candidate.seeds),
        "anchor_density_run_id": anchors.density.manifest.get("run_id"),
        "anchor_r2_run_id": anchors.r2.manifest.get("run_id"),
        "candidate_run_id": candidate.manifest.get("run_id"),
    }


def _cadence(packet: LoadedBenchmarkPacket, observable: str) -> dict[str, Any]:
    config = packet.pure_config
    prefix = "density_" if observable == "density" else ""
    lags = config.get(f"{prefix}lag_steps", config.get("lag_steps"))
    stride = config.get(f"{prefix}collection_stride_steps", config.get("collection_stride_steps"))
    try:
        selected = _selected_lags(packet, observable)
        diagnostics = _mapping(
            _observable(packet, observable).get("aggregate_plateau_diagnostics"),
            "diagnostics",
        )
        support = _mapping(diagnostics.get("lag_support"), "lag_support")
        counts = {
            str(lag): _positive_int(
                _mapping(support.get(str(lag)), "support").get("min_block_count"),
                "block count",
            )
            for lag in selected
        }
    except ValueError:
        selected, counts = [], {}
    return {
        "dt": packet.dt,
        "walkers": packet.walkers,
        "stride_steps": stride,
        "lag_steps": lags,
        "collection_stride_steps": stride,
        "physical_lags": [float(lag) * packet.dt for lag in lags]
        if isinstance(lags, list)
        else None,
        "minimum_selected_source_window_count": min(counts.values(), default=0),
        "required_minimum_source_window_count": packet.pure_config.get("min_block_count"),
    }


def _cadence_comparison(
    anchor: LoadedBenchmarkPacket, candidate: LoadedBenchmarkPacket, observable: str
) -> dict[str, Any]:
    first, second = _cadence(anchor, observable), _cadence(candidate, observable)
    return {
        "status": "common_cadence"
        if math.isclose(
            float(first["dt"]) * int(first["stride_steps"]),
            float(second["dt"]) * int(second["stride_steps"]),
        )
        else "varied_cadence",
        "anchor": first,
        "candidate": second,
    }


def _validate_controls(
    rms: float,
    density: float,
    confidence: float,
    *,
    anchor_treatment: tuple[float, int] | None = None,
    candidate_treatment: tuple[float, int] | None = None,
) -> None:
    if not (math.isfinite(rms) and 0.0 < rms < 1.0):
        raise ValueError("rms_relative_margin must lie in (0,1)")
    if not (math.isfinite(density) and density > 0.0):
        raise ValueError("density_relative_l2_margin must be positive")
    if not (math.isfinite(confidence) and 0.0 < confidence < 1.0):
        raise ValueError("confidence_level must lie in (0,1)")
    if anchor_treatment is not None and candidate_treatment == anchor_treatment:
        raise ValueError("candidate treatment equals the anchor")
