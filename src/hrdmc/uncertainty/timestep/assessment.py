from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from hrdmc.production.matrix.assembly import (
    load_final_matrix_energy_selection,
)
from hrdmc.statistics.timestep_fit import (
    LargestTimeStepStability,
    TimeStepExtrapolation,
    absolute_difference_upper_allowance,
)
from hrdmc.uncertainty.timestep.contract import (
    EnergyReportingResolutionPolicy,
    LoadedTimeStepPoint,
)
from hrdmc.uncertainty.timestep.sources import (
    _required_int,
    _required_string,
)


def _distinct_values(values: Sequence[Any]) -> list[Any]:
    distinct: list[Any] = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    return distinct


def _practical_resolution_assessment(
    analysis: TimeStepExtrapolation,
    *,
    policy: EnergyReportingResolutionPolicy | None,
    input_quality: dict[str, Any],
    cross_timestep_covariance: dict[str, Any],
    energy_semantics: dict[str, Any],
) -> dict[str, Any]:
    separate_components = (
        "statistical stderr, model-order upper allowance, and fit-window upper "
        "allowance are reported separately and are not combined in quadrature"
    )
    if policy is None:
        return {
            "status": "not_requested",
            "accepted_with_model_bound": False,
            "statistical_classification": analysis.classification,
            "policy": None,
            "model_order": None,
            "fit_window": None,
            "curvature_diagnostic": None,
            "checks": {},
            "failed_checks": [],
            "uncertainty_component_combination": separate_components,
            "scope": "fixed-walker-population mixed-energy time-step extrapolation",
        }
    energy_unit = _required_string(energy_semantics, "energy_unit")
    report_energy_unit = _required_string(energy_semantics, "report_energy_unit")
    if policy.energy_unit != energy_unit or policy.energy_unit != report_energy_unit:
        raise ValueError(
            "energy reporting resolution unit must match the input energy and report units"
        )
    sensitivity = analysis.leading_model_sensitivity
    model_order = absolute_difference_upper_allowance(
        sensitivity.absolute_spread,
        sensitivity.comparison_uncertainty,
        confidence_level=policy.confidence_level,
    ).to_dict()
    linear_window = _fit_window_upper_allowance(
        analysis.leading_linear_largest_point_stability,
        confidence_level=policy.confidence_level,
    )
    quadratic_window = _fit_window_upper_allowance(
        analysis.leading_quadratic_largest_point_stability,
        confidence_level=policy.confidence_level,
    )
    available_window_allowances = [
        float(value["upper_allowance"])
        for value in (linear_window, quadratic_window)
        if value is not None
    ]
    fit_window_upper_allowance = (
        max(available_window_allowances) if available_window_allowances else None
    )
    fit_window = {
        "leading_linear": linear_window,
        "leading_quadratic": quadratic_window,
        "selection_rule": "maximum upper allowance across the two leading models",
        "upper_allowance": fit_window_upper_allowance,
    }
    curvature_fit = analysis.curvature_fit
    curvature_stability = analysis.curvature_largest_point_stability
    curvature_four_point_unavailable = (
        curvature_stability is not None
        and not curvature_stability.available
        and len(analysis.points) == 4
    )
    curvature_window_accepted = (
        curvature_stability is None
        or curvature_four_point_unavailable
        or (
            curvature_stability.available
            and curvature_stability.classification == "largest_point_stable"
        )
    )
    curvature_diagnostic = {
        "fit_available": curvature_fit is not None,
        "fit_goodness_of_fit_status": (
            None if curvature_fit is None else curvature_fit.goodness_of_fit_status
        ),
        "largest_point_check_available": (
            None if curvature_stability is None else curvature_stability.available
        ),
        "largest_point_classification": (
            None if curvature_stability is None else curvature_stability.classification
        ),
        "four_point_unavailable_allowed": curvature_four_point_unavailable,
        "interpretation": (
            "an unavailable curvature leave-largest-time-step-out check is allowed "
            "only for a four-point fit window; every available check must be stable"
        ),
    }
    statistically_resolved_model_difference = (
        sensitivity.classification == "model_sensitive"
        and sensitivity.absolute_spread
        > sensitivity.sensitivity_sigma * sensitivity.comparison_uncertainty
    )
    checks = {
        "analysis_is_model_sensitive": analysis.classification == "model_sensitive",
        "model_difference_is_statistically_resolved": (statistically_resolved_model_difference),
        "leading_linear_fit_adequate": (
            analysis.leading_linear_fit.goodness_of_fit_status == "accepted"
        ),
        "leading_quadratic_fit_adequate": (
            analysis.leading_quadratic_fit.goodness_of_fit_status == "accepted"
        ),
        "leading_linear_window_stable": (
            analysis.leading_linear_largest_point_stability.classification == "largest_point_stable"
        ),
        "leading_quadratic_window_stable": (
            analysis.leading_quadratic_largest_point_stability.classification
            == "largest_point_stable"
        ),
        "curvature_fit_adequate_or_absent": (
            curvature_fit is None or curvature_fit.goodness_of_fit_status == "accepted"
        ),
        "curvature_window_stable_or_four_point_unavailable": curvature_window_accepted,
        "fit_window_accepted": analysis.fit_window_status == "accepted",
        "model_order_upper_allowance_within_resolution": (
            float(model_order["upper_allowance"]) <= policy.resolution
        ),
        "fit_window_upper_allowance_within_resolution": (
            fit_window_upper_allowance is not None
            and fit_window_upper_allowance <= policy.resolution
        ),
        "input_quality_accepted": input_quality["publication_accepted"] is True,
        "cross_timestep_covariance_accepted": (cross_timestep_covariance["status"] == "accepted"),
    }
    accepted = all(checks.values())
    return {
        "status": "bounded_below_reporting_resolution" if accepted else "not_bounded",
        "accepted_with_model_bound": accepted,
        "statistical_classification": analysis.classification,
        "policy": policy.to_dict(),
        "model_order": model_order,
        "fit_window": fit_window,
        "curvature_diagnostic": curvature_diagnostic,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "uncertainty_component_combination": separate_components,
        "bound_scope": (
            "declared leading-linear and leading-quadratic models and their "
            "leave-largest-time-step-out windows; not every possible asymptotic model"
        ),
        "scope": "fixed-walker-population mixed-energy time-step extrapolation",
    }
def _fit_window_upper_allowance(
    stability: LargestTimeStepStability,
    *,
    confidence_level: float,
) -> dict[str, Any] | None:
    if (
        not stability.available
        or stability.absolute_shift is None
        or stability.comparison_uncertainty is None
    ):
        return None
    allowance = absolute_difference_upper_allowance(
        stability.absolute_shift,
        stability.comparison_uncertainty,
        confidence_level=confidence_level,
    ).to_dict()
    return {
        "model": stability.model,
        "statistical_classification": stability.classification,
        **allowance,
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
    precision_warning_rows = [
        row for row in rows if row["publication_status"] == "accepted_with_precision_warning"
    ]
    retrospective_rows = [
        row for row in rows if row["publication_status"] == "accepted_with_retrospective_assessment"
    ]
    return {
        "status": (
            "unresolved" if unresolved else "accepted_with_warnings" if warning_rows else "accepted"
        ),
        "publication_accepted": not unresolved,
        "publication_accepted_statuses": [
            "accepted",
            "accepted_with_precision_warning",
            "accepted_with_retrospective_assessment",
        ],
        "points": rows,
        "unresolved_point_count": len(unresolved),
        "warning_point_count": len(warning_rows),
        "precision_warning_point_count": len(precision_warning_rows),
        "retrospective_assessment_point_count": len(retrospective_rows),
        "interpretation": (
            "Energy-specific method failures remain unresolved unless an exact "
            "manifest-bound matrix assessment selects that source summary. Precision "
            "warnings and retrospective assessment timing remain visible but do not "
            "veto the mixed-energy extrapolation."
        ),
    }
def _apply_energy_quality_assessment(
    points: Sequence[LoadedTimeStepPoint],
    *,
    energy_assessment_manifest: Path,
) -> tuple[list[LoadedTimeStepPoint], dict[str, Any]]:
    if not points:
        raise ValueError("energy assessment requires at least one time-step point")
    case_ids = {point.case_id for point in points}
    if len(case_ids) != 1:
        raise ValueError("energy assessment requires one shared case identity")
    selection = load_final_matrix_energy_selection(
        energy_assessment_manifest,
        case_id=next(iter(case_ids)),
    )
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
    ]
    if len(matches) != 1:
        raise ValueError(
            "energy assessment must select exactly one input summary by path and digest"
        )
    selected_index = matches[0]
    selected_point = points[selected_index]
    source_quality = selected_point.energy_quality
    assessed_quality = {
        **source_quality,
        "status_basis": selection["energy_status_basis"],
        "source_publication_accepted": source_quality["publication_accepted"],
        "source_publication_status": source_quality["publication_status"],
        "publication_accepted": selection["publication_accepted"],
        "publication_status": selection["publication_status"],
    }
    updated = list(points)
    updated[selected_index] = replace(
        selected_point,
        energy_quality=assessed_quality,
        energy_quality_assessment=selection,
    )
    return updated, selection
