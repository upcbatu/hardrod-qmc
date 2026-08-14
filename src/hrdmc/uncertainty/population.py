from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.manifest import (
    config_fingerprint,
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    write_csv,
    write_json,
    write_run_manifest,
)
from hrdmc.statistics.population_bound import PopulationEnergyPoint
from hrdmc.statistics.population_fit import (
    PopulationLadderAssessment,
    TimeStepPopulationInteraction,
    analyze_population_ladder,
    analyze_timestep_population_interaction,
)

POPULATION_SYSTEMATICS_RUN_NAME = "dmc_population_systematics"
FIXED_ENERGY_REPORTING_RESOLUTION = 0.01
PUBLICATION_READY_STATUSES = {
    "accepted_finite_population_bound",
    "accepted_population_limit",
    "accepted_with_warnings",
}
_SOURCE_RUNS = {"dmc_benchmark_packet", "dmc_trapped_stationarity_grid"}


@dataclass(frozen=True)
class _LoadedPopulationPoint:
    point: PopulationEnergyPoint
    dt: float
    case_id: str
    controls: dict[str, Any]
    stationarity: dict[str, Any]
    summary_path: Path
    manifest_path: Path
    run_name: str
    run_id: str
    manifest_warnings: tuple[str, ...]

    def reference(self) -> dict[str, Any]:
        return {
            **self.point.to_dict(),
            "dt": self.dt,
            "case_id": self.case_id,
            "controls": self.controls,
            "summary_path": str(self.summary_path),
            "summary_sha256": file_sha256(self.summary_path),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "run_name": self.run_name,
            "run_id": self.run_id,
            "manifest_verification_warnings": list(self.manifest_warnings),
            "telemetry": {
                key: self.stationarity.get(key)
                for key in (
                    "rhat_energy",
                    "neff_energy",
                    "population_weight_status",
                    "log_weight_span_max",
                    "local_acceptance_fraction_mean",
                    "invalid_proposal_fraction_max",
                    "metropolis_rejection_fraction_max",
                    "configuration_esjd_mean",
                )
            },
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
    _validate_controls(reporting_resolution, confidence_level, fit_alpha)
    if write_artifacts and output_dir is None:
        raise ValueError("output_dir is required when artifacts are written")
    paths = _validated_population_paths(
        summary_paths,
        energy_assessment_manifest,
        apply_simultaneous_energy_stationarity,
    )
    points = sorted(
        (_load_population_point(path) for path in paths), key=lambda p: (p.dt, p.point.walkers)
    )
    _require_common_identity(points)
    groups: dict[float, list[_LoadedPopulationPoint]] = defaultdict(list)
    for point in points:
        groups[point.dt].append(point)
    fine_dt, coarse_dt, selected = _selected_timestep_groups(
        groups, selected_dt=selected_dt, interaction_dt=interaction_dt
    )
    ladders = {
        dt: analyze_population_ladder(
            [item.point for item in group],
            reporting_resolution=reporting_resolution,
            confidence_level=confidence_level,
            fit_alpha=fit_alpha,
        )
        for dt, group in groups.items()
    }
    interaction = _interaction(groups, fine_dt, coarse_dt, reporting_resolution, confidence_level)
    chosen = ladders[selected]
    quality_reasons = _quality_reasons(points)
    unresolved = list(quality_reasons)
    if chosen.classification not in PUBLICATION_READY_STATUSES:
        unresolved.append(chosen.classification)
    if interaction is None:
        unresolved.append("timestep_population_interaction_not_assessed")
    elif not interaction.bounded_below_reporting_resolution:
        unresolved.append("timestep_population_interaction_unresolved")
    status = chosen.classification if not unresolved else unresolved[0]
    publication_ready = not unresolved
    identity = _identity(points[0])
    payload: dict[str, Any] = {
        "schema_version": "dmc_population_systematics_v2",
        "status": status,
        "classification": chosen.classification,
        "case_id": points[0].case_id,
        "identity": identity,
        "identity_fingerprint": config_fingerprint(identity),
        "reference_fine_dt": fine_dt,
        "coarse_dt": coarse_dt,
        "selected_dt": selected,
        "selected_dt_basis": "explicit_selected_dt"
        if len(groups) == 2
        else "only_supplied_timestep",
        "selected_treatment_role": "reference_fine" if selected == fine_dt else "coarse",
        "selected_walkers": chosen.reference_walkers,
        "selected_walkers_basis": "reference_population_of_selected_timestep_ladder",
        "energy_reporting_policy": {
            "resolution": reporting_resolution,
            "confidence_level": confidence_level,
            "energy_unit": "hbar*Omega",
        },
        "fit_alpha": fit_alpha,
        "input_summaries": [point.reference() for point in points],
        "input_quality": {
            "status": "accepted" if not quality_reasons else "unresolved",
            "reasons": quality_reasons,
            "publication_accepted": not quality_reasons,
        },
        "reference_fine_population_ladder": ladders[fine_dt].to_dict(),
        "coarse_population_ladder": None if coarse_dt is None else ladders[coarse_dt].to_dict(),
        "selected_population_ladder": chosen.to_dict(),
        "timestep_population_interaction": None if interaction is None else interaction.to_dict(),
        "timestep_population_interaction_status": (
            "not_assessed"
            if interaction is None
            else "bounded_below_reporting_resolution"
            if interaction.bounded_below_reporting_resolution
            else "unresolved"
        ),
        "unresolved_reasons": unresolved,
        "publication_ready_within_population_systematic_scope": publication_ready,
        "qualified_systematics": {
            "finite_population": "accepted" if publication_ready else "unresolved",
            "timestep_population_interaction": "accepted" if publication_ready else "unresolved",
        },
    }
    _add_selected_energy(payload, chosen, groups[selected])
    if interaction is not None:
        payload["timestep_population_interaction_upper_allowance"] = interaction.upper_allowance
    payload["population_bounds"] = {
        "reference_fine_last_doubling_upper_allowance": ladders[
            fine_dt
        ].last_doubling.upper_allowance,
        "coarse_last_doubling_upper_allowance": None
        if coarse_dt is None
        else ladders[coarse_dt].last_doubling.upper_allowance,
        "selected_last_doubling_upper_allowance": chosen.last_doubling.upper_allowance,
        "timestep_population_interaction_upper_allowance": None
        if interaction is None
        else interaction.upper_allowance,
    }
    artifacts = (
        _write_population_artifacts(output_dir, payload, points, command)
        if write_artifacts
        else _empty_artifacts(output_dir)
    )
    payload["workflow_artifacts"] = artifacts
    return payload


def _validated_population_paths(
    summary_paths: Sequence[Path],
    energy_assessment_manifest: Path | None,
    apply_simultaneous_energy_stationarity: bool,
) -> tuple[Path, ...]:
    paths = tuple(Path(path).resolve() for path in summary_paths)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("population systematics requires at least two unique summaries")
    if energy_assessment_manifest is not None and apply_simultaneous_energy_stationarity:
        raise ValueError("choose one energy reassessment mode")
    return paths


def _selected_timestep_groups(
    groups: Mapping[float, list[_LoadedPopulationPoint]],
    *,
    selected_dt: float | None,
    interaction_dt: float | None,
) -> tuple[float, float | None, float]:
    if len(groups) > 2:
        raise ValueError("at most two timestep treatments are supported")
    fine_dt = min(groups)
    coarse_dt = max(groups) if len(groups) == 2 else None
    if coarse_dt is not None and selected_dt is None:
        raise ValueError("selected_dt is required for two timestep treatments")
    selected = fine_dt if selected_dt is None else _matching_dt(groups, selected_dt)
    if interaction_dt is not None and (coarse_dt is None or not _same(interaction_dt, coarse_dt)):
        raise ValueError("interaction_dt must identify the supplied coarse timestep")
    if coarse_dt is not None:
        fine, coarse = groups[fine_dt][0], groups[coarse_dt][0]
        for field in ("burn_tau", "production_tau", "grid_extent", "n_bins"):
            if fine.controls.get(field) != coarse.controls.get(field):
                raise ValueError(f"timestep-population interaction disagrees on {field}")
    return fine_dt, coarse_dt, selected


def _load_population_point(path: Path) -> _LoadedPopulationPoint:
    manifest_path = path.parent / "run_manifest.json"
    manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    summary = _mapping_json(path)
    run_name = _string(manifest, "run_name")
    if run_name not in _SOURCE_RUNS:
        raise ValueError(f"unsupported population source: {run_name}")
    if manifest.get("status") != summary.get("status"):
        raise ValueError(f"summary/manifest status mismatch: {path}")
    config = _mapping(manifest.get("config"), "manifest config")
    controls = _mapping(summary.get("controls"), "summary controls")
    if controls != _mapping(config.get("controls"), "manifest controls"):
        raise ValueError(f"summary/manifest controls mismatch: {path}")
    case_id, stationarity = _stationarity(summary, config, run_name)
    seeds = _seeds(summary, config, stationarity, run_name)
    rows = stationarity.get("seed_summaries")
    if not isinstance(rows, list):
        raise ValueError(f"stationarity seed summaries are missing: {path}")
    by_seed = {row.get("seed"): row for row in rows if isinstance(row, dict)}
    energies = np.asarray(
        [_finite(by_seed[seed].get("mixed_energy"), "seed energy") for seed in seeds]
    )
    energy = _finite(stationarity.get("mixed_energy"), "mixed energy")
    if not math.isclose(energy, float(np.mean(energies)), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"mixed energy is not the seed mean: {path}")
    stderr = max(
        _positive(stationarity.get("mixed_energy_conservative_stderr"), "energy stderr"),
        *(
            _finite(stationarity.get(key), key)
            for key in (
                "mixed_energy_seed_stderr",
                "mixed_energy_blocking_stderr",
                "mixed_energy_correlated_stderr",
            )
            if stationarity.get(key) is not None
        ),
    )
    return _LoadedPopulationPoint(
        point=PopulationEnergyPoint(
            walkers=_positive_int(controls.get("walkers"), "walkers"),
            energy=energy,
            conservative_stderr=stderr,
            seed_ids=seeds,
            seed_energies=energies,
            label=str(path),
        ),
        dt=_positive(controls.get("dt"), "dt"),
        case_id=case_id,
        controls=controls,
        stationarity=stationarity,
        summary_path=path,
        manifest_path=manifest_path,
        run_name=run_name,
        run_id=_string(manifest, "run_id"),
        manifest_warnings=tuple(warnings),
    )


def _stationarity(
    summary: dict[str, Any], config: dict[str, Any], run_name: str
) -> tuple[str, dict[str, Any]]:
    if run_name == "dmc_benchmark_packet":
        case_id = _string(summary, "case_id")
        if config.get("case") != case_id:
            raise ValueError("benchmark case identity mismatch")
        return case_id, _mapping(summary.get("stationarity"), "stationarity")
    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("stationarity source must contain one case")
    case_id = _string(cases[0], "case_id")
    if config.get("cases") != [case_id]:
        raise ValueError("stationarity case identity mismatch")
    return case_id, cases[0]


def _seeds(
    summary: dict[str, Any], config: dict[str, Any], stationarity: dict[str, Any], run_name: str
) -> tuple[int, ...]:
    candidates = [stationarity.get("seeds"), config.get("seeds")]
    if run_name == "dmc_benchmark_packet":
        candidates.append(summary.get("seeds"))
    parsed = [_seed_tuple(value) for value in candidates]
    if any(value != parsed[0] for value in parsed[1:]):
        raise ValueError("seed identities disagree")
    return parsed[0]


def _require_common_identity(points: Sequence[_LoadedPopulationPoint]) -> None:
    case_id = points[0].case_id
    keys = ("drift_limiter", "relative_alpha")
    reference = {key: points[0].controls.get(key) for key in keys}
    groups: dict[float, list[_LoadedPopulationPoint]] = defaultdict(list)
    for point in points[1:]:
        if point.case_id != case_id:
            raise ValueError("population points must share one case")
        if {key: point.controls.get(key) for key in keys} != reference:
            raise ValueError("population points must share sampler and guide controls")
    for point in points:
        groups[point.dt].append(point)
    for dt, group in groups.items():
        if len(group) not in {2, 3}:
            raise ValueError(f"dt={dt:.17g} requires W/2W or W/2,W,2W")
        controls = {key: value for key, value in group[0].controls.items() if key != "walkers"}
        if any(
            {key: value for key, value in point.controls.items() if key != "walkers"} != controls
            for point in group[1:]
        ):
            raise ValueError(f"dt={dt:.17g} points may vary only walker count")


def _identity(point: _LoadedPopulationPoint) -> dict[str, Any]:
    return {
        "case_id": point.case_id,
        "guide_family": point.stationarity.get("guide_family"),
        "guide_parameters": {
            "relative_alpha": point.controls.get("relative_alpha"),
        },
        "drift_limiter": point.controls.get("drift_limiter"),
        "energy_unit": point.stationarity.get("energy_unit", "hbar*Omega"),
    }


def _quality_reasons(points: Sequence[_LoadedPopulationPoint]) -> list[str]:
    reasons: list[str] = []
    accepted_chain = {"accepted", "spread_warning"}
    accepted_uncertainty = {
        "accepted",
        "conservative_error_inflated",
        "blocking_plateau_unresolved_correlated_error_available",
    }
    for point in points:
        stationarity = point.stationarity
        if stationarity.get("stationarity_energy") not in accepted_chain:
            reasons.append(f"{point.dt:g}/{point.point.walkers}:energy_chain")
        if stationarity.get("mixed_energy_uncertainty_status") not in accepted_uncertainty:
            reasons.append(f"{point.dt:g}/{point.point.walkers}:energy_uncertainty")
        if (
            stationarity.get("valid_finite_clean") is False
            or stationarity.get("density_accounting_clean") is False
        ):
            reasons.append(f"{point.dt:g}/{point.point.walkers}:base_numerics")
        if stationarity.get("population_weight_status") not in {None, "accepted"}:
            reasons.append(f"{point.dt:g}/{point.point.walkers}:population_weights")
    return reasons


def _interaction(
    groups: Mapping[float, list[_LoadedPopulationPoint]],
    fine_dt: float,
    coarse_dt: float | None,
    resolution: float,
    confidence: float,
) -> TimeStepPopulationInteraction | None:
    if coarse_dt is None:
        return None
    return analyze_timestep_population_interaction(
        [item.point for item in _doubling_pair(groups[fine_dt])],
        [item.point for item in _doubling_pair(groups[coarse_dt])],
        reporting_resolution=resolution,
        confidence_level=confidence,
    )


def _doubling_pair(
    points: Sequence[_LoadedPopulationPoint],
) -> tuple[_LoadedPopulationPoint, _LoadedPopulationPoint]:
    ordered = sorted(points, key=lambda item: item.point.walkers)
    for first, second in itertools.pairwise(ordered):
        if second.point.walkers == 2 * first.point.walkers:
            return first, second
    raise ValueError("each timestep treatment requires a W/2W pair")


def _add_selected_energy(
    payload: dict[str, Any],
    assessment: PopulationLadderAssessment,
    points: Sequence[_LoadedPopulationPoint],
) -> None:
    payload["selected_population_last_doubling_upper_allowance"] = (
        assessment.last_doubling.upper_allowance
    )
    if assessment.classification == "accepted_population_limit":
        assert (
            assessment.inverse_population_fit
            and assessment.richardson_window
            and assessment.population_limit_correction
        )
        payload.update(
            selected_energy_population_basis="population_limit_at_selected_timestep",
            population_limit_energy_at_selected_timestep=assessment.inverse_population_fit.intercept,
            population_limit_energy_statistical_stderr=assessment.inverse_population_fit.intercept_stderr,
            population_limit_model_window_upper_allowance=assessment.richardson_window.upper_allowance,
            population_limit_correction_at_selected_timestep=assessment.population_limit_correction.value,
        )
        return
    reference = next(
        item.point for item in points if item.point.walkers == assessment.reference_walkers
    )
    payload.update(
        selected_energy_population_basis="finite_population_at_selected_walkers",
        finite_population_energy_at_selected_timestep=reference.energy,
        finite_population_energy_statistical_stderr=reference.conservative_stderr,
        finite_population_walkers=reference.walkers,
        finite_population_w_to_2w_upper_allowance=assessment.last_doubling.upper_allowance,
    )


def _write_population_artifacts(
    output_dir: Path | None,
    payload: dict[str, Any],
    points: Sequence[_LoadedPopulationPoint],
    command: list[str] | None,
) -> dict[str, str]:
    assert output_dir is not None
    root = ensure_dir(output_dir.resolve())
    summary = root / "summary.json"
    point_table = write_csv(root / "population_points.csv", [point.reference() for point in points])
    comparison_table = write_csv(
        root / "population_comparisons.csv",
        [
            {
                "dt": payload["selected_dt"],
                "classification": payload["classification"],
                "reference_walkers": payload["selected_walkers"],
                "last_doubling_upper_allowance": payload["selected_population_ladder"][
                    "last_doubling"
                ]["upper_allowance"],
                "interaction_upper_allowance": payload.get(
                    "timestep_population_interaction_upper_allowance"
                ),
            }
        ],
    )
    write_json(summary, payload)
    manifest = write_run_manifest(
        root,
        run_name=POPULATION_SYSTEMATICS_RUN_NAME,
        config={
            "case_id": payload["case_id"],
            "selected_dt": payload["selected_dt"],
            "inputs": [point.reference() for point in points],
            "command": command,
        },
        artifacts=[summary, point_table, comparison_table],
        status=str(payload["status"]),
    )
    return {
        "summary": str(summary),
        "point_table": str(point_table),
        "comparison_table": str(comparison_table),
        "run_manifest": str(manifest),
        "output_dir": str(root),
    }


def _empty_artifacts(output_dir: Path | None) -> dict[str, str | None]:
    return {
        "summary": None,
        "point_table": None,
        "comparison_table": None,
        "run_manifest": None,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
    }


def _validate_controls(resolution: float, confidence: float, alpha: float) -> None:
    if not _same(resolution, FIXED_ENERGY_REPORTING_RESOLUTION):
        raise ValueError(f"reporting_resolution is fixed at {FIXED_ENERGY_REPORTING_RESOLUTION:g}")
    if not _same(confidence, 0.95) or not _same(alpha, 0.05):
        raise ValueError("confidence_level and fit_alpha are fixed at 0.95 and 0.05")


def _matching_dt(groups: Mapping[float, Any], selected: float) -> float:
    for value in groups:
        if _same(value, selected):
            return value
    raise ValueError("selected_dt is not one of the supplied treatments")


def _same(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1e-15)


def _mapping_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed_tuple(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value)
    ):
        raise ValueError("seed list must contain at least two integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError("seed identities must be unique")
    return result
