from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import (
    csv_text,
    ensure_dir,
    file_sha256,
    load_manifest_bound_artifact,
    write_json,
    write_run_manifest,
)
from hrdmc.system.settings import THESIS_CASE_ORDER, parse_case
from hrdmc.theory.tonks_girardeau import (
    trapped_tg_energy_total,
    trapped_tg_r2_radius,
    trapped_tg_rms_radius,
)
from hrdmc.uncertainty.forward_walking.outputs import FW_SENSITIVITY_RUN_NAME
from hrdmc.uncertainty.forward_walking.run import ACCEPTED_FW_STATUSES
from hrdmc.uncertainty.population import (
    POPULATION_SYSTEMATICS_RUN_NAME,
    PUBLICATION_READY_STATUSES,
)
from hrdmc.uncertainty.timestep.run import TIMESTEP_EXTRAPOLATION_RUN_NAME

NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME = "dmc_numerical_systematics_package"
SYSTEMATIC_LANES = ("timestep", "population", "forward_walking")
REQUIRED_CASE_ORDER = THESIS_CASE_ORDER
FINITE_CASE_ORDER = tuple(case for case in THESIS_CASE_ORDER if not case.endswith("_A0"))
_OWNERS = {
    "timestep": TIMESTEP_EXTRAPOLATION_RUN_NAME,
    "population": POPULATION_SYSTEMATICS_RUN_NAME,
    "forward_walking": FW_SENSITIVITY_RUN_NAME,
}
_FILES = {
    "summary": "summary.json",
    "case_status_table": "case_status.csv",
    "thesis_energy_table": "thesis_energy_table.csv",
    "uncertainty_table": "uncertainty_components.csv",
    "proposal_table": "proposal_efficiency.csv",
    "source_table": "source_artifacts.csv",
}


@dataclass(frozen=True)
class _BoundSystematicAssessment:
    lane: str
    case_id: str
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]

    def reference(self, root: Path) -> dict[str, Any]:
        return {
            "manifest_path": _relative(self.manifest_path, root),
            "manifest_sha256": file_sha256(self.manifest_path),
            "summary_path": _relative(self.summary_path, root),
            "summary_sha256": file_sha256(self.summary_path),
            "run_id": self.manifest.get("run_id"),
            "run_name": self.manifest.get("run_name"),
            "status": self.summary.get("status"),
        }


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
    """Join one verified summary per lane; lane owners retain scientific checks."""
    _validate_case_maps(timestep_manifests, population_manifests, fw_sensitivity_manifests)
    qualifiers = _validate_qualifiers(bounded_qualifiers or {})
    root = output_dir.resolve()
    final_path = final_matrix_manifest.resolve()
    final_summary_path = final_path.parent / "final_matrix_summary.json"
    final_manifest, _ = load_manifest_bound_artifact(final_path, final_summary_path)
    final_summary = _json(final_summary_path)
    if final_manifest.get("run_name") != "dmc_final_matrix_assembly":
        raise ValueError("final matrix input has the wrong owner")
    if final_summary.get("case_order") != list(REQUIRED_CASE_ORDER):
        raise ValueError("final matrix case order is not canonical")
    final_rows = {
        _string(row, "case"): row for row in final_summary.get("rows", []) if isinstance(row, dict)
    }
    lanes = {
        "timestep": _load_lane(timestep_manifests, "timestep"),
        "population": _load_lane(population_manifests, "population"),
        "forward_walking": _load_lane(fw_sensitivity_manifests, "forward_walking"),
    }
    rows = [
        _exact_tg_row(case_id, final_rows[case_id])
        if case_id.endswith("_A0")
        else _finite_row(
            case_id,
            final_rows[case_id],
            *(lanes[lane].get(case_id) for lane in SYSTEMATIC_LANES),
            qualifiers=qualifiers,
        )
        for case_id in REQUIRED_CASE_ORDER
    ]
    missing = {
        lane: [case for case in FINITE_CASE_ORDER if case not in values]
        for lane, values in lanes.items()
    }
    unresolved = [row["case"] for row in rows if not row["publication_ready"]]
    payload = {
        "schema_version": "dmc_numerical_systematics_package_v2",
        "status": "accepted"
        if not unresolved and not any(missing.values())
        else "systematics_incomplete",
        "case_order": list(REQUIRED_CASE_ORDER),
        "rows": rows,
        "publication_ready_case_count": len(rows) - len(unresolved),
        "unresolved_cases": unresolved,
        "missing_inputs": missing,
        "bounded_qualifiers": _qualifiers_payload(qualifiers),
        "sources": {
            "final_matrix": {
                "manifest_path": _relative(final_path, root),
                "manifest_sha256": file_sha256(final_path),
                "summary_path": _relative(final_summary_path, root),
                "summary_sha256": file_sha256(final_summary_path),
            },
            **{
                lane: {case: item.reference(root) for case, item in values.items()}
                for lane, values in lanes.items()
            },
        },
    }
    payload["thesis_energy_rows"] = _table_rows(payload)["thesis_energy_table"]
    artifacts = _write_package(root, payload, command)
    return payload, artifacts


def verify_numerical_systematics_package_manifest(path: Path) -> tuple[bool, list[str]]:
    """Verify manifest binding and deterministic tables without replaying lane owners."""
    try:
        root = path.resolve().parent
        summary_path = root / _FILES["summary"]
        manifest, _ = load_manifest_bound_artifact(path.resolve(), summary_path)
        summary = _json(summary_path)
        if manifest.get("run_name") != NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME:
            raise ValueError("numerical package has the wrong owner")
        if manifest.get("status") != summary.get("status"):
            raise ValueError("numerical package status mismatch")
        tables = _table_rows(summary)
        for name, rows in tables.items():
            if (root / _FILES[name]).read_text(encoding="utf-8") != csv_text(rows):
                raise ValueError(f"{_FILES[name]} disagrees with summary.json")
        source_map = summary.get("sources")
        if not isinstance(source_map, dict):
            raise ValueError("systematics sources are invalid")
        _verify_package_sources(root, source_map)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, [str(exc)]
    return True, []


def _verify_package_sources(root: Path, source_map: dict[str, Any]) -> None:
    lanes = {key: value for key, value in source_map.items() if key != "final_matrix"}
    if any(not isinstance(sources, dict) for sources in lanes.values()):
        raise ValueError("systematics source lane is invalid")
    lane_sources = [
        source
        for sources in lanes.values()
        if isinstance(sources, dict)
        for source in sources.values()
    ]
    if any(not isinstance(source, dict) for source in lane_sources):
        raise ValueError("systematics source reference is invalid")
    fw_sources = lanes.get("forward_walking", {})
    assert isinstance(fw_sources, dict)
    for source in fw_sources.values():
        assert isinstance(source, dict)
        manifest_path = _bound_path(root, source.get("manifest_path"))
        config = _json(manifest_path).get("config")
        candidate = config.get("candidate") if isinstance(config, dict) else None
        _verify_bound_reference(manifest_path.parent, candidate)


def _load_lane(paths: Mapping[str, Path], lane: str) -> dict[str, _BoundSystematicAssessment]:
    result: dict[str, _BoundSystematicAssessment] = {}
    for declared_case, value in paths.items():
        manifest_path = value.resolve()
        summary_path = manifest_path.parent / "summary.json"
        manifest, _ = load_manifest_bound_artifact(manifest_path, summary_path)
        summary = _json(summary_path)
        if manifest.get("run_name") != _OWNERS[lane]:
            raise ValueError(f"{declared_case}: {lane} input has the wrong owner")
        if manifest.get("status") != summary.get("status"):
            raise ValueError(f"{declared_case}: {lane} status mismatch")
        if summary.get("case_id") != declared_case:
            raise ValueError(f"{declared_case}: {lane} case mismatch")
        result[declared_case] = _BoundSystematicAssessment(
            lane, declared_case, manifest_path, summary_path, manifest, summary
        )
    return result


def _exact_tg_row(case_id: str, final: dict[str, Any]) -> dict[str, Any]:
    particles = parse_case(case_id).n_particles
    exact_energy = trapped_tg_energy_total(particles, 1.0)
    return {
        "case": case_id,
        "n_particles": particles,
        "rod_length_ho": 0.0,
        "status": "accepted_exact_tg",
        "publication_ready": True,
        "exact_tg": True,
        "raw_dt": final.get("dt"),
        "raw_walkers": final.get("walkers"),
        "raw_finite_dt_energy": final.get("energy"),
        "raw_finite_dt_energy_stderr": final.get("energy_stderr"),
        "thesis_energy": exact_energy,
        "energy_lda": exact_energy,
        "energy_relative_delta_vs_lda": 0.0,
        "r2": trapped_tg_r2_radius(particles, 1.0),
        "rms_radius": trapped_tg_rms_radius(particles, 1.0),
        "lane_status": {
            "final_matrix": final.get("status"),
            "timestep": "exact",
            "population": "exact",
            "forward_walking": "exact",
        },
        "uncertainty_components": {},
        "unresolved_reasons": [],
    }


def _finite_row(
    case_id: str,
    final: dict[str, Any],
    timestep: _BoundSystematicAssessment | None,
    population: _BoundSystematicAssessment | None,
    fw: _BoundSystematicAssessment | None,
    *,
    qualifiers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    summaries = {
        "timestep": None if timestep is None else timestep.summary,
        "population": None if population is None else population.summary,
        "forward_walking": None if fw is None else fw.summary,
    }
    case = parse_case(case_id)
    accepted = {
        "timestep": _accepted("timestep", summaries["timestep"]),
        "population": _accepted("population", summaries["population"]),
        "forward_walking": _accepted("forward_walking", summaries["forward_walking"]),
    }
    for lane in SYSTEMATIC_LANES:
        if (case_id, lane) in qualifiers and _bounded(lane, summaries[lane]):
            accepted[lane] = True
    ready = final.get("status") == "accepted" and all(accepted.values())
    ts = summaries["timestep"] or {}
    pop = summaries["population"] or {}
    fw_summary = summaries["forward_walking"] or {}
    zero_energy = _optional_float(ts.get("extrapolated_energy"))
    population_correction = _optional_float(
        pop.get("population_limit_correction_at_selected_timestep")
    )
    thesis_energy = (
        None if not ready or zero_energy is None else zero_energy + (population_correction or 0.0)
    )
    energy_lda = _optional_float(final.get("energy_lda"))
    comparison = fw_summary.get("observable_comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    timestep_extrapolation = ts.get("extrapolation")
    timestep_extrapolation = (
        timestep_extrapolation if isinstance(timestep_extrapolation, dict) else {}
    )
    unresolved = [lane for lane, value in accepted.items() if not value]
    return {
        "case": case_id,
        "n_particles": case.n_particles,
        "rod_length_ho": case.rod_length_ho,
        "status": "accepted" if ready else "systematics_incomplete",
        "publication_ready": ready,
        "exact_tg": None,
        "raw_dt": final.get("dt"),
        "raw_walkers": final.get("walkers"),
        "raw_finite_dt_energy": final.get("energy"),
        "raw_finite_dt_energy_stderr": final.get("energy_stderr"),
        "candidate_zero_timestep_energy_at_selected_walkers": zero_energy,
        "population_limit_correction_at_selected_timestep": population_correction,
        "thesis_energy": thesis_energy,
        "energy_lda": energy_lda,
        "energy_relative_delta_vs_lda": _relative_delta(thesis_energy, energy_lda),
        "lane_status": {
            "final_matrix": final.get("status"),
            **{lane: "accepted" if accepted[lane] else "unresolved" for lane in SYSTEMATIC_LANES},
        },
        "source_lane_status": {
            lane: None if (summary := summaries[lane]) is None else summary.get("status")
            for lane in SYSTEMATIC_LANES
        },
        "fw_sensitivity_treatment": _fw_treatment(fw_summary),
        "proposal_telemetry": fw_summary.get("proposal_telemetry"),
        "uncertainty_components": {
            "energy_statistical_stderr": ts.get("extrapolated_energy_statistical_stderr"),
            "timestep_fit_window_upper_allowance": _nested(
                timestep_extrapolation,
                "largest_point_leave_one_out",
                "leading_linear",
                "absolute_shift",
            ),
            "timestep_model_order_upper_allowance": timestep_extrapolation.get(
                "leading_model_intercept_spread"
            ),
            "population_selected_last_doubling_upper_allowance": pop.get(
                "selected_population_last_doubling_upper_allowance"
            ),
            "timestep_population_interaction_upper_allowance": pop.get(
                "timestep_population_interaction_upper_allowance"
            ),
            "fw_anchor_rms_lag_relative_upper_bound": final.get(
                "rms_fw_lag_systematic_relative_upper_bound"
            ),
            "fw_anchor_density_lag_relative_l2_upper_bound": final.get(
                "density_fw_lag_systematic_relative_l2_upper_bound"
            ),
            "fw_treatment_rms_relative_upper_bound": _nested(
                comparison, "rms_radius", "simultaneous_relative_upper_bound"
            ),
            "fw_treatment_density_relative_l2_upper_bound": _nested(
                comparison, "density", "simultaneous_upper_bound"
            ),
        },
        "bounded_qualifiers": {
            lane: qualifiers[(case_id, lane)]
            for lane in SYSTEMATIC_LANES
            if (case_id, lane) in qualifiers
        },
        "unresolved_reasons": unresolved,
    }


def _accepted(lane: str, summary: dict[str, Any] | None) -> bool:
    if summary is None:
        return False
    status = str(summary.get("status"))
    if lane == "timestep":
        return status in {"accepted", "accepted_with_warnings"}
    if lane == "population":
        return status in PUBLICATION_READY_STATUSES or bool(
            summary.get("publication_ready_within_population_systematic_scope")
        )
    return status in ACCEPTED_FW_STATUSES


def _bounded(lane: str, summary: dict[str, Any] | None) -> bool:
    if summary is None:
        return False
    qualified = summary.get("qualified_systematics")
    if isinstance(qualified, dict):
        return any(value == "accepted" for value in qualified.values())
    if isinstance(qualified, list):
        return bool(qualified) and all(isinstance(value, str) for value in qualified)
    return False


def _fw_treatment(summary: dict[str, Any]) -> dict[str, Any] | None:
    treatments = summary.get("treatments")
    if not isinstance(treatments, dict) or not isinstance(treatments.get("candidate"), dict):
        return None
    candidate = treatments["candidate"]
    return {
        "dt": candidate.get("dt"),
        "walkers": candidate.get("walkers"),
        "role": "coordinate_observable_sensitivity_treatment",
    }


def _write_package(
    root: Path, payload: dict[str, Any], command: list[str] | None
) -> dict[str, Path]:
    ensure_dir(root)
    paths = {name: root / filename for name, filename in _FILES.items()}
    write_json(paths["summary"], payload)
    for name, rows in _table_rows(payload).items():
        paths[name].write_text(csv_text(rows), encoding="utf-8")
    paths["run_manifest"] = write_run_manifest(
        root,
        run_name=NUMERICAL_SYSTEMATICS_PACKAGE_RUN_NAME,
        config={
            "case_order": list(REQUIRED_CASE_ORDER),
            "sources": payload["sources"],
            "command": command,
        },
        artifacts=[paths[name] for name in _FILES],
        status=str(payload["status"]),
    )
    paths["output_dir"] = root
    return paths


def _table_rows(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    return {
        "case_status_table": [
            {
                "case": row["case"],
                "status": row["status"],
                "publication_ready": row["publication_ready"],
                "unresolved_reasons": ",".join(row["unresolved_reasons"]),
            }
            for row in rows
        ],
        "thesis_energy_table": [
            {
                "case": row["case"],
                "energy": row.get("thesis_energy"),
                "energy_lda": row.get("energy_lda"),
                "relative_delta_vs_lda": row.get("energy_relative_delta_vs_lda"),
            }
            for row in rows
            if row.get("publication_ready") is True
        ],
        "uncertainty_table": [
            {"case": row["case"], **row.get("uncertainty_components", {})} for row in rows
        ],
        "proposal_table": [
            {"case": row["case"], **(row.get("proposal_telemetry") or {})}
            for row in rows
            if row.get("proposal_telemetry")
        ],
        "source_table": _source_rows(payload.get("sources")),
    }


def _source_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for lane, sources in sorted(value.items()):
        if lane == "final_matrix":
            rows.append({"lane": lane, "case": "all", **sources})
        elif isinstance(sources, dict):
            rows.extend(
                {"lane": lane, "case": case, **source} for case, source in sorted(sources.items())
            )
    return rows


def _verify_bound_reference(root: Path, value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("forward-walking candidate reference is invalid")
    for name in ("summary", "manifest"):
        path = _bound_path(root, value.get(f"{name}_path"))
        if not path.is_file() or file_sha256(path) != value.get(f"{name}_sha256"):
            raise ValueError(f"bound source identity mismatch: {path}")


def _bound_path(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return (path if path.is_absolute() else root / path).resolve()


def _validate_case_maps(*maps: Mapping[str, Path]) -> None:
    allowed = set(FINITE_CASE_ORDER)
    for values in maps:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported systematics cases: {sorted(unknown)}")


def _validate_qualifiers(value: Mapping[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for (case, lane), reason in value.items():
        if case not in FINITE_CASE_ORDER or lane not in SYSTEMATIC_LANES or not reason.strip():
            raise ValueError(f"invalid bounded qualifier: {case}:{lane}")
        result[(case, lane)] = reason.strip()
    return result


def _qualifiers_payload(value: Mapping[tuple[str, str], str]) -> dict[str, str]:
    return {f"{case}:{lane}": reason for (case, lane), reason in value.items()}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _relative_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return (value - reference) / reference


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    result = float(value)  # type: ignore[arg-type]
    return result if math.isfinite(result) else None
