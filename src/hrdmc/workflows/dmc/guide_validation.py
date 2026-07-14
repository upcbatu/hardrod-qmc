from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts import (
    build_run_provenance,
    config_fingerprint,
    file_sha256,
    implementation_identity,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.workflows.dmc.guide_mala_diagnostic import (
    GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION,
    guide_mala_manifest_config,
)
from hrdmc.workflows.dmc.trapped import TrappedCase, parse_case

CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION = "contact_guide_mala_validation_v2"
CONTACT_GUIDE_VALIDATION_RUN_NAME = "contact_guide_mala_validation"


@dataclass(frozen=True)
class ContactGuideValidationThresholds:
    """Conservative calibration checks, not universal physical constants.

    The checks reject visible dependence on compact versus expanded starts and
    reject a MALA step that is nearly frozen or frequently invalid. They do not
    establish DMC branching, population, timestep, or projector systematics.
    """

    min_start_scale_separation: float = 0.20
    min_tail_trace_rows: int = 20
    min_acceptance_fraction: float = 0.80
    max_acceptance_fraction_difference: float = 0.05
    max_invalid_proposal_fraction: float = 0.05
    max_r2_relative_difference: float = 0.02
    max_weighted_free_gap_relative_difference: float = 0.05
    max_free_gap_p01_relative_difference: float = 0.20
    max_energy_mean_difference_mad_units: float = 1.0
    max_local_energy_mad_ratio: float = 1.5
    min_normalized_configuration_esjd: float = 0.05
    min_configuration_esjd_ratio: float = 0.50
    min_r2_esjd_ratio: float = 0.50
    min_weighted_free_gap_esjd_ratio: float = 0.50


@dataclass(frozen=True)
class ValidatedContactGuideArtifact:
    relative_alpha: float
    contact_beta: float
    summary_path: Path
    summary_sha256: str
    manifest_sha256: str
    identity_fingerprint: str
    source_tree_sha256: str


@dataclass(frozen=True)
class _CalibrationInput:
    path: Path
    manifest_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]


def validate_production_contact_guide_binding(
    *,
    case: TrappedCase,
    guide_family: str,
    relative_alpha: float | None,
    contact_beta: float | None,
    source: str,
    source_sha256: str | None,
    source_manifest_sha256: str | None,
    source_identity_fingerprint: str | None,
) -> None:
    """Require a live, manifest-verified validation artifact at workflow entry."""

    identity_values = (
        source_sha256,
        source_manifest_sha256,
        source_identity_fingerprint,
    )
    if guide_family != "contact-corrected-reduced-tg":
        if contact_beta is not None:
            raise ValueError("contact_beta requires the contact-corrected guide family")
        if any(value is not None for value in identity_values):
            raise ValueError("validated contact-guide identity cannot bind another guide family")
        return
    if relative_alpha is None or contact_beta is None or source == "explicit":
        raise ValueError("contact-corrected production requires a validated guide artifact")
    artifact = load_validated_contact_guide(Path(source), case=case)
    observed = (
        float(relative_alpha),
        float(contact_beta),
        source_sha256,
        source_manifest_sha256,
        source_identity_fingerprint,
    )
    expected = (
        artifact.relative_alpha,
        artifact.contact_beta,
        artifact.summary_sha256,
        artifact.manifest_sha256,
        artifact.identity_fingerprint,
    )
    if observed != expected:
        raise ValueError("contact-guide parameters or artifact identity do not match validation")


def validate_contact_guide_calibrations(
    *,
    compact_summary: Path,
    expanded_summary: Path,
) -> dict[str, Any]:
    """Compare two manifest-bound guide-squared MALA calibration runs."""

    thresholds = ContactGuideValidationThresholds()
    loaded: dict[str, _CalibrationInput] = {}
    input_errors: dict[str, str] = {}
    for role, path in (("compact", compact_summary), ("expanded", expanded_summary)):
        try:
            loaded[role] = _load_calibration_input(path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            input_errors[role] = str(exc)

    base: dict[str, Any] = {
        "schema_version": CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION,
        "thresholds": asdict(thresholds),
        "threshold_scope": (
            "Conservative numerical calibration criteria for two independently "
            "started guide-squared MALA runs; not universal physical constants."
        ),
        "scientific_scope": {
            "established_if_validated": (
                "the fixed contact-guide parameters pass branching-free MALA "
                "start-sensitivity, tail-geometry, tail-energy, and mobility checks"
            ),
            "not_established": [
                "DMC branching-weight stability",
                "population-control independence",
                "DMC timestep independence",
                "production-seed agreement",
            ],
        },
    }
    if input_errors:
        return {
            **base,
            "status": "input_artifact_invalid",
            "inputs": {
                role: _untrusted_input_identity(path)
                for role, path in (
                    ("compact", compact_summary),
                    ("expanded", expanded_summary),
                )
            },
            "input_errors": input_errors,
            "checks": [],
            "failed_checks": [f"{role}_input_artifact" for role in sorted(input_errors)],
        }

    compact = loaded["compact"]
    expanded = loaded["expanded"]
    compact_summary_payload = compact.summary
    expanded_summary_payload = expanded.summary
    compact_controls = compact_summary_payload["controls"]
    expanded_controls = expanded_summary_payload["controls"]
    compact_tail = compact_summary_payload["tail_means"]
    expanded_tail = expanded_summary_payload["tail_means"]

    checks: list[dict[str, Any]] = []
    _append_check(
        checks,
        name="case_identity",
        group="contract",
        observed=[compact_summary_payload["case_id"], expanded_summary_payload["case_id"]],
        criterion="identical",
        passed=compact_summary_payload["case_id"] == expanded_summary_payload["case_id"],
    )
    _append_check(
        checks,
        name="independent_seeds",
        group="contract",
        observed=[compact_summary_payload["seed"], expanded_summary_payload["seed"]],
        criterion="distinct",
        passed=compact_summary_payload["seed"] != expanded_summary_payload["seed"],
    )
    compact_scale = float(compact_controls["start_scale"])
    expanded_scale = float(expanded_controls["start_scale"])
    _append_check(
        checks,
        name="compact_expanded_start_bracket",
        group="contract",
        observed={"compact": compact_scale, "expanded": expanded_scale},
        criterion={
            "compact_less_than": 1.0,
            "expanded_greater_than": 1.0,
            "minimum_separation": thresholds.min_start_scale_separation,
        },
        passed=(
            compact_scale < 1.0
            and expanded_scale > 1.0
            and expanded_scale - compact_scale >= thresholds.min_start_scale_separation
        ),
    )
    shared_control_names = (
        "dt",
        "walkers",
        "duration_tau",
        "steps",
        "store_every",
        "grid_extent",
        "n_bins",
        "relative_alpha",
        "guide_family",
        "contact_beta",
        "tail_tau",
    )
    control_differences = {
        name: [compact_controls.get(name), expanded_controls.get(name)]
        for name in shared_control_names
        if compact_controls.get(name) != expanded_controls.get(name)
    }
    _append_check(
        checks,
        name="shared_calibration_controls",
        group="contract",
        observed=control_differences,
        criterion="no differences except seed and start_scale",
        passed=not control_differences,
    )
    source_identities = [
        _guide_parameter_source_identity(compact_summary_payload),
        _guide_parameter_source_identity(expanded_summary_payload),
    ]
    _append_check(
        checks,
        name="guide_parameter_source_identity",
        group="contract",
        observed=source_identities,
        criterion="identical candidate or explicit source",
        passed=source_identities[0] == source_identities[1],
    )
    source_tree_hashes = [
        _source_tree_sha256(compact.manifest),
        _source_tree_sha256(expanded.manifest),
    ]
    _append_check(
        checks,
        name="implementation_source_identity",
        group="contract",
        observed=source_tree_hashes,
        criterion="identical source_tree_sha256",
        passed=source_tree_hashes[0] == source_tree_hashes[1],
    )
    current_source_tree = implementation_identity().get("source_tree_sha256")
    _append_check(
        checks,
        name="current_implementation_source_identity",
        group="contract",
        observed={
            "calibrations": source_tree_hashes,
            "validation": current_source_tree,
        },
        criterion="both calibrations equal the validation source_tree_sha256",
        passed=(
            isinstance(current_source_tree, str)
            and all(value == current_source_tree for value in source_tree_hashes)
        ),
    )
    dependencies = [
        compact.manifest["provenance"].get("dependencies"),
        expanded.manifest["provenance"].get("dependencies"),
    ]
    _append_check(
        checks,
        name="runtime_dependency_identity",
        group="contract",
        observed=dependencies,
        criterion="identical recorded dependency versions",
        passed=dependencies[0] == dependencies[1],
    )
    tail_rows = [
        int(compact_summary_payload["tail_trace_rows"]),
        int(expanded_summary_payload["tail_trace_rows"]),
    ]
    _append_check(
        checks,
        name="tail_trace_support",
        group="contract",
        observed=tail_rows,
        criterion={"minimum_each": thresholds.min_tail_trace_rows},
        passed=min(tail_rows) >= thresholds.min_tail_trace_rows,
    )

    acceptance = [
        float(compact_tail["acceptance_fraction"]),
        float(expanded_tail["acceptance_fraction"]),
    ]
    _append_check(
        checks,
        name="tail_acceptance_floor",
        group="proposal",
        observed=acceptance,
        criterion={"minimum_each": thresholds.min_acceptance_fraction},
        passed=min(acceptance) >= thresholds.min_acceptance_fraction,
    )
    _append_check(
        checks,
        name="tail_acceptance_start_difference",
        group="proposal",
        observed=abs(acceptance[0] - acceptance[1]),
        criterion={"maximum": thresholds.max_acceptance_fraction_difference},
        passed=(
            abs(acceptance[0] - acceptance[1]) <= thresholds.max_acceptance_fraction_difference
        ),
    )
    invalid = [
        float(compact_tail["invalid_proposal_fraction"]),
        float(expanded_tail["invalid_proposal_fraction"]),
    ]
    _append_check(
        checks,
        name="tail_invalid_proposal_ceiling",
        group="proposal",
        observed=invalid,
        criterion={"maximum_each": thresholds.max_invalid_proposal_fraction},
        passed=max(invalid) <= thresholds.max_invalid_proposal_fraction,
    )

    normalized_configuration_esjd = [
        float(compact_tail["configuration_esjd"])
        / (int(compact_summary_payload["n_particles"]) * float(compact_controls["dt"])),
        float(expanded_tail["configuration_esjd"])
        / (int(expanded_summary_payload["n_particles"]) * float(expanded_controls["dt"])),
    ]
    _append_check(
        checks,
        name="tail_normalized_configuration_esjd_floor",
        group="mobility",
        observed=normalized_configuration_esjd,
        criterion={
            "minimum_each": thresholds.min_normalized_configuration_esjd,
            "normalization": "configuration_esjd / (n_particles * dt)",
        },
        passed=(min(normalized_configuration_esjd) >= thresholds.min_normalized_configuration_esjd),
    )

    _append_relative_difference_check(
        checks,
        name="tail_r2_start_difference",
        group="geometry",
        values=(compact_tail["r2_mean"], expanded_tail["r2_mean"]),
        maximum=thresholds.max_r2_relative_difference,
    )
    _append_relative_difference_check(
        checks,
        name="tail_weighted_free_gap_start_difference",
        group="geometry",
        values=(
            compact_tail["weighted_free_gap_mean"],
            expanded_tail["weighted_free_gap_mean"],
        ),
        maximum=thresholds.max_weighted_free_gap_relative_difference,
    )
    _append_relative_difference_check(
        checks,
        name="tail_free_gap_p01_start_difference",
        group="geometry",
        values=(compact_tail["free_gap_p01"], expanded_tail["free_gap_p01"]),
        maximum=thresholds.max_free_gap_p01_relative_difference,
    )

    energy_difference_mad_units = _difference_in_max_scale(
        float(compact_tail["local_energy_mean"]),
        float(expanded_tail["local_energy_mean"]),
        float(compact_tail["local_energy_mad"]),
        float(expanded_tail["local_energy_mad"]),
    )
    _append_check(
        checks,
        name="tail_local_energy_mean_start_difference",
        group="energy",
        observed=energy_difference_mad_units,
        criterion={
            "maximum_in_larger_tail_mad_units": (thresholds.max_energy_mean_difference_mad_units)
        },
        passed=energy_difference_mad_units <= thresholds.max_energy_mean_difference_mad_units,
    )
    energy_mad_ratio = _positive_similarity_ratio(
        float(compact_tail["local_energy_mad"]),
        float(expanded_tail["local_energy_mad"]),
    )
    _append_check(
        checks,
        name="tail_local_energy_mad_ratio",
        group="energy",
        observed=energy_mad_ratio,
        criterion={"maximum_larger_over_smaller": thresholds.max_local_energy_mad_ratio},
        passed=energy_mad_ratio <= thresholds.max_local_energy_mad_ratio,
    )

    for metric, minimum in (
        ("configuration_esjd", thresholds.min_configuration_esjd_ratio),
        ("r2_esjd", thresholds.min_r2_esjd_ratio),
        ("weighted_free_gap_esjd", thresholds.min_weighted_free_gap_esjd_ratio),
    ):
        similarity = _positive_inverse_ratio(
            float(compact_tail[metric]),
            float(expanded_tail[metric]),
        )
        _append_check(
            checks,
            name=f"tail_{metric}_start_ratio",
            group="mobility",
            observed=similarity,
            criterion={"minimum_smaller_over_larger": minimum},
            passed=similarity >= minimum,
        )

    failed_checks = [str(check["name"]) for check in checks if not check["passed"]]
    failed_contract = any(not check["passed"] and check["group"] == "contract" for check in checks)
    if failed_contract:
        status = "calibration_contract_mismatch"
    elif failed_checks:
        status = "calibration_disagreement"
    else:
        status = "validated"

    parameters = {
        "guide_family": compact_controls["guide_family"],
        "relative_alpha": compact_controls["relative_alpha"],
        "contact_beta": compact_controls["contact_beta"],
    }
    payload: dict[str, Any] = {
        **base,
        "status": status,
        "case_id": compact_summary_payload["case_id"],
        "inputs": {
            "compact": _trusted_input_identity(compact),
            "expanded": _trusted_input_identity(expanded),
        },
        "candidate_parameters": parameters,
        "checks": checks,
        "failed_checks": failed_checks,
    }
    if status == "validated":
        payload["validated_parameters"] = parameters
    return payload


def write_contact_guide_validation_output(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    command: list[str] | None,
) -> dict[str, Path]:
    """Persist an immutable source-bound offline validation artifact."""

    if payload.get("schema_version") != CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION:
        raise ValueError("contact-guide validation payload has an unsupported schema")
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"contact-guide validation output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, payload)
    written_manifest = write_run_manifest(
        output_dir,
        run_name=CONTACT_GUIDE_VALIDATION_RUN_NAME,
        config={
            "inputs": payload.get("inputs", {}),
            "thresholds": payload["thresholds"],
        },
        artifacts=[summary_path],
        schema_version=CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=str(payload["status"]),
    )
    return {"summary": summary_path, "run_manifest": written_manifest}


def load_validated_contact_parameters(
    path: Path,
    *,
    case: TrappedCase,
) -> tuple[float, float]:
    """Load contact-guide parameters only from a validated calibration artifact."""

    artifact = load_validated_contact_guide(path, case=case)
    return artifact.relative_alpha, artifact.contact_beta


def load_validated_contact_guide(
    path: Path,
    *,
    case: TrappedCase,
) -> ValidatedContactGuideArtifact:
    """Load validated parameters together with their immutable artifact identity."""

    resolved_path = path.resolve()
    manifest_path = resolved_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("contact-guide validation summary has no run manifest")
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError(
            "contact-guide validation manifest failed verification: " + "; ".join(errors)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != CONTACT_GUIDE_VALIDATION_RUN_NAME:
        raise ValueError("contact-guide validation manifest has the wrong owner")
    if manifest.get("result_schema_version") != CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION:
        raise ValueError("contact-guide validation manifest has the wrong result schema")
    if manifest.get("status") != "validated":
        raise ValueError("contact-guide validation manifest is not validated")
    _require_manifest_artifact(resolved_path, manifest)
    implementation = manifest.get("provenance", {}).get("implementation", {})
    if implementation.get("status") != "identified":
        raise ValueError("contact-guide validation manifest has no identified source tree")
    source_tree_sha256 = implementation.get("source_tree_sha256")
    if not isinstance(source_tree_sha256, str) or len(source_tree_sha256) != 64:
        raise ValueError("contact-guide validation manifest has an invalid source identity")
    current_implementation = implementation_identity()
    if current_implementation.get("source_tree_sha256") != source_tree_sha256:
        raise ValueError(
            "contact-guide validation was produced by a different scientific source tree"
        )

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION:
        raise ValueError("contact-guide validation summary has an unsupported schema")
    if payload.get("status") != "validated":
        raise ValueError("contact-guide validation summary is not validated")
    if payload.get("case_id") != case.case_id:
        raise ValueError(
            f"contact-guide validation is for {payload.get('case_id')}, expected {case.case_id}"
        )
    if payload.get("failed_checks"):
        raise ValueError("contact-guide validation summary contains failed checks")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("contact-guide validation summary has no bound calibration inputs")
    for role in ("compact", "expanded"):
        input_identity = inputs.get(role)
        if not isinstance(input_identity, dict):
            raise ValueError(f"contact-guide validation has no {role} input identity")
        if input_identity.get("source_tree_sha256") != source_tree_sha256:
            raise ValueError(
                f"contact-guide validation {role} input has a different source identity"
            )
    checks = payload.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or not all(isinstance(check, dict) and check.get("passed") is True for check in checks)
    ):
        raise ValueError("contact-guide validation checks are incomplete")
    if payload.get("thresholds") != to_jsonable(asdict(ContactGuideValidationThresholds())):
        raise ValueError("contact-guide validation thresholds do not match this schema")
    parameters = payload.get("validated_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("contact-guide validation has no validated_parameters")
    if parameters.get("guide_family") != "contact-corrected-reduced-tg":
        raise ValueError("contact-guide validation has the wrong guide family")
    alpha = float(parameters["relative_alpha"])
    beta = float(parameters["contact_beta"])
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("validated relative_alpha must be finite and positive")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("validated contact_beta must be finite and lie in [0, 1]")
    summary_sha256 = file_sha256(resolved_path)
    manifest_sha256 = file_sha256(manifest_path)
    return ValidatedContactGuideArtifact(
        relative_alpha=alpha,
        contact_beta=beta,
        summary_path=resolved_path,
        summary_sha256=summary_sha256,
        manifest_sha256=manifest_sha256,
        identity_fingerprint=config_fingerprint(
            {
                "schema_version": CONTACT_GUIDE_VALIDATION_SCHEMA_VERSION,
                "summary_sha256": summary_sha256,
                "manifest_sha256": manifest_sha256,
            }
        ),
        source_tree_sha256=source_tree_sha256,
    )


def _load_calibration_input(path: Path) -> _CalibrationInput:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"summary does not exist: {resolved}")
    manifest_path = resolved.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("summary has no sibling run_manifest.json")
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError("run manifest failed verification: " + "; ".join(errors))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != "guide_mala_diagnostic":
        raise ValueError("run manifest owner is not guide_mala_diagnostic")
    if manifest.get("result_schema_version") != GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("run manifest has the wrong guide-MALA result schema")
    if manifest.get("status") != "diagnostic_completed":
        raise ValueError("run manifest status is not diagnostic_completed")
    _require_manifest_artifact(resolved, manifest)
    _require_manifest_paths(manifest, {"summary.json", "trace.csv"})
    implementation = manifest.get("provenance", {}).get("implementation", {})
    if implementation.get("status") != "identified":
        raise ValueError("run manifest has no identified source tree")

    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("schema_version") != GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("summary has an unsupported guide-MALA schema")
    if summary.get("status") != "diagnostic_completed":
        raise ValueError("summary status is not diagnostic_completed")
    if summary.get("all_final_walkers_valid") is not True:
        raise ValueError("summary reports invalid final walkers")
    controls = summary.get("controls")
    tail = summary.get("tail_means")
    if not isinstance(controls, dict) or not isinstance(tail, dict):
        raise ValueError("summary is missing controls or tail_means")
    if controls.get("guide_family") != "contact-corrected-reduced-tg":
        raise ValueError("summary does not use the contact-corrected reduced-TG guide")
    case = parse_case(str(summary.get("case_id")))
    if int(summary.get("n_particles", -1)) != case.n_particles or not math.isclose(
        float(summary.get("rod_length", float("nan"))),
        case.rod_length,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("summary particle count or rod length does not match its case")
    required_controls = {
        "dt",
        "walkers",
        "duration_tau",
        "steps",
        "store_every",
        "start_scale",
        "grid_extent",
        "n_bins",
        "relative_alpha",
        "guide_family",
        "contact_beta",
        "tail_tau",
    }
    if not required_controls.issubset(controls):
        missing = sorted(required_controls.difference(controls))
        raise ValueError(f"summary controls are missing: {', '.join(missing)}")
    alpha = float(controls["relative_alpha"])
    beta = float(controls["contact_beta"])
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("summary relative_alpha is invalid")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("summary contact_beta is invalid")
    required_tail = {
        "acceptance_fraction",
        "invalid_proposal_fraction",
        "configuration_esjd",
        "r2_esjd",
        "weighted_free_gap_esjd",
        "local_energy_mean",
        "local_energy_mad",
        "r2_mean",
        "weighted_free_gap_mean",
        "free_gap_p01",
    }
    if not required_tail.issubset(tail):
        missing = sorted(required_tail.difference(tail))
        raise ValueError(f"summary tail_means are missing: {', '.join(missing)}")
    if not all(math.isfinite(float(tail[name])) for name in required_tail):
        raise ValueError("summary tail_means contain non-finite values")
    if int(summary.get("tail_trace_rows", 0)) <= 0:
        raise ValueError("summary has no tail trace support")
    physics = summary.get("physics")
    if not isinstance(physics, dict) or any(
        physics.get(name) is not False
        for name in (
            "branching_enabled",
            "population_resampling_enabled",
            "walker_weights_enabled",
        )
    ):
        raise ValueError("summary is not a branching-free guide-MALA diagnostic")
    source = summary.get("guide_parameter_source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "explicit",
        "optimization_candidate",
    }:
        raise ValueError("summary has no recognized guide parameter source")
    if source["kind"] == "optimization_candidate":
        for name in ("summary_sha256", "manifest_sha256"):
            digest = source.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"summary candidate source has invalid {name}")
    if manifest.get("config") != to_jsonable(guide_mala_manifest_config(summary)):
        raise ValueError("run manifest config does not match guide-MALA summary semantics")
    return _CalibrationInput(
        path=resolved,
        manifest_path=manifest_path,
        summary=summary,
        manifest=manifest,
    )


def _require_manifest_artifact(path: Path, manifest: dict[str, Any]) -> None:
    root = path.parent.resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("summary is outside its manifest directory") from exc
    artifact_paths = {str(entry.get("path", "")) for entry in manifest.get("artifacts", [])}
    if relative not in artifact_paths:
        raise ValueError("summary is not listed by its run manifest")


def _require_manifest_paths(manifest: dict[str, Any], required: set[str]) -> None:
    artifact_paths = {
        str(entry.get("path", ""))
        for entry in manifest.get("artifacts", [])
        if isinstance(entry, dict)
    }
    missing = sorted(required.difference(artifact_paths))
    if missing:
        raise ValueError("run manifest is missing required artifacts: " + ", ".join(missing))


def _trusted_input_identity(value: _CalibrationInput) -> dict[str, Any]:
    return {
        "summary_path": str(value.path),
        "summary_sha256": file_sha256(value.path),
        "manifest_path": str(value.manifest_path),
        "manifest_sha256": file_sha256(value.manifest_path),
        "seed": value.summary["seed"],
        "start_scale": value.summary["controls"]["start_scale"],
        "source_tree_sha256": _source_tree_sha256(value.manifest),
    }


def _untrusted_input_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    identity: dict[str, Any] = {"summary_path": str(resolved)}
    if resolved.is_file():
        identity["summary_sha256"] = file_sha256(resolved)
    manifest_path = resolved.parent / "run_manifest.json"
    identity["manifest_path"] = str(manifest_path)
    if manifest_path.is_file():
        identity["manifest_sha256"] = file_sha256(manifest_path)
    return identity


def _source_tree_sha256(manifest: dict[str, Any]) -> str:
    return str(manifest["provenance"]["implementation"]["source_tree_sha256"])


def _guide_parameter_source_identity(summary: dict[str, Any]) -> dict[str, Any]:
    source = summary.get("guide_parameter_source")
    if not isinstance(source, dict):
        return {"kind": "missing"}
    return {
        name: source.get(name)
        for name in ("kind", "summary_sha256", "manifest_sha256")
        if source.get(name) is not None
    }


def _append_relative_difference_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    group: str,
    values: tuple[Any, Any],
    maximum: float,
) -> None:
    numeric = (float(values[0]), float(values[1]))
    observed = _symmetric_relative_difference(*numeric)
    _append_check(
        checks,
        name=name,
        group=group,
        observed=observed,
        criterion={"maximum_symmetric_relative_difference": maximum},
        passed=observed <= maximum,
    )


def _append_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    group: str,
    observed: Any,
    criterion: Any,
    passed: bool,
) -> None:
    checks.append(
        {
            "name": name,
            "group": group,
            "observed": to_jsonable(observed),
            "criterion": to_jsonable(criterion),
            "passed": bool(passed),
        }
    )


def _symmetric_relative_difference(first: float, second: float) -> float:
    scale = 0.5 * (abs(first) + abs(second))
    if scale == 0.0:
        return 0.0 if first == second else math.inf
    return abs(first - second) / scale


def _difference_in_max_scale(
    first: float,
    second: float,
    first_scale: float,
    second_scale: float,
) -> float:
    scale = max(abs(first_scale), abs(second_scale))
    if scale == 0.0:
        return 0.0 if first == second else math.inf
    return abs(first - second) / scale


def _positive_similarity_ratio(first: float, second: float) -> float:
    smaller = min(first, second)
    larger = max(first, second)
    if smaller <= 0.0 or larger <= 0.0:
        return math.inf
    return larger / smaller


def _positive_inverse_ratio(first: float, second: float) -> float:
    smaller = min(first, second)
    larger = max(first, second)
    if smaller <= 0.0 or larger <= 0.0:
        return 0.0
    return smaller / larger
