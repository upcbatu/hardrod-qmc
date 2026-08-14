from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import file_sha256, load_manifest_bound_artifact
from hrdmc.statistics.timestep_fit import TimeStepPoint
from hrdmc.uncertainty.timestep.contract import (
    ENERGY_CHAIN_ACCEPTED_STATUSES,
    SUPPORTED_INPUTS,
    LoadedTimeStepPoint,
)


def _load_time_step_point(summary_path: Path) -> LoadedTimeStepPoint:
    if not summary_path.is_file():
        raise FileNotFoundError(f"time-step summary does not exist: {summary_path}")
    manifest_path = summary_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
    manifest = _load_mapping(manifest_path, "run manifest")
    summary = _load_mapping(summary_path, "summary")
    warnings = _verify_summary_binding(
        summary_path=summary_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    run_name = _required_string(manifest, "run_name")
    if run_name not in SUPPORTED_INPUTS:
        raise ValueError(f"unsupported time-step input {run_name}: {summary_path}")
    if summary.get("status") != manifest.get("status"):
        raise ValueError(f"summary status does not match its manifest: {summary_path}")
    config = _required_mapping(manifest, "config")
    manifest_controls = _required_mapping(config, "controls")
    summary_controls = summary.get("controls")
    if summary_controls is None:
        controls = manifest_controls
    elif not isinstance(summary_controls, dict):
        raise ValueError(f"summary controls must be a mapping: {summary_path}")
    else:
        controls = summary_controls
        if controls != manifest_controls:
            raise ValueError(f"summary controls do not match the manifest: {summary_path}")
    _validate_step_tau_controls(controls, summary_path=summary_path)
    case_id = _case_id(summary, config, run_name=run_name)
    _verify_summary_method_identity(summary, config, summary_path=summary_path)
    stationarity, energy, stderr, energy_status = _energy_fields(
        summary,
        run_name=run_name,
        summary_path=summary_path,
    )
    _verify_conservative_stderr(stationarity, stderr, summary_path=summary_path)
    if stationarity.get("base_numerics_valid") is not True:
        raise ValueError(f"time-step point failed base numerical checks: {summary_path}")
    if stationarity.get("population_weights_controlled") is not True:
        raise ValueError(f"time-step point has uncontrolled population weights: {summary_path}")
    identity = _scientific_identity(
        summary,
        controls=controls,
        stationarity=stationarity,
        case_id=case_id,
    )
    seeds = _verified_seeds(
        summary,
        config,
        stationarity,
        run_name=run_name,
        summary_path=summary_path,
    )
    dt = _required_positive_float(controls.get("dt"), "dt")
    return LoadedTimeStepPoint(
        point=TimeStepPoint(
            dt=dt,
            energy=energy,
            conservative_stderr=stderr,
            label=str(summary_path),
        ),
        case_id=case_id,
        identity=identity,
        summary_path=summary_path,
        summary_sha256=file_sha256(summary_path),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        run_name=run_name,
        run_id=_required_string(manifest, "run_id"),
        run_status=_required_string(manifest, "status"),
        energy_status=energy_status,
        energy_quality=_energy_input_quality(stationarity, reported_status=energy_status),
        energy_quality_assessment=None,
        seeds=seeds,
        manifest_verification_warnings=tuple(warnings),
        controls=controls,
        telemetry=_point_telemetry(stationarity),
    )


def _verify_summary_binding(
    *,
    summary_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[str]:
    loaded_manifest, warnings = load_manifest_bound_artifact(
        manifest_path,
        summary_path,
        allowed_unrelated_artifact_roots=("plots",),
    )
    if loaded_manifest != manifest:
        raise ValueError(f"run manifest changed while loading: {manifest_path}")
    return list(warnings)


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
        raise ValueError("stationarity time-step summary must contain exactly one case")
    case_id = _required_string(cases[0], "case_id")
    if config.get("cases") != [case_id]:
        raise ValueError("stationarity summary case does not match its manifest")
    return case_id


def _verify_summary_method_identity(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    summary_path: Path,
) -> None:
    fields = (
        "guide_family",
        "initialization_mode",
        "init_width_log_sigma",
        "breathing_preburn_steps",
        "breathing_preburn_log_step",
    )
    for field in fields:
        if summary.get(field) != config.get(field):
            raise ValueError(f"summary {field} does not match its manifest: {summary_path}")
    summary_guide = _required_mapping(summary, "guide_parameters")
    manifest_guide = _manifest_guide_parameters(config)
    if summary_guide != manifest_guide:
        raise ValueError(f"guide identity does not match the manifest: {summary_path}")


def _manifest_guide_parameters(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("guide_parameters")
    if isinstance(nested, dict):
        return nested
    controls = _required_mapping(config, "controls")
    return {
        "relative_alpha": controls.get("relative_alpha"),
        "source": config.get("guide_parameter_source"),
    }


def _energy_fields(
    summary: dict[str, Any],
    *,
    run_name: str,
    summary_path: Path,
) -> tuple[dict[str, Any], float, float, str]:
    if run_name == "dmc_benchmark_packet":
        stationarity = _required_mapping(summary, "stationarity")
        estimates = _required_mapping(summary, "estimates")
        energy_estimate = _required_mapping(estimates, "energy")
        energy = _required_float(energy_estimate.get("value"), "energy value")
        stderr = _required_positive_float(
            energy_estimate.get("stderr"),
            "energy conservative stderr",
        )
        stationarity_energy = _required_float(
            stationarity.get("mixed_energy"),
            "stationarity mixed energy",
        )
        stationarity_stderr = _required_positive_float(
            stationarity.get("mixed_energy_conservative_stderr"),
            "stationarity conservative energy stderr",
        )
        if energy != stationarity_energy or stderr != stationarity_stderr:
            raise ValueError(
                f"benchmark energy estimate is not the stationarity conservative value: "
                f"{summary_path}"
            )
        return (
            stationarity,
            energy,
            stderr,
            str(energy_estimate.get("status", "unknown")),
        )
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
    reported: list[tuple[str, tuple[int, ...]]] = [
        ("stationarity seeds", _seed_list(stationarity.get("seeds"), "stationarity seeds")),
        (
            "stationarity seed summaries",
            _seed_ids_from_rows(
                stationarity.get("seed_summaries"),
                "stationarity seed_summaries",
            ),
        ),
    ]
    if run_name == "dmc_benchmark_packet":
        reported.extend(
            [
                ("benchmark seeds", _seed_list(summary.get("seeds"), "benchmark seeds")),
                (
                    "benchmark seed results",
                    _seed_ids_from_rows(summary.get("seed_results"), "benchmark seed_results"),
                ),
            ]
        )
    for description, seeds in reported:
        if seeds != config_seeds:
            raise ValueError(f"{description} do not match manifest seeds: {summary_path}")
    for owner in (summary, stationarity):
        seed_count = owner.get("seed_count")
        if seed_count is not None and seed_count != len(config_seeds):
            raise ValueError(f"seed_count does not match manifest seeds: {summary_path}")
    return config_seeds


def _seed_list(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list")
    return tuple(_required_int(seed, description) for seed in value)


def _seed_ids_from_rows(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{description} must contain seed mappings")
    return tuple(_required_int(row.get("seed"), description) for row in value)


def _energy_semantics(
    summary: dict[str, Any],
    stationarity: dict[str, Any],
) -> dict[str, str]:
    estimator_labels: list[str] = []
    for value in (
        stationarity.get("energy_estimator"),
        _nested_energy_estimator(summary),
    ):
        if isinstance(value, str) and value:
            estimator_labels.append(value)
    if not estimator_labels or any(
        "mixed" not in label.lower()
        or "local" not in label.lower()
        or "energy" not in label.lower()
        for label in estimator_labels
    ):
        raise ValueError("time-step extrapolation requires a mixed local-energy estimator")
    return {
        "estimator": "mixed_local_energy",
        "energy_unit": _consistent_semantic_string(
            summary,
            stationarity,
            "energy_unit",
        ),
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
    values = [value for value in (summary.get(field), stationarity.get(field)) if value is not None]
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{field} must be recorded for time-step extrapolation")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"summary and case {field} values differ")
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
    finite_components = [
        float(value)
        for value in components
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if finite_components and stderr + 1.0e-15 < max(finite_components):
        raise ValueError(
            f"declared conservative energy stderr is smaller than a component: {summary_path}"
        )


def _scientific_identity(
    summary: dict[str, Any],
    *,
    controls: dict[str, Any],
    stationarity: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    cases = summary.get("cases")
    case_summary = (
        cases[0]
        if isinstance(cases, list) and len(cases) == 1 and isinstance(cases[0], dict)
        else summary
    )
    return {
        "case_id": case_id,
        "case_parameterization": case_summary.get("case_parameterization"),
        "guide_family": summary.get("guide_family"),
        "guide_parameters": _required_mapping(summary, "guide_parameters"),
        "walkers": _required_int(controls.get("walkers"), "walkers"),
        "drift_limiter": controls.get("drift_limiter"),
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
    }


def _point_telemetry(stationarity: dict[str, Any]) -> dict[str, Any]:
    seed_rows_value = stationarity.get("seed_summaries")
    seed_rows = (
        [row for row in seed_rows_value if isinstance(row, dict)]
        if isinstance(seed_rows_value, list)
        else []
    )

    def values(field: str) -> list[float]:
        return [
            float(value)
            for row in seed_rows
            if isinstance((value := row.get(field)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]

    def mean(field: str) -> float | None:
        observed = values(field)
        return None if not observed else float(math.fsum(observed) / len(observed))

    def maximum(field: str) -> float | None:
        observed = values(field)
        return None if not observed else float(max(observed))

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
            raise ValueError(f"invalid {steps_field}/{tau_field} controls: {summary_path}")
        expected_steps = max(1, round(tau / dt))
        product_consistent = tau == 0.0 or math.isclose(
            steps * dt,
            tau,
            rel_tol=1.0e-12,
            abs_tol=0.5 * dt + 1.0e-15,
        )
        if steps != expected_steps or not product_consistent:
            raise ValueError(f"{steps_field} * dt is inconsistent with {tau_field}: {summary_path}")


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(value: Any, description: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _required_positive_float(value: Any, description: str) -> float:
    number = _required_float(value, description)
    if number <= 0.0:
        raise ValueError(f"{description} must be positive")
    return number


def _optional_finite_float(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _required_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    return value


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
    return {
        "validation_passed": validation_passed,
        "method_status": method_status,
        "energy_chain_status": chain_status,
        "precision_status": stationarity.get("precision_status"),
        "status_basis": "source_summary",
        "source_publication_accepted": publication_accepted,
        "source_publication_status": (
            "accepted_with_precision_warning"
            if precision_warning
            else "accepted"
            if publication_accepted
            else "unresolved"
        ),
        "publication_accepted": publication_accepted,
        "publication_status": (
            "accepted_with_precision_warning"
            if precision_warning
            else "accepted"
            if publication_accepted
            else "unresolved"
        ),
    }
