from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.analysis import relative_density_l2_error
from hrdmc.artifacts import ensure_dir, write_json_atomic
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.results import (
    PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    PLATEAU_INSUFFICIENT_BLOCKS,
    PLATEAU_NO_BLOCKS,
    PLATEAU_RESOLVED,
    PLATEAU_UNRESOLVED,
    PURE_STATUS_ACCEPTED,
    PURE_STATUS_INSUFFICIENT_SAMPLES,
    PURE_STATUS_NO_BLOCKS,
    PURE_STATUS_PLATEAU_UNRESOLVED,
    PURE_STATUS_SCHEMA_INVALID,
    SCHEMA_VALID,
)
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.runners import run_seed_batch
from hrdmc.workflows.dmc.benchmark_packet.fw_aggregation import (
    aggregate_fw_observable_summary,
)
from hrdmc.workflows.dmc.benchmark_packet.selection import (
    benchmark_validation_status,
    energy_validation_status,
    pure_fw_validation_status,
    scalar_seed_mean,
)
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.pure_walking.case import (
    pure_config_for_case,
    pure_config_metadata,
)
from hrdmc.workflows.dmc.pure_walking.seed import (
    PureWalkingSeedRun,
    run_pure_walking_seed_run,
)
from hrdmc.workflows.dmc.stationarity import summarize_stationarity_from_seed_summaries
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    controls_to_dict,
    make_grid,
    resolve_parallel_workers,
)

BENCHMARK_PACKET_SCHEMA_VERSION = "dmc_benchmark_packet_v3"


def summarize_benchmark_packet_case(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_invalid_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Run one DMC packet, optionally with collective RN transport, and derive FW."""

    initialization = InitializationControls() if initialization is None else initialization
    grid = make_grid(controls, case)
    config = pure_config_for_case(pure_config, grid=grid, case=case)
    worker_count = resolve_parallel_workers(len(seeds), parallel_workers)
    seed_runs, actual_workers = run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            _run_benchmark_seed_worker,
            case,
            controls,
            seed,
            config,
            grid,
            progress_queue,
            initialization,
            collective_rn,
            guide_family,
        ),
        run_serial_seed=lambda seed: run_pure_walking_seed_run(
            case,
            controls,
            seed,
            pure_config=config,
            density_grid=grid,
            progress=progress,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=guide_family,
        ),
    )
    seed_payloads = [run.to_payload() for run in seed_runs]
    if trace_output_dir is not None:
        write_seed_payload_checkpoints(trace_output_dir, case.case_id, seed_payloads)
    stationarity = summarize_stationarity_from_seed_summaries(
        case,
        controls,
        seeds,
        grid,
        [run.dmc_summary for run in seed_runs],
        actual_workers,
        requested_worker_count=worker_count,
        trace_output_dir=trace_output_dir,
        ess_warning_fraction=ess_warning_fraction,
        ess_invalid_fraction=ess_invalid_fraction,
        log_weight_span_warning=log_weight_span_warning,
        initialization=initialization,
        collective_rn=collective_rn,
        guide_family=guide_family,
    )
    pure_summary = summarize_pure_seed_payloads(seed_payloads, config=config)
    energy_status = energy_validation_status(stationarity)
    pure_status = pure_fw_validation_status(pure_summary)
    status = benchmark_validation_status(energy_status=energy_status, pure_status=pure_status)
    calculation_name = (
        "DMC with scheduled collective RN moves" if collective_rn is not None else "DMC"
    )
    return {
        "schema_version": BENCHMARK_PACKET_SCHEMA_VERSION,
        "status": status,
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        **case.unit_metadata(),
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "parallel_workers_requested": worker_count,
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "collective_rn_controls": (None if collective_rn is None else collective_rn.to_metadata()),
        "guide_family": guide_family,
        "pure_config": pure_config_metadata(config),
        "energy_validation_status": energy_status,
        "pure_fw_validation_status": pure_status,
        "estimates": estimates(
            stationarity,
            pure_summary,
            calculation_name=calculation_name,
        ),
        "stationarity": stationarity,
        "pure_walking": pure_summary,
        "seed_results": seed_payloads,
        "method": {
            "energy": f"{calculation_name} mixed local energy",
            "coordinate_observables": "transported auxiliary forward walking",
            "r2": _fw_estimator_label(pure_summary.get("observables", {}), "r2"),
            "density": _fw_estimator_label(pure_summary.get("observables", {}), "density"),
        },
    }


def summarize_pure_seed_payloads(
    seed_payloads: list[dict[str, Any]],
    *,
    config: PureWalkingConfig | None = None,
) -> dict[str, Any]:
    if not seed_payloads:
        raise ValueError("seed_payloads must contain at least one independent seed")
    if config is None:
        first_observables = seed_payloads[0]["pure_walking"]["observable_results"]
        observable_names = tuple(first_observables)
        observable_configs = {
            name: _plateau_config_from_seed_result(result, name)
            for name, result in first_observables.items()
        }
    else:
        observable_names = config.observables
        observable_configs = {name: config for name in observable_names}
    aggregates = {
        name: aggregate_fw_observable_summary(
            seed_payloads,
            observable=name,
            config=observable_configs[name],
        )
        for name in observable_names
    }
    observables = {
        name: summarize_pure_observable(seed_payloads, name, aggregate=aggregates[name])
        for name in observable_names
    }
    r2_stderr_values: list[float] = []
    rms_values: list[float] = []
    schema_statuses: list[str] = []
    plateau_statuses: list[str] = []
    genealogy_statuses: list[str] = []
    for payload in seed_payloads:
        r2 = payload["pure_walking"]["observable_results"].get("r2", {})
        schema_statuses.append(str(r2.get("schema_status", "")))
        plateau_statuses.append(str(r2.get("plateau_status", "")))
        genealogy_statuses.append(str(r2.get("genealogy_status", "")))
        r2_stderr_values.append(float(r2.get("plateau_stderr", float("nan"))))
        rms_values.append(float(r2.get("rms_radius", float("nan"))))
    aggregate = aggregates.get("r2", {})
    aggregate_value = _optional_float(aggregate.get("plateau_value"))
    aggregate_stderr = _optional_float(aggregate.get("plateau_stderr"))
    pure_r2 = aggregate_value
    pure_r2_stderr = aggregate_stderr
    rms_radius = (
        float(np.sqrt(pure_r2)) if np.isfinite(pure_r2) and pure_r2 >= 0.0 else float("nan")
    )
    rms_radius_stderr = (
        float(0.5 * pure_r2_stderr / rms_radius)
        if np.isfinite(pure_r2_stderr) and np.isfinite(rms_radius) and rms_radius > 0.0
        else float("nan")
    )
    status = _pure_case_status_from_observables(observables)
    return {
        "status": status,
        "seed_count": len(seed_payloads),
        "observables": observables,
        "r2_schema_statuses": schema_statuses,
        "r2_plateau_statuses": plateau_statuses,
        "r2_genealogy_statuses": genealogy_statuses,
        "r2_seed_plateau_resolved_count": plateau_statuses.count(PLATEAU_RESOLVED),
        "r2_seed_plateau_unresolved_count": sum(
            status == PLATEAU_UNRESOLVED for status in plateau_statuses
        ),
        "r2_aggregate_plateau_status": aggregate.get("plateau_status"),
        "r2_aggregate_plateau_value": aggregate.get("plateau_value"),
        "r2_aggregate_plateau_stderr": aggregate.get("plateau_stderr"),
        "r2_aggregate_plateau_diagnostics": aggregate.get("plateau_diagnostics"),
        "pure_r2": pure_r2,
        "pure_r2_stderr": pure_r2_stderr,
        "pure_r2_seed_stderr": aggregate_stderr,
        "pure_r2_aggregate_plateau_stderr": aggregate_stderr,
        "pure_r2_mean_internal_stderr": scalar_seed_mean(r2_stderr_values),
        "rms_radius": rms_radius,
        "rms_radius_stderr": rms_radius_stderr,
        "rms_radius_seed_stderr": (
            float(0.5 * aggregate_stderr / rms_radius)
            if np.isfinite(aggregate_stderr) and np.isfinite(rms_radius) and rms_radius > 0.0
            else float("nan")
        ),
        "mean_seed_rms_diagnostic": scalar_seed_mean(rms_values),
        "rms_semantics": "rms_radius=sqrt(seed-aggregated pure_r2)",
    }


def aggregate_r2_plateau_summary(
    seed_payloads: list[dict[str, Any]],
    *,
    config: PureWalkingConfig | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the workflow-owned generic FW aggregation."""

    if not seed_payloads:
        return aggregate_fw_observable_summary(
            seed_payloads,
            observable="r2",
            config=PureWalkingConfig(lag_steps=(0,)),
        )
    first_r2 = seed_payloads[0]["pure_walking"]["observable_results"].get("r2", {})
    if config is None:
        config = _r2_plateau_config_from_seed_result(first_r2)
    return aggregate_fw_observable_summary(
        seed_payloads,
        observable="r2",
        config=config,
    )


def _r2_plateau_config_from_seed_result(r2: dict[str, Any]) -> PureWalkingConfig:
    return _plateau_config_from_seed_result(r2, "r2")


def _plateau_config_from_seed_result(
    result: dict[str, Any],
    observable: str,
) -> PureWalkingConfig:
    metadata = result.get("metadata", {})
    return PureWalkingConfig(
        lag_steps=tuple(int(lag) for lag in result.get("lag_steps", (0,))),
        observables=(observable,),
        density_bin_edges=(
            np.asarray(metadata.get("bin_edges"), dtype=float)
            if observable == "density" and metadata.get("bin_edges") is not None
            else None
        ),
        min_block_count=int(metadata.get("min_block_count", 30)),
        min_walker_weight_ess=float(metadata.get("min_walker_weight_ess", 30.0)),
        min_source_ancestor_ess=float(metadata.get("min_source_ancestor_ess", 1.0)),
        max_source_family_fraction=float(metadata.get("max_source_family_fraction", 1.0)),
        plateau_sigma_threshold=float(metadata.get("plateau_sigma_threshold", 1.0)),
        plateau_abs_tolerance=float(metadata.get("plateau_abs_tolerance", 0.0)),
        rms_plateau_relative_tolerance=float(metadata.get("rms_plateau_relative_tolerance", 0.0)),
        plateau_equivalence_confidence_level=float(
            metadata.get("plateau_equivalence_confidence_level", 0.95)
        ),
        plateau_window_lag_count=int(metadata.get("plateau_window_lag_count", 4)),
        density_plateau_relative_l2_tolerance=float(
            metadata.get("density_plateau_relative_l2_tolerance", 0.03)
        ),
        density_expected_particles=(
            float(metadata["density_expected_particles"])
            if metadata.get("density_expected_particles") is not None
            else None
        ),
        density_accounting_abs_tolerance=float(
            metadata.get("density_accounting_abs_tolerance", 5.0e-3)
        ),
    )


def summarize_pure_observable(
    seed_payloads: list[dict[str, Any]],
    observable: str,
    *,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    results = [
        payload.get("pure_walking", {}).get("observable_results", {}).get(observable, {})
        for payload in seed_payloads
    ]
    mean_value = np.asarray(aggregate.get("plateau_value"), dtype=float).reshape(-1)
    stderr_value = np.asarray(aggregate.get("plateau_stderr"), dtype=float).reshape(-1)
    metadata = next(
        (result.get("metadata", {}) for result in results if result),
        {},
    )
    summary: dict[str, Any] = {
        "status": _aggregate_observable_status(aggregate),
        "decision_level": "independent_seed_aggregate",
        "plateau_statuses": [result.get("plateau_status") for result in results],
        "schema_statuses": [result.get("schema_status") for result in results],
        "genealogy_statuses": [result.get("genealogy_status") for result in results],
        "plateau_diagnostics_by_seed": [
            result.get("plateau_diagnostics", {}) for result in results
        ],
        "genealogy_diagnostics_by_seed": [
            result.get("genealogy_diagnostics", {}) for result in results
        ],
        "aggregate_plateau_status": aggregate.get("plateau_status"),
        "aggregate_genealogy_status": aggregate.get("genealogy_status"),
        "aggregate_schema_status": aggregate.get("schema_status"),
        "aggregate_plateau_diagnostics": aggregate.get("plateau_diagnostics", {}),
        "value": _result_value(mean_value, scalar=observable == "r2"),
        "stderr": _result_value(stderr_value, scalar=observable == "r2"),
        "metadata": metadata,
    }
    if observable in {"density", "pair_distance_density"}:
        edges = np.asarray(metadata.get("bin_edges", []), dtype=float)
        if edges.ndim == 1 and edges.size == mean_value.size + 1:
            widths = np.diff(edges)
            summary["bin_edges"] = edges.tolist()
            summary["x"] = (0.5 * (edges[:-1] + edges[1:])).tolist()
            summary["integral"] = float(np.sum(mean_value * widths))
    if observable == "structure_factor":
        summary["k_values"] = metadata.get("k_values", [])
    return summary


def _aggregate_observable_status(aggregate: dict[str, Any]) -> str:
    if aggregate.get("schema_status") != SCHEMA_VALID:
        return PURE_STATUS_SCHEMA_INVALID
    plateau_status = aggregate.get("plateau_status")
    if plateau_status == PLATEAU_RESOLVED:
        return PURE_STATUS_ACCEPTED
    if plateau_status == PLATEAU_NO_BLOCKS:
        return PURE_STATUS_NO_BLOCKS
    if plateau_status in {
        PLATEAU_INSUFFICIENT_BLOCKS,
        PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    }:
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    return PURE_STATUS_PLATEAU_UNRESOLVED


def _pure_case_status_from_observables(observables: dict[str, dict[str, Any]]) -> str:
    statuses = {str(result.get("status")) for result in observables.values()}
    for status in (
        PURE_STATUS_SCHEMA_INVALID,
        PURE_STATUS_NO_BLOCKS,
        PURE_STATUS_INSUFFICIENT_SAMPLES,
        PURE_STATUS_PLATEAU_UNRESOLVED,
    ):
        if status in statuses:
            return status
    return PURE_STATUS_ACCEPTED if statuses == {PURE_STATUS_ACCEPTED} else "observable_disagreement"


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_none(value: Any) -> float | None:
    result = _optional_float(value)
    return result if np.isfinite(result) else None


def estimates(
    stationarity: dict[str, Any],
    pure_summary: dict[str, Any],
    *,
    calculation_name: str,
) -> dict[str, Any]:
    lda_rms = float(stationarity.get("lda_rms_radius", float("nan")))
    pure_r2 = float(pure_summary.get("pure_r2", float("nan")))
    pure_rms = float(pure_summary.get("rms_radius", float("nan")))
    observables = pure_summary.get("observables", {})
    r2_diagnostics = pure_summary.get("r2_aggregate_plateau_diagnostics", {})
    if not isinstance(r2_diagnostics, dict):
        r2_diagnostics = {}
    density_diagnostics = observables.get("density", {}).get("aggregate_plateau_diagnostics", {})
    if not isinstance(density_diagnostics, dict):
        density_diagnostics = {}
    lag_systematic = _finite_or_none(r2_diagnostics.get("simultaneous_rms_upper_bound"))
    relative_lag_systematic = _finite_or_none(
        r2_diagnostics.get("simultaneous_rms_relative_upper_bound")
    )
    lag_margin = _finite_or_none(r2_diagnostics.get("rms_relative_equivalence_margin"))
    lag_confidence = _finite_or_none(r2_diagnostics.get("confidence_level"))
    rms_stderr = pure_summary.get("rms_radius_stderr")
    density_comparisons = _density_comparison_metrics(observables, stationarity)
    return {
        "energy": {
            "value": stationarity.get("mixed_energy"),
            "stderr": stationarity.get("mixed_energy_conservative_stderr"),
            "estimator": f"{calculation_name} mixed local energy",
            "status": energy_validation_status(stationarity),
            "lda_value": stationarity.get("lda_total_energy"),
            "delta_vs_lda": stationarity.get("energy_dmc_minus_lda"),
        },
        "r2": {
            "value": pure_r2,
            "stderr": pure_summary.get("pure_r2_stderr"),
            "estimator": _fw_estimator_label(observables, "r2"),
            "status": _observable_status(observables, "r2"),
            "mixed_diagnostic": stationarity.get("r2_radius"),
            "lda_value": lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
            "delta_vs_lda": pure_r2 - lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
            "rms_lag_systematic_upper_bound": lag_systematic,
            "rms_lag_systematic_relative_upper_bound": relative_lag_systematic,
        },
        "rms": {
            "value": pure_rms,
            "stderr": rms_stderr,
            "mc_statistical_stderr": rms_stderr,
            "fw_lag_systematic_upper_bound": lag_systematic,
            "fw_lag_systematic_relative_upper_bound": relative_lag_systematic,
            "fw_lag_relative_equivalence_margin": lag_margin,
            "fw_lag_equivalence_confidence_level": lag_confidence,
            "uncertainty_semantics": (
                "mc_statistical_stderr and fw_lag_systematic_upper_bound are reported "
                "separately and are not combined"
            ),
            "estimator": _fw_estimator_label(observables, "r2"),
            "status": _observable_status(observables, "r2"),
            "mixed_diagnostic": stationarity.get("rms_radius"),
            "lda_value": lda_rms,
            "delta_vs_lda": pure_rms - lda_rms if np.isfinite(lda_rms) else float("nan"),
        },
        "density": {
            "status": _observable_status(observables, "density"),
            "estimator": _fw_estimator_label(observables, "density"),
            "value": observables.get("density", {}).get("value"),
            "stderr": observables.get("density", {}).get("stderr"),
            "x": observables.get("density", {}).get("x"),
            "bin_edges": observables.get("density", {}).get("bin_edges"),
            "integral": observables.get("density", {}).get("integral"),
            "lda_x": stationarity.get("density_profile", {}).get("x"),
            "lda_value": stationarity.get("density_profile", {}).get("lda_n_x"),
            "mixed_diagnostic_x": stationarity.get("density_profile", {}).get("x"),
            "mixed_diagnostic_value": stationarity.get("density_profile", {}).get("mixed_n_x"),
            "mixed_diagnostic_density_l2": stationarity.get("density_relative_l2"),
            "fw_lag_systematic_relative_l2_upper_bound": _finite_or_none(
                density_diagnostics.get("simultaneous_density_relative_l2_upper_bound")
            ),
            "fw_lag_relative_l2_equivalence_margin": _finite_or_none(
                density_diagnostics.get("density_relative_l2_equivalence_margin")
            ),
            "fw_lag_equivalence_confidence_level": _finite_or_none(
                density_diagnostics.get("confidence_level")
            ),
            "comparison_semantics": (
                "FW-versus-LDA and FW-versus-mixed relative L2 values are descriptive "
                "shape discrepancies, not acceptance criteria"
            ),
            **density_comparisons,
        },
        "pair_distance_density": {
            "status": _observable_status(observables, "pair_distance_density"),
            "estimator": "transported auxiliary forward walking",
            "value": observables.get("pair_distance_density", {}).get("value"),
            "stderr": observables.get("pair_distance_density", {}).get("stderr"),
            "x": observables.get("pair_distance_density", {}).get("x"),
            "bin_edges": observables.get("pair_distance_density", {}).get("bin_edges"),
            "integral": observables.get("pair_distance_density", {}).get("integral"),
        },
        "structure_factor": {
            "status": _observable_status(observables, "structure_factor"),
            "estimator": "transported auxiliary forward walking",
            "value": observables.get("structure_factor", {}).get("value"),
            "stderr": observables.get("structure_factor", {}).get("stderr"),
            "k_values": observables.get("structure_factor", {}).get("k_values"),
        },
    }


def _density_comparison_metrics(
    observables: dict[str, Any],
    stationarity: dict[str, Any],
) -> dict[str, float | None]:
    density = observables.get("density", {})
    profile = stationarity.get("density_profile", {})
    try:
        x = np.asarray(density.get("x"), dtype=float)
        value = np.asarray(density.get("value"), dtype=float)
        lda_x = np.asarray(profile.get("x"), dtype=float)
        lda_value = np.asarray(profile.get("lda_n_x"), dtype=float)
        mixed_value = np.asarray(profile.get("mixed_n_x"), dtype=float)
    except (TypeError, ValueError):
        return {
            "fw_relative_l2_vs_lda": None,
            "fw_relative_l2_vs_mixed": None,
        }
    if (
        x.ndim != 1
        or x.size < 2
        or value.shape != x.shape
        or lda_x.ndim != 1
        or lda_x.size < 2
        or lda_value.shape != lda_x.shape
        or mixed_value.shape != lda_x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(lda_x))
        or not np.all(np.isfinite(lda_value))
        or not np.all(np.isfinite(mixed_value))
        or not np.all(np.diff(x) > 0.0)
        or not np.all(np.diff(lda_x) > 0.0)
    ):
        return {
            "fw_relative_l2_vs_lda": None,
            "fw_relative_l2_vs_mixed": None,
        }
    lda_on_x = np.interp(x, lda_x, lda_value)
    mixed_on_x = np.interp(x, lda_x, mixed_value)
    try:
        return {
            "fw_relative_l2_vs_lda": relative_density_l2_error(x, value, lda_on_x),
            "fw_relative_l2_vs_mixed": relative_density_l2_error(x, value, mixed_on_x),
        }
    except ValueError:
        return {
            "fw_relative_l2_vs_lda": None,
            "fw_relative_l2_vs_mixed": None,
        }


def _result_value(value: np.ndarray, *, scalar: bool) -> float | list[float]:
    if scalar and value.shape == (1,):
        return float(value[0])
    return value.tolist()


def _observable_status(observables: dict[str, Any], name: str) -> str:
    if name not in observables:
        return "not_evaluated_transported_fw_required"
    return str(observables[name].get("status", "not_evaluated"))


def _fw_estimator_label(observables: dict[str, Any], name: str) -> str:
    metadata = observables.get(name, {}).get("metadata", {})
    if name == "r2" and metadata.get("observable_source") == "r2_rb":
        return "transported auxiliary forward walking with exact COM Rao-Blackwellization"
    if name == "density" and metadata.get("density_source") == "com_rao_blackwell":
        suffix = " and exact parity averaging" if metadata.get("density_parity_average") else ""
        return f"transported relative-coordinate forward walking with exact COM integration{suffix}"
    return "transported auxiliary forward walking"


def write_seed_payload_checkpoints(
    output_dir: Any,
    case_id: str,
    seed_payloads: list[dict[str, Any]],
) -> None:
    """Write per-seed payloads before top-level aggregation and plotting."""

    root = ensure_dir(output_dir) / "seed_payloads" / case_id
    ensure_dir(root)
    for payload in seed_payloads:
        seed = int(payload["seed"])
        write_json_atomic(root / f"seed_{seed}.json", payload)


def _run_benchmark_seed_worker(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray,
    progress_queue: Any | None,
    initialization: InitializationControls,
    collective_rn: CollectiveRNControls | None,
    guide_family: str,
) -> tuple[int, PureWalkingSeedRun]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, run_pure_walking_seed_run(
            case,
            controls,
            seed,
            pure_config=pure_config,
            density_grid=density_grid,
            progress=worker_progress,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()
