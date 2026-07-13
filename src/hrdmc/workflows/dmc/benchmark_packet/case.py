from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.artifacts import ensure_dir, write_json_atomic
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.diagnostics import plateau_summary
from hrdmc.estimators.pure.forward_walking.results import (
    GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    GENEALOGY_SOURCE_FAMILY_DOMINANCE,
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
    LagValue,
)
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.runners import run_seed_batch
from hrdmc.workflows.dmc.benchmark_packet.selection import (
    benchmark_validation_status,
    energy_validation_status,
    pure_fw_validation_status,
    scalar_seed_mean,
    scalar_seed_stderr,
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
        "schema_version": "dmc_benchmark_packet_v2",
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
        },
    }


def summarize_pure_seed_payloads(
    seed_payloads: list[dict[str, Any]],
    *,
    config: PureWalkingConfig | None = None,
) -> dict[str, Any]:
    observables = {
        name: summarize_pure_observable(seed_payloads, name)
        for name in seed_payloads[0]["pure_walking"]["observable_results"]
    }
    r2_values: list[float] = []
    r2_stderr_values: list[float] = []
    rms_values: list[float] = []
    schema_statuses: list[str] = []
    plateau_statuses: list[str] = []
    genealogy_statuses: list[str] = []
    seed_pure_statuses: list[str] = []
    for payload in seed_payloads:
        r2 = payload["pure_walking"]["observable_results"].get("r2", {})
        seed_pure_statuses.append(str(payload["pure_walking"].get("status", "")))
        schema_statuses.append(str(r2.get("schema_status", "")))
        plateau_statuses.append(str(r2.get("plateau_status", "")))
        genealogy_statuses.append(str(r2.get("genealogy_status", "")))
        r2_values.append(float(r2.get("plateau_value", float("nan"))))
        r2_stderr_values.append(float(r2.get("plateau_stderr", float("nan"))))
        rms_values.append(float(r2.get("rms_radius", float("nan"))))
    aggregate = aggregate_r2_plateau_summary(seed_payloads, config=config)
    aggregate_value = float(aggregate.get("plateau_value", float("nan")))
    aggregate_stderr = float(aggregate.get("plateau_stderr", float("nan")))
    seed_r2_stderr = scalar_seed_stderr(r2_values)
    pure_r2 = aggregate_value if np.isfinite(aggregate_value) else scalar_seed_mean(r2_values)
    pure_r2_stderr = _max_finite(aggregate_stderr, seed_r2_stderr)
    rms_radius = (
        float(np.sqrt(pure_r2)) if np.isfinite(pure_r2) and pure_r2 >= 0.0 else float("nan")
    )
    rms_radius_stderr = (
        float(0.5 * pure_r2_stderr / rms_radius)
        if np.isfinite(pure_r2_stderr) and np.isfinite(rms_radius) and rms_radius > 0.0
        else float("nan")
    )
    status = pure_case_status_from_aggregate(
        schema_statuses=schema_statuses,
        seed_pure_statuses=seed_pure_statuses,
        aggregate_status=str(aggregate.get("plateau_status", "")),
    )
    if "r2" in observables:
        observables["r2"] = {
            **observables["r2"],
            "status": status,
            "decision_level": "seed_aggregated",
            "aggregate_plateau_status": aggregate.get("plateau_status"),
            "aggregate_plateau_value": aggregate.get("plateau_value"),
            "aggregate_plateau_stderr": aggregate.get("plateau_stderr"),
            "aggregate_plateau_diagnostics": aggregate.get("plateau_diagnostics"),
        }
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
        "pure_r2_seed_stderr": seed_r2_stderr,
        "pure_r2_aggregate_plateau_stderr": aggregate_stderr,
        "pure_r2_mean_internal_stderr": scalar_seed_mean(r2_stderr_values),
        "rms_radius": rms_radius,
        "rms_radius_stderr": rms_radius_stderr,
        "rms_radius_seed_stderr": (
            float(0.5 * seed_r2_stderr / rms_radius)
            if np.isfinite(seed_r2_stderr) and np.isfinite(rms_radius) and rms_radius > 0.0
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
    if not seed_payloads:
        return {
            "plateau_status": PLATEAU_NO_BLOCKS,
            "plateau_value": float("nan"),
            "plateau_stderr": float("nan"),
            "plateau_diagnostics": {"reason": "no_seed_payloads"},
        }
    first_r2 = seed_payloads[0]["pure_walking"]["observable_results"].get("r2", {})
    if config is None:
        config = _r2_plateau_config_from_seed_result(first_r2)
    lag_steps = tuple(int(lag) for lag in config.lag_steps)
    values_by_lag: dict[int, LagValue] = {}
    stderr_by_lag: dict[int, LagValue] = {}
    block_count_by_lag: dict[int, int] = {}
    weight_ess_min_by_lag: dict[int, float] = {}
    for lag in lag_steps:
        seed_values: list[float] = []
        seed_stderrs: list[float] = []
        block_counts: list[int] = []
        weight_ess: list[float] = []
        for payload in seed_payloads:
            r2 = payload["pure_walking"]["observable_results"].get("r2", {})
            value = _lag_dict_float(r2.get("values_by_lag", {}), lag)
            stderr = _lag_dict_float(r2.get("stderr_by_lag", {}), lag)
            if np.isfinite(value):
                seed_values.append(value)
            if np.isfinite(stderr):
                seed_stderrs.append(stderr)
            block_count = _lag_dict_int(r2.get("block_count_by_lag", {}), lag)
            if block_count is not None:
                block_counts.append(block_count)
            ess = _lag_dict_float(r2.get("block_weight_ess_min_by_lag", {}), lag)
            if np.isfinite(ess):
                weight_ess.append(ess)
        values = np.asarray(seed_values, dtype=float)
        stderrs = np.asarray(seed_stderrs, dtype=float)
        values_by_lag[lag] = float(np.mean(values)) if values.size else float("nan")
        seed_stderr = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size >= 2
            else float("nan")
        )
        internal_stderr = (
            float(np.sqrt(np.sum(stderrs * stderrs)) / stderrs.size)
            if stderrs.size
            else float("nan")
        )
        stderr_by_lag[lag] = _max_finite(seed_stderr, internal_stderr)
        block_count_by_lag[lag] = min(block_counts) if block_counts else 0
        weight_ess_min_by_lag[lag] = min(weight_ess) if weight_ess else 0.0
    value, stderr, bracket, plateau_status, diagnostics = plateau_summary(
        config=config,
        observable="r2",
        values_by_lag=values_by_lag,
        stderr_by_lag=stderr_by_lag,
        block_count_by_lag=block_count_by_lag,
        weight_ess_min_by_lag=weight_ess_min_by_lag,
    )
    diagnostics = {
        **diagnostics,
        "decision_level": "seed_aggregated",
        "seed_count": len(seed_payloads),
        "seed_plateau_statuses": [
            payload["pure_walking"]["observable_results"].get("r2", {}).get("plateau_status", "")
            for payload in seed_payloads
        ],
        "values_by_lag": values_by_lag,
        "stderr_by_lag": stderr_by_lag,
    }
    return {
        "plateau_status": plateau_status,
        "plateau_value": float(value) if value is not None else float("nan"),
        "plateau_stderr": float(stderr) if stderr is not None else float("nan"),
        "bias_bracket": None if bracket is None else [float(bracket[0]), float(bracket[1])],
        "plateau_diagnostics": diagnostics,
    }


def pure_case_status_from_aggregate(
    *,
    schema_statuses: list[str],
    seed_pure_statuses: list[str],
    aggregate_status: str,
) -> str:
    if any(status != SCHEMA_VALID for status in schema_statuses):
        return PURE_STATUS_SCHEMA_INVALID
    if PURE_STATUS_INSUFFICIENT_SAMPLES in seed_pure_statuses:
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    if aggregate_status == PLATEAU_RESOLVED:
        return PURE_STATUS_ACCEPTED
    if aggregate_status == PLATEAU_NO_BLOCKS:
        return PURE_STATUS_NO_BLOCKS
    if aggregate_status in {
        PLATEAU_INSUFFICIENT_BLOCKS,
        PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    }:
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    return PURE_STATUS_PLATEAU_UNRESOLVED


def _r2_plateau_config_from_seed_result(r2: dict[str, Any]) -> PureWalkingConfig:
    metadata = r2.get("metadata", {})
    return PureWalkingConfig(
        lag_steps=tuple(int(lag) for lag in r2.get("lag_steps", (0,))),
        observables=("r2",),
        min_block_count=int(metadata.get("min_block_count", 30)),
        min_walker_weight_ess=float(metadata.get("min_walker_weight_ess", 30.0)),
        min_source_ancestor_ess=float(metadata.get("min_source_ancestor_ess", 1.0)),
        max_source_family_fraction=float(metadata.get("max_source_family_fraction", 1.0)),
        plateau_sigma_threshold=float(metadata.get("plateau_sigma_threshold", 1.0)),
        plateau_abs_tolerance=float(metadata.get("plateau_abs_tolerance", 0.0)),
        plateau_window_lag_count=int(metadata.get("plateau_window_lag_count", 4)),
    )


def _lag_dict_float(values: Any, lag: int) -> float:
    if not isinstance(values, dict):
        return float("nan")
    value = values.get(lag, values.get(str(lag), float("nan")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _lag_dict_int(values: Any, lag: int) -> int | None:
    if not isinstance(values, dict):
        return None
    value = values.get(lag, values.get(str(lag)))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_finite(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else float("nan")


def summarize_pure_observable(
    seed_payloads: list[dict[str, Any]],
    observable: str,
) -> dict[str, Any]:
    results = [
        payload["pure_walking"]["observable_results"][observable] for payload in seed_payloads
    ]
    values = np.asarray([result["plateau_value"] for result in results], dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    finite = np.all(np.isfinite(values), axis=1)
    finite_values = values[finite]
    mean_value = (
        np.mean(finite_values, axis=0)
        if finite_values.size
        else np.full(values.shape[1], np.nan, dtype=float)
    )
    stderr_value = (
        np.std(finite_values, axis=0, ddof=1) / np.sqrt(finite_values.shape[0])
        if finite_values.shape[0] >= 2
        else np.full(values.shape[1], np.nan, dtype=float)
    )
    metadata = results[0].get("metadata", {})
    summary: dict[str, Any] = {
        "status": _pure_observable_status(results),
        "plateau_statuses": [result.get("plateau_status") for result in results],
        "schema_statuses": [result.get("schema_status") for result in results],
        "genealogy_statuses": [result.get("genealogy_status") for result in results],
        "plateau_diagnostics_by_seed": [
            result.get("plateau_diagnostics", {}) for result in results
        ],
        "genealogy_diagnostics_by_seed": [
            result.get("genealogy_diagnostics", {}) for result in results
        ],
        "value": _result_value(mean_value),
        "stderr": _result_value(stderr_value),
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


def _pure_observable_status(results: list[dict[str, Any]]) -> str:
    if any(result.get("schema_status") != SCHEMA_VALID for result in results):
        return PURE_STATUS_SCHEMA_INVALID
    if any(
        result.get("genealogy_status")
        in {
            GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
            GENEALOGY_SOURCE_FAMILY_DOMINANCE,
        }
        for result in results
    ):
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    if all(result.get("plateau_status") == PLATEAU_RESOLVED for result in results):
        return PURE_STATUS_ACCEPTED
    if any(result.get("plateau_status") == PLATEAU_NO_BLOCKS for result in results):
        return PURE_STATUS_NO_BLOCKS
    if any(
        result.get("plateau_status")
        in {
            PLATEAU_INSUFFICIENT_BLOCKS,
            PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
        }
        for result in results
    ):
        return PURE_STATUS_INSUFFICIENT_SAMPLES
    return PURE_STATUS_PLATEAU_UNRESOLVED


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
            "estimator": "transported auxiliary forward walking",
            "status": pure_fw_validation_status(pure_summary),
            "mixed_diagnostic": stationarity.get("r2_radius"),
            "lda_value": lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
            "delta_vs_lda": pure_r2 - lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
        },
        "rms": {
            "value": pure_rms,
            "stderr": pure_summary.get("rms_radius_stderr"),
            "estimator": "transported auxiliary forward walking",
            "status": pure_fw_validation_status(pure_summary),
            "mixed_diagnostic": stationarity.get("rms_radius"),
            "lda_value": lda_rms,
            "delta_vs_lda": pure_rms - lda_rms if np.isfinite(lda_rms) else float("nan"),
        },
        "density": {
            "status": _observable_status(observables, "density"),
            "estimator": "transported auxiliary forward walking",
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


def _result_value(value: np.ndarray) -> float | list[float]:
    if value.shape == (1,):
        return float(value[0])
    return value.tolist()


def _observable_status(observables: dict[str, Any], name: str) -> str:
    if name not in observables:
        return "not_evaluated_transported_fw_required"
    return str(observables[name].get("status", "not_evaluated"))


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
