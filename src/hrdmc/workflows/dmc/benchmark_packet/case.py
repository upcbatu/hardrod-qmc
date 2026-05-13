from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io.artifacts import ensure_dir, write_json_atomic
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.runners import run_seed_batch
from hrdmc.workflows.dmc.benchmark_packet.selection import (
    benchmark_packet_status,
    energy_claim_status,
    pure_fw_claim_status,
    scalar_seed_mean,
    scalar_seed_stderr,
)
from hrdmc.workflows.dmc.pure_walking.case import (
    case_status,
    pure_config_metadata,
    pure_config_with_density_edges_if_needed,
)
from hrdmc.workflows.dmc.pure_walking.seed import (
    PureWalkingSeedRun,
    run_pure_walking_seed_run,
)
from hrdmc.workflows.dmc.rn_block import (
    RNCase,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
    make_grid,
    resolve_parallel_workers,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls
from hrdmc.workflows.dmc.rn_block_stationarity import summarize_stationarity_from_seed_summaries


def summarize_benchmark_packet_case(
    case: RNCase,
    controls: RNRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_no_go_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: RNInitializationControls | None = None,
    proposal: RNCollectiveProposalControls | None = None,
    proposal_family: str = "harmonic-mehler",
    guide_family: str = "auto",
) -> dict[str, Any]:
    """Run one RN-DMC packet and derive energy gates plus transported FW."""

    initialization = RNInitializationControls() if initialization is None else initialization
    proposal = RNCollectiveProposalControls() if proposal is None else proposal
    proposal.validate()
    grid = make_grid(controls, case)
    config = pure_config_with_density_edges_if_needed(pure_config, grid)
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
            proposal,
            proposal_family,
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
            proposal=proposal,
            proposal_family=proposal_family,
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
        [run.rn_summary for run in seed_runs],
        actual_workers,
        requested_worker_count=worker_count,
        trace_output_dir=trace_output_dir,
        ess_warning_fraction=ess_warning_fraction,
        ess_no_go_fraction=ess_no_go_fraction,
        log_weight_span_warning=log_weight_span_warning,
        initialization=initialization,
        proposal=proposal,
        proposal_family=proposal_family,
        guide_family=guide_family,
    )
    pure_summary = summarize_pure_seed_payloads(seed_payloads)
    energy_status = energy_claim_status(stationarity)
    pure_status = pure_fw_claim_status(pure_summary)
    status = benchmark_packet_status(energy_status=energy_status, pure_status=pure_status)
    return {
        "schema_version": "rn_block_benchmark_packet_v1",
        "status": status,
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "omega": case.omega,
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "parallel_workers_requested": worker_count,
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        **proposal.to_metadata(),
        "proposal_family": proposal_family,
        "guide_family": guide_family,
        "pure_config": pure_config_metadata(config),
        "energy_claim_status": energy_status,
        "pure_fw_claim_status": pure_status,
        "paper_values": paper_values(stationarity, pure_summary),
        "stationarity": stationarity,
        "pure_walking": pure_summary,
        "seed_results": seed_payloads,
        "claim_boundary": (
            "Single RN-DMC packet: energy from RN-DMC mixed local-energy gate; "
            "coordinate and density/pair/structure values only from transported "
            "auxiliary FW in this packet. "
            "Hellmann-Feynman energy response remains a separate cross-check workflow."
        ),
    }


def summarize_pure_seed_payloads(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    status = case_status(seed_payloads)
    observables = {
        name: summarize_pure_observable(seed_payloads, name)
        for name in seed_payloads[0]["pure_walking"]["observable_results"]
    }
    r2_values: list[float] = []
    r2_stderr_values: list[float] = []
    rms_values: list[float] = []
    schema_statuses: list[str] = []
    plateau_statuses: list[str] = []
    for payload in seed_payloads:
        r2 = payload["pure_walking"]["observable_results"].get("r2", {})
        schema_statuses.append(str(r2.get("schema_status", "")))
        plateau_statuses.append(str(r2.get("plateau_status", "")))
        r2_values.append(float(r2.get("plateau_value", float("nan"))))
        r2_stderr_values.append(float(r2.get("plateau_stderr", float("nan"))))
        rms_values.append(float(r2.get("paper_rms_radius", float("nan"))))
    pure_r2 = scalar_seed_mean(r2_values)
    pure_r2_stderr = scalar_seed_stderr(r2_values)
    paper_rms = float(np.sqrt(pure_r2)) if np.isfinite(pure_r2) and pure_r2 >= 0.0 else float("nan")
    paper_rms_stderr = (
        float(0.5 * pure_r2_stderr / paper_rms)
        if np.isfinite(pure_r2_stderr) and np.isfinite(paper_rms) and paper_rms > 0.0
        else float("nan")
    )
    return {
        "status": status,
        "seed_count": len(seed_payloads),
        "observables": observables,
        "r2_schema_statuses": schema_statuses,
        "r2_plateau_statuses": plateau_statuses,
        "pure_r2": pure_r2,
        "pure_r2_seed_stderr": pure_r2_stderr,
        "pure_r2_mean_internal_stderr": scalar_seed_mean(r2_stderr_values),
        "paper_rms_radius": paper_rms,
        "paper_rms_radius_seed_stderr": paper_rms_stderr,
        "mean_seed_rms_diagnostic": scalar_seed_mean(rms_values),
        "rms_semantics": "paper_rms_radius=sqrt(seed-aggregated pure_r2)",
    }


def summarize_pure_observable(
    seed_payloads: list[dict[str, Any]],
    observable: str,
) -> dict[str, Any]:
    results = [
        payload["pure_walking"]["observable_results"][observable]
        for payload in seed_payloads
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
        "status": "PURE_FW_GO"
        if all(result.get("plateau_status") == "PLATEAU_FOUND" for result in results)
        and all(result.get("schema_status") == "SCHEMA_GO" for result in results)
        else "PURE_FW_NO_GO",
        "plateau_statuses": [result.get("plateau_status") for result in results],
        "schema_statuses": [result.get("schema_status") for result in results],
        "plateau_diagnostics_by_seed": [
            result.get("plateau_diagnostics", {}) for result in results
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


def paper_values(stationarity: dict[str, Any], pure_summary: dict[str, Any]) -> dict[str, Any]:
    lda_rms = float(stationarity.get("lda_rms_radius", float("nan")))
    pure_r2 = float(pure_summary.get("pure_r2", float("nan")))
    pure_rms = float(pure_summary.get("paper_rms_radius", float("nan")))
    observables = pure_summary.get("observables", {})
    return {
        "energy": {
            "value": stationarity.get("mixed_energy"),
            "stderr": stationarity.get("mixed_energy_conservative_stderr"),
            "estimator": "RN-DMC mixed local energy",
            "status": energy_claim_status(stationarity),
            "lda_value": stationarity.get("lda_total_energy"),
            "delta_vs_lda": stationarity.get("energy_dmc_minus_lda"),
        },
        "r2": {
            "value": pure_r2,
            "stderr": pure_summary.get("pure_r2_seed_stderr"),
            "estimator": "transported auxiliary forward walking",
            "status": pure_fw_claim_status(pure_summary),
            "mixed_diagnostic": stationarity.get("r2_radius"),
            "lda_value": lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
            "delta_vs_lda": pure_r2 - lda_rms * lda_rms if np.isfinite(lda_rms) else float("nan"),
        },
        "rms": {
            "value": pure_rms,
            "stderr": pure_summary.get("paper_rms_radius_seed_stderr"),
            "estimator": "transported auxiliary forward walking",
            "status": pure_fw_claim_status(pure_summary),
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
        return "NOT_EVALUATED_TRANSPORTED_FW_REQUIRED"
    return str(observables[name].get("status", "PURE_FW_NO_GO"))


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
    case: RNCase,
    controls: RNRunControls,
    seed: int,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray,
    progress_queue: Any | None,
    initialization: RNInitializationControls,
    proposal: RNCollectiveProposalControls,
    proposal_family: str,
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
            proposal=proposal,
            proposal_family=proposal_family,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()
