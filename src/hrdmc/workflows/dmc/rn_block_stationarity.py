from __future__ import annotations

from typing import Any, cast

import numpy as np

from hrdmc.analysis import (
    blocking_curve,
    detect_blocking_plateau,
    diagnose_chains,
    relative_density_l2_error,
)
from hrdmc.io.artifacts import ensure_dir
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.monte_carlo.dmc.rn_block import RNBlockStreamingSummary
from hrdmc.runners import run_seed_batch
from hrdmc.theory import lda_density_profile, lda_rms_radius, lda_total_energy
from hrdmc.workflows.dmc.rn_block import (
    RNCase,
    RNRunControls,
    build_case_objects,
    make_grid,
    resolve_parallel_workers,
    run_streaming_seed,
)

GO_CLASSIFICATIONS = {"GO", "WARNING_SPREAD_ONLY"}


def summarize_stationarity_case(
    case: RNCase,
    controls: RNRunControls,
    seeds: list[int],
    *,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_no_go_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
) -> dict[str, Any]:
    grid = make_grid(controls, case)
    worker_count = resolve_parallel_workers(len(seeds), parallel_workers)
    seed_summaries, actual_worker_count = run_stationarity_seeds(
        case,
        controls,
        seeds,
        grid,
        worker_count=worker_count,
        progress=progress,
    )
    system, trap, _guide, _target, _proposal = build_case_objects(case)
    density = np.mean([summary.density for summary in seed_summaries], axis=0)
    energy_values = np.asarray([summary.mixed_energy for summary in seed_summaries])
    rms_values = np.asarray([summary.rms_radius for summary in seed_summaries])
    r2_values = np.asarray([summary.r2_radius for summary in seed_summaries])
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    diagnostics = diagnose_stationarity(seed_summaries)
    spread = blocked_spread_diagnostics(seeds, seed_summaries)
    trace_artifacts = (
        write_seed_trace_artifacts(trace_output_dir, case.case_id, seeds, seed_summaries)
        if trace_output_dir is not None
        else []
    )
    density_l2_values = np.asarray(
        [relative_density_l2_error(grid, summary.density, lda.n_x) for summary in seed_summaries],
        dtype=float,
    )
    energy_uncertainty = uncertainty_summary(energy_values, diagnostics["energy"])
    rms_uncertainty = uncertainty_summary(rms_values, diagnostics["rms"])
    r2_uncertainty = uncertainty_summary(r2_values, diagnostics["r2"])
    density_integral = float(np.sum(density) * (grid[1] - grid[0]))
    density_accounting_clean = abs(density_integral - system.n_particles) <= 5e-3
    valid_finite_clean = all(
        summary.metadata["finite_local_energy_fraction"] == 1.0
        and summary.metadata["valid_snapshot_fraction"] == 1.0
        for summary in seed_summaries
    )
    ess_fraction_min = min_seed_metadata(seed_summaries, "ess_fraction_min")
    log_weight_span_max = max_seed_metadata(seed_summaries, "log_weight_span_max")
    rn_weight_status = classify_rn_weight_status(
        ess_fraction_min=ess_fraction_min,
        log_weight_span_max=log_weight_span_max,
        ess_warning_fraction=ess_warning_fraction,
        ess_no_go_fraction=ess_no_go_fraction,
        log_weight_span_warning=log_weight_span_warning,
    )
    rn_weight_controlled = rn_weight_status != "RN_WEIGHT_NO_GO"
    hygiene_gate = density_accounting_clean and valid_finite_clean and rn_weight_controlled
    old_case_gate = (
        density_accounting_clean
        and valid_finite_clean
        and rn_weight_controlled
        and all(
            diagnostics[name]["classification"] in GO_CLASSIFICATIONS
            for name in ("energy", "rms", "r2")
        )
    )
    final_classification = classify_blocked_case(
        hygiene_gate=hygiene_gate,
        rn_weight_status=rn_weight_status,
        energy=spread["energy"],
        rms=spread["rms"],
        r2=spread["r2"],
        diagnostics=diagnostics,
    )
    case_gate = final_classification in {
        "PASS_CANDIDATE",
        "WEAK_TRAP_WARNING",
        "MIXED_OBSERVABLE_WARNING",
    }
    classification = classify_case(
        case_gate=case_gate,
        density_accounting_clean=density_accounting_clean,
        valid_finite_clean=valid_finite_clean,
        rn_weight_controlled=rn_weight_controlled,
        diagnostics=diagnostics,
    )
    lda_energy = lda_total_energy(lda, rod_length=system.rod_length)
    lda_rms = lda_rms_radius(lda, center=trap.center)
    return {
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "omega": case.omega,
        "case_gate": case_gate,
        "old_case_gate": old_case_gate,
        "hygiene_gate": hygiene_gate,
        "classification": classification,
        "final_classification": final_classification,
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_worker_count,
        "parallel_workers_requested": worker_count,
        "effective_grid_extent": float(max(abs(grid[0]), abs(grid[-1]))),
        "mixed_energy": mean(energy_values),
        "mixed_energy_seed_stderr": stderr(energy_values),
        "mixed_energy_blocking_stderr": energy_uncertainty["blocking_stderr"],
        "mixed_energy_conservative_stderr": energy_uncertainty["conservative_stderr"],
        "mixed_energy_uncertainty_status": energy_uncertainty["status"],
        "rms_radius": mean(rms_values),
        "rms_radius_seed_stderr": stderr(rms_values),
        "rms_radius_blocking_stderr": rms_uncertainty["blocking_stderr"],
        "rms_radius_conservative_stderr": rms_uncertainty["conservative_stderr"],
        "rms_radius_uncertainty_status": rms_uncertainty["status"],
        "r2_radius": mean(r2_values),
        "r2_radius_seed_stderr": stderr(r2_values),
        "r2_radius_blocking_stderr": r2_uncertainty["blocking_stderr"],
        "r2_radius_conservative_stderr": r2_uncertainty["conservative_stderr"],
        "r2_radius_uncertainty_status": r2_uncertainty["status"],
        "density_integral": density_integral,
        "density_accounting_clean": density_accounting_clean,
        "valid_finite_clean": valid_finite_clean,
        "rn_weight_controlled": rn_weight_controlled,
        "lost_out_of_grid_sample_count_total": int(
            sum(summary.lost_out_of_grid_sample_count for summary in seed_summaries)
        ),
        "lda_total_energy": lda_energy,
        "energy_dmc_minus_lda": mean(energy_values) - lda_energy,
        "lda_rms_radius": lda_rms,
        "rms_dmc_minus_lda": mean(rms_values) - lda_rms,
        "density_relative_l2": relative_density_l2_error(grid, density, lda.n_x),
        "density_relative_l2_seed_stderr": stderr(density_l2_values),
        "uncertainty_status": case_uncertainty_status(
            [energy_uncertainty, rms_uncertainty, r2_uncertainty]
        ),
        "max_spread_blocking_z": max_spread_blocking_z(diagnostics),
        "blocking_plateau_energy": spread["energy"]["plateau_all"],
        "blocking_plateau_rms": spread["rms"]["plateau_all"],
        "blocking_plateau_r2": spread["r2"]["plateau_all"],
        "blocked_zscore_max_energy": spread["energy"]["blocked_zscore_max"],
        "blocked_zscore_max_rms": spread["rms"]["blocked_zscore_max"],
        "blocked_zscore_max_r2": spread["r2"]["blocked_zscore_max"],
        "robust_zscore_max_energy": spread["energy"]["robust_zscore_max"],
        "robust_zscore_max_rms": spread["rms"]["robust_zscore_max"],
        "robust_zscore_max_r2": spread["r2"]["robust_zscore_max"],
        "rhat_energy": diagnostics["energy"]["rhat"],
        "rhat_rms": diagnostics["rms"]["rhat"],
        "rhat_r2": diagnostics["r2"]["rhat"],
        "neff_energy": diagnostics["energy"]["min_effective_independent_samples"],
        "neff_rms": diagnostics["rms"]["min_effective_independent_samples"],
        "neff_r2": diagnostics["r2"]["min_effective_independent_samples"],
        "stationarity_energy": diagnostics["energy"]["classification"],
        "stationarity_rms": diagnostics["rms"]["classification"],
        "stationarity_r2": diagnostics["r2"]["classification"],
        "spread_warning_count": int(
            diagnostics["energy"]["spread_warning_count"]
            + diagnostics["rms"]["spread_warning_count"]
            + diagnostics["r2"]["spread_warning_count"]
        ),
        "ess_fraction_min": ess_fraction_min,
        "log_weight_span_max": log_weight_span_max,
        "rn_weight_status": rn_weight_status,
        "rn_weight_ess_warning_fraction": ess_warning_fraction,
        "rn_weight_ess_no_go_fraction": ess_no_go_fraction,
        "rn_weight_log_weight_span_warning": log_weight_span_warning,
        "guide_batch_backend": ",".join(
            sorted({str(summary.metadata["guide_batch_backend"]) for summary in seed_summaries})
        ),
        "trace_artifacts": trace_artifacts,
        "seed_summaries": [
            {
                "seed": seed,
                "mixed_energy": summary.mixed_energy,
                "rms_radius": summary.rms_radius,
                "r2_radius": summary.r2_radius,
                "density_integral": summary.density_integral,
                "lost_out_of_grid_sample_count": summary.lost_out_of_grid_sample_count,
                "finite_local_energy_fraction": summary.metadata["finite_local_energy_fraction"],
                "valid_snapshot_fraction": summary.metadata["valid_snapshot_fraction"],
                "killed_count": summary.metadata["killed_count"],
                "resample_count": summary.metadata["resample_count"],
                "ess_min": summary.metadata["ess_min"],
                "ess_mean": summary.metadata["ess_mean"],
                "ess_fraction_min": summary.metadata.get("ess_fraction_min", float("nan")),
                "log_weight_span_max": summary.metadata.get("log_weight_span_max", float("nan")),
                "invalid_proposal_fraction_max": summary.metadata.get(
                    "invalid_proposal_fraction_max",
                    float("nan"),
                ),
                "guide_batch_backend": summary.metadata["guide_batch_backend"],
            }
            for seed, summary in zip(seeds, seed_summaries, strict=True)
        ],
        "diagnostics": diagnostics,
    }


def run_stationarity_seeds(
    case: RNCase,
    controls: RNRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    *,
    worker_count: int,
    progress: ProgressBar | None,
) -> tuple[list[RNBlockStreamingSummary], int]:
    return run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            run_seed_worker,
            case,
            controls,
            seed,
            density_grid,
            progress_queue,
        ),
        run_serial_seed=lambda seed: run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=progress,
        ),
    )


def run_seed_worker(
    case: RNCase,
    controls: RNRunControls,
    seed: int,
    density_grid: np.ndarray,
    progress_queue: Any | None = None,
) -> tuple[int, RNBlockStreamingSummary]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=worker_progress,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def diagnose_stationarity(seed_summaries: list[RNBlockStreamingSummary]) -> dict[str, dict]:
    times_by_seed = [require_trace(summary.trace_times) for summary in seed_summaries]
    energy_by_seed = [require_trace(summary.mixed_energy_trace) for summary in seed_summaries]
    r2_by_seed = [require_trace(summary.r2_radius_trace) for summary in seed_summaries]
    rms_by_seed = [np.sqrt(np.maximum(values, 0.0)) for values in r2_by_seed]
    return {
        "energy": diagnose_chains(times_by_seed, energy_by_seed).to_dict(),
        "rms": diagnose_chains(times_by_seed, rms_by_seed).to_dict(),
        "r2": diagnose_chains(times_by_seed, r2_by_seed).to_dict(),
    }


def require_trace(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        raise ValueError("RN streaming summary does not contain trace data")
    return np.asarray(values, dtype=float)


def classify_case(
    *,
    case_gate: bool,
    density_accounting_clean: bool,
    valid_finite_clean: bool,
    rn_weight_controlled: bool,
    diagnostics: dict[str, dict],
) -> str:
    if case_gate:
        if any(
            diagnostics[name]["classification"] == "WARNING_SPREAD_ONLY"
            for name in ("energy", "rms", "r2")
        ):
            return "RN_TRAPPED_STATIONARITY_WARNING"
        return "RN_TRAPPED_STATIONARITY_GO"
    if not density_accounting_clean:
        return "RN_TRAPPED_DENSITY_ACCOUNTING_NO_GO"
    if not valid_finite_clean:
        return "RN_TRAPPED_VALID_FINITE_NO_GO"
    if not rn_weight_controlled:
        return "RN_TRAPPED_WEIGHT_CONTROL_NO_GO"
    return "RN_TRAPPED_STATIONARITY_NO_GO"


def classify_grid(rows: list[dict[str, Any]]) -> str:
    if all(row["case_gate"] for row in rows):
        if any(row["classification"].endswith("_WARNING") for row in rows):
            return "RN_TRAPPED_STATIONARITY_GRID_WARNING"
        return "RN_TRAPPED_STATIONARITY_GRID_GO"
    return "RN_TRAPPED_STATIONARITY_GRID_NO_GO"


def blocked_spread_diagnostics(
    seeds: list[int],
    seed_summaries: list[RNBlockStreamingSummary],
) -> dict[str, dict[str, Any]]:
    metric_values = {
        "energy": [require_trace(summary.mixed_energy_trace) for summary in seed_summaries],
        "rms": [require_trace(summary.rms_radius_trace) for summary in seed_summaries],
        "r2": [require_trace(summary.r2_radius_trace) for summary in seed_summaries],
    }
    return {
        metric: blocked_metric_diagnostics(seeds, traces)
        for metric, traces in metric_values.items()
    }


def blocked_metric_diagnostics(
    seeds: list[int],
    traces: list[np.ndarray],
    *,
    blocked_z_threshold: float = 3.0,
) -> dict[str, Any]:
    means = np.asarray([float(np.mean(trace[np.isfinite(trace)])) for trace in traces], dtype=float)
    rows: list[dict[str, Any]] = []
    plateau_flags: list[bool] = []
    blocked_zscores: list[float] = []
    robust_zscores: list[float] = []
    for index, (seed, trace) in enumerate(zip(seeds, traces, strict=True)):
        trace = np.asarray(trace, dtype=float)
        trace = trace[np.isfinite(trace)]
        curve = blocking_curve(trace, min_blocks=16)
        plateau = detect_blocking_plateau(
            curve.block_sizes,
            curve.n_blocks,
            curve.stderr,
            min_blocks=16,
            window=3,
            rel_tol=0.10,
        )
        plateau_flags.append(plateau.plateau_found)
        others = np.delete(means, index)
        other_mean = float(np.mean(others)) if others.size else float("nan")
        other_se = stderr(others)
        blocked_z = safe_abs_z(
            float(means[index] - other_mean),
            float(np.sqrt(plateau.plateau_stderr**2 + other_se**2)),
        )
        robust_z = robust_zscore(float(means[index]), means)
        blocked_zscores.append(blocked_z)
        robust_zscores.append(robust_z)
        rows.append(
            {
                "seed": seed,
                "mean": float(means[index]),
                "blocking_plateau_found": plateau.plateau_found,
                "blocking_plateau_stderr": plateau.plateau_stderr,
                "blocking_plateau_block_size": plateau.plateau_block_size,
                "blocking_plateau_n_blocks": plateau.plateau_n_blocks,
                "blocking_plateau_reason": plateau.reason,
                "blocked_zscore": blocked_z,
                "robust_zscore": robust_z,
            }
        )
    finite_blocked_z = [value for value in blocked_zscores if np.isfinite(value)]
    finite_robust_z = [value for value in robust_zscores if np.isfinite(value)]
    plateau_all = all(plateau_flags)
    blocked_z_max = max(finite_blocked_z) if finite_blocked_z else float("inf")
    return {
        "plateau_all": plateau_all,
        "plateau_count": int(sum(plateau_flags)),
        "seed_count": len(seeds),
        "blocked_zscore_max": float(blocked_z_max),
        "robust_zscore_max": float(max(finite_robust_z) if finite_robust_z else float("nan")),
        "blocked_zscore_threshold": blocked_z_threshold,
        "blocked_zscore_gate": bool(plateau_all and blocked_z_max <= blocked_z_threshold),
        "per_seed": rows,
    }


def classify_blocked_case(
    *,
    hygiene_gate: bool,
    rn_weight_status: str,
    energy: dict[str, Any],
    rms: dict[str, Any],
    r2: dict[str, Any],
    diagnostics: dict[str, dict],
) -> str:
    if rn_weight_status == "RN_WEIGHT_NO_GO":
        return "RN_WEIGHT_NO_GO"
    if not hygiene_gate:
        return "HYGIENE_NO_GO"
    if not energy["plateau_all"]:
        return "NO_GO_NO_BLOCKING_PLATEAU"
    if diagnostics["energy"]["classification"] == "NO_GO_STATIONARITY":
        return "NO_GO_STATIONARITY"
    if not energy["blocked_zscore_gate"]:
        return "SPREAD_VETO_NO_GO"
    mixed_warning = (
        not rms["plateau_all"]
        or not r2["plateau_all"]
        or not rms["blocked_zscore_gate"]
        or not r2["blocked_zscore_gate"]
    )
    if mixed_warning:
        return "MIXED_OBSERVABLE_WARNING"
    if any(
        diagnostics[name]["classification"] == "WARNING_SPREAD_ONLY"
        for name in ("energy", "rms", "r2")
    ):
        return "WEAK_TRAP_WARNING"
    return "PASS_CANDIDATE"


def classify_rn_weight_status(
    *,
    ess_fraction_min: float,
    log_weight_span_max: float,
    ess_warning_fraction: float,
    ess_no_go_fraction: float,
    log_weight_span_warning: float,
) -> str:
    if not np.isfinite(ess_fraction_min) or ess_fraction_min < ess_no_go_fraction:
        return "RN_WEIGHT_NO_GO"
    if ess_fraction_min < ess_warning_fraction:
        return "RN_WEIGHT_WARNING"
    if np.isfinite(log_weight_span_max) and log_weight_span_max > log_weight_span_warning:
        return "RN_WEIGHT_WARNING"
    return "RN_WEIGHT_GO"


def write_seed_trace_artifacts(
    output_dir: Any,
    case_id: str,
    seeds: list[int],
    seed_summaries: list[RNBlockStreamingSummary],
) -> list[str]:
    root = ensure_dir(output_dir) / "seed_traces" / case_id
    ensure_dir(root)
    paths: list[str] = []
    for seed, summary in zip(seeds, seed_summaries, strict=True):
        trace = seed_trace_dict(summary)
        csv_path = root / f"{case_id}_seed{seed}_trace.csv"
        write_trace_csv(csv_path, trace)
        npz_path = root / f"{case_id}_seed{seed}_trace.npz"
        save_npz = cast(Any, np.savez_compressed)
        save_npz(str(npz_path), **trace)
        paths.extend([str(csv_path), str(npz_path)])
        for metric in ("mixed_energy", "rms_radius", "r2_radius"):
            paths.append(str(write_blocking_curve_csv(root, case_id, seed, metric, trace[metric])))
    return paths


def seed_trace_dict(summary: RNBlockStreamingSummary) -> dict[str, np.ndarray]:
    trace_times = require_trace(summary.trace_times)
    return {
        "tau": trace_times,
        "mixed_energy": require_trace(summary.mixed_energy_trace),
        "rms_radius": require_trace(summary.rms_radius_trace),
        "r2_radius": require_trace(summary.r2_radius_trace),
        "local_energy_variance": require_trace(summary.local_energy_variance_trace),
        "log_weight_span": require_trace(summary.log_weight_span_trace),
        "ess_fraction": require_trace(summary.ess_fraction_trace),
        "invalid_proposal_fraction": require_trace(summary.invalid_proposal_fraction_trace),
        "hard_wall_kill_fraction": require_trace(summary.hard_wall_kill_fraction_trace),
        "zero_weight_excluded_fraction": require_trace(summary.zero_weight_excluded_fraction_trace),
        "rn_logk_mean": require_trace(summary.rn_logk_mean_trace),
        "rn_logq_mean": require_trace(summary.rn_logq_mean_trace),
        "rn_logw_increment_mean": require_trace(summary.rn_logw_increment_mean_trace),
        "rn_logw_increment_variance": require_trace(summary.rn_logw_increment_variance_trace),
        "retained_fraction": require_trace(summary.retained_fraction_trace),
    }


def write_trace_csv(path: Any, trace: dict[str, np.ndarray]) -> None:
    import csv

    keys = list(trace.keys())
    row_count = len(trace["tau"])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({key: float(trace[key][index]) for key in keys})


def write_blocking_curve_csv(
    output_dir: Any,
    case_id: str,
    seed: int,
    metric: str,
    values: np.ndarray,
) -> Any:
    import csv

    curve = blocking_curve(values, min_blocks=16)
    plateau = detect_blocking_plateau(
        curve.block_sizes,
        curve.n_blocks,
        curve.stderr,
        min_blocks=16,
    )
    path = output_dir / f"blocking_{case_id}_seed{seed}_{metric}.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "block_size",
                "n_blocks",
                "stderr",
                "plateau_found",
                "plateau_stderr",
                "plateau_block_size",
                "plateau_n_blocks",
                "plateau_reason",
            ],
        )
        writer.writeheader()
        for block_size, n_blocks, stderr_value in zip(
            curve.block_sizes,
            curve.n_blocks,
            curve.stderr,
            strict=True,
        ):
            writer.writerow(
                {
                    "metric": metric,
                    "block_size": int(block_size),
                    "n_blocks": int(n_blocks),
                    "stderr": float(stderr_value),
                    "plateau_found": plateau.plateau_found,
                    "plateau_stderr": plateau.plateau_stderr,
                    "plateau_block_size": plateau.plateau_block_size,
                    "plateau_n_blocks": plateau.plateau_n_blocks,
                    "plateau_reason": plateau.reason,
                }
            )
    return path


def mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def stderr(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def uncertainty_summary(values: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any]:
    seed_stderr = stderr(np.asarray(values, dtype=float))
    blocking_stderr = combined_blocking_stderr(diagnostics)
    conservative = finite_max([seed_stderr, blocking_stderr])
    spread_warning = int(diagnostics["spread_warning_count"]) > 0
    status = (
        "INFLATED"
        if spread_warning or blocking_stderr_exceeds_seed(blocking_stderr, seed_stderr)
        else "STANDARD"
    )
    if not np.isfinite(conservative):
        status = "UNAVAILABLE"
    return {
        "seed_stderr": float(seed_stderr),
        "blocking_stderr": float(blocking_stderr),
        "conservative_stderr": float(conservative),
        "status": status,
    }


def combined_blocking_stderr(diagnostics: dict[str, Any]) -> float:
    chain_rows = diagnostics["chain_diagnostics"]
    stderrs = [
        float(row["blocking_stderr"])
        for row in chain_rows
        if np.isfinite(float(row["blocking_stderr"]))
    ]
    if not stderrs:
        return float("nan")
    return float(np.sqrt(np.sum(np.square(stderrs))) / len(stderrs))


def max_spread_blocking_z(diagnostics: dict[str, dict]) -> float:
    values: list[float] = []
    for observable in ("energy", "rms", "r2"):
        values.extend(
            float(row["spread_blocking_z"])
            for row in diagnostics[observable]["chain_diagnostics"]
            if np.isfinite(float(row["spread_blocking_z"]))
        )
    return max(values) if values else float("nan")


def case_uncertainty_status(summaries: list[dict[str, Any]]) -> str:
    statuses = {str(summary["status"]) for summary in summaries}
    if "UNAVAILABLE" in statuses:
        return "UNAVAILABLE"
    if "INFLATED" in statuses:
        return "INFLATED"
    return "STANDARD"


def blocking_stderr_exceeds_seed(blocking_stderr: float, seed_stderr: float) -> bool:
    if not np.isfinite(blocking_stderr):
        return False
    if not np.isfinite(seed_stderr):
        return True
    return blocking_stderr > seed_stderr


def finite_max(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else float("nan")


def safe_abs_z(delta: float, stderr_value: float) -> float:
    if stderr_value > 0.0 and np.isfinite(stderr_value):
        return float(abs(delta) / stderr_value)
    return 0.0 if delta == 0.0 else float("inf")


def robust_zscore(value: float, values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 0.0 or not np.isfinite(mad):
        return 0.0 if value == median else float("inf")
    return float(0.6745 * (value - median) / mad)


def min_seed_metadata(seed_summaries: list[RNBlockStreamingSummary], key: str) -> float:
    values = [
        float(summary.metadata.get(key, float("nan")))
        for summary in seed_summaries
        if np.isfinite(float(summary.metadata.get(key, float("nan"))))
    ]
    return min(values) if values else float("nan")


def max_seed_metadata(seed_summaries: list[RNBlockStreamingSummary], key: str) -> float:
    values = [
        float(summary.metadata.get(key, float("nan")))
        for summary in seed_summaries
        if np.isfinite(float(summary.metadata.get(key, float("nan"))))
    ]
    return max(values) if values else float("nan")
