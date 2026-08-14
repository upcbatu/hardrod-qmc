from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.artifacts.progress import ProgressBar, QueuedProgress
from hrdmc.sampling.dmc.results import DMCStreamingSummary
from hrdmc.sampling.dmc.run import resolve_parallel_workers, run_streaming_seed
from hrdmc.sampling.initial_conditions import InitializationControls
from hrdmc.sampling.seed_batch import run_seed_batch
from hrdmc.statistics.density import relative_density_l2_error
from hrdmc.statistics.timeseries import (
    CHAIN_ACCEPTED,
    CHAIN_SPREAD_WARNING,
    diagnose_chains,
)
from hrdmc.system.settings import DMCRunControls, TrappedCase, build_case_geometry, make_grid
from hrdmc.theory.lda import lda_density_profile, lda_rms_radius, lda_total_energy
from hrdmc.trial.guide import DEFAULT_GUIDE_FAMILY

_ACCEPTED = {CHAIN_ACCEPTED, CHAIN_SPREAD_WARNING}


def summarize_stationarity_case(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_invalid_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: InitializationControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Run independent seeds and reduce them to the stationarity evidence row."""
    del trace_output_dir
    initialization = initialization or InitializationControls()
    grid = make_grid(controls, case)
    requested = resolve_parallel_workers(len(seeds), parallel_workers)
    summaries, workers = _run_seeds(
        case,
        controls,
        seeds,
        grid,
        requested,
        progress,
        initialization,
        guide_family,
    )
    return summarize_stationarity_from_seed_summaries(
        case,
        controls,
        seeds,
        grid,
        summaries,
        workers,
        requested_worker_count=requested,
        ess_warning_fraction=ess_warning_fraction,
        ess_invalid_fraction=ess_invalid_fraction,
        log_weight_span_warning=log_weight_span_warning,
        initialization=initialization,
        guide_family=guide_family,
    )


def summarize_stationarity_from_seed_summaries(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    grid: np.ndarray,
    seed_summaries: list[DMCStreamingSummary],
    actual_worker_count: int,
    *,
    requested_worker_count: int | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_invalid_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: InitializationControls,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Compute the thesis-facing diagnostics from already-produced seed summaries."""
    del trace_output_dir
    if len(seeds) != len(seed_summaries) or not seeds:
        raise ValueError("stationarity requires one summary per independent seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("stationarity seeds must be unique")
    system, trap = build_case_geometry(case)
    density = np.mean([summary.density for summary in seed_summaries], axis=0)
    energy = np.asarray([summary.mixed_energy for summary in seed_summaries])
    r2 = np.asarray([summary.r2_radius for summary in seed_summaries])
    rms = np.sqrt(r2)
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    diagnostics = _diagnose(seed_summaries)
    audit = {name: _audit(seeds, value) for name, value in diagnostics.items()}
    energy_uncertainty = _uncertainty(energy, diagnostics["energy"])
    r2_uncertainty = _uncertainty(r2, diagnostics["r2"])
    rms_uncertainty = _uncertainty(rms, diagnostics["rms"])
    density_integral = float(np.sum(density * np.diff(seed_summaries[0].density_bin_edges)))
    density_clean = abs(density_integral - case.n_particles) <= 5e-3
    finite_clean = all(
        summary.metadata.get("finite_local_energy_fraction") == 1.0
        and summary.metadata.get("valid_snapshot_fraction") == 1.0
        for summary in seed_summaries
    )
    ess_min = min(_metadata(summary, "ess_fraction_min") for summary in seed_summaries)
    weight_span = max(_metadata(summary, "log_weight_span_max") for summary in seed_summaries)
    weight_status = _weight_status(
        ess_min,
        weight_span,
        ess_warning_fraction,
        ess_invalid_fraction,
        log_weight_span_warning,
    )
    base_valid = density_clean and finite_clean and weight_status != "weight_collapse"
    energy_status = str(diagnostics["energy"]["classification"])
    validation = base_valid and energy_status in _ACCEPTED
    classification = (
        "accepted"
        if validation and energy_status == CHAIN_ACCEPTED
        else "spread_warning"
        if validation
        else "weight_collapse"
        if weight_status == "weight_collapse"
        else "base_numerics_invalid"
        if not base_valid
        else "trace_nonstationary"
    )
    lda_energy = lda_total_energy(lda, rod_length=system.rod_length)
    lda_rms = lda_rms_radius(lda, center=trap.center)
    metadata = seed_summaries[0].metadata
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        **case.unit_metadata(),
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "guide_family": guide_family,
        "resolved_guide_family": metadata.get("resolved_guide_family", guide_family),
        "validation_passed": validation,
        "base_numerics_valid": base_valid,
        "classification": classification,
        "final_classification": classification,
        "method_status": "accepted" if base_valid else "base_numerics_invalid",
        "precision_status": energy_uncertainty["status"],
        "energy_estimator": "mixed_local_energy",
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_worker_count,
        "parallel_workers_requested": requested_worker_count or actual_worker_count,
        "effective_grid_extent": float(max(abs(grid[0]), abs(grid[-1]))),
        "density_integral": density_integral,
        "density_accounting_clean": density_clean,
        "valid_finite_clean": finite_clean,
        "population_weights_controlled": weight_status != "weight_collapse",
        "population_weight_status": weight_status,
        "ess_fraction_min": ess_min,
        "log_weight_span_max": weight_span,
        "mixed_energy": float(np.mean(energy)),
        "r2_radius": float(np.mean(r2)),
        "rms_radius": float(np.mean(rms)),
        "lda_total_energy": lda_energy,
        "energy_dmc_minus_lda": float(np.mean(energy) - lda_energy),
        "lda_rms_radius": lda_rms,
        "rms_dmc_minus_lda": float(np.mean(rms) - lda_rms),
        "density_relative_l2": relative_density_l2_error(grid, density, lda.n_x),
        "density_profile": {
            "x": grid.tolist(),
            "mixed_n_x": density.tolist(),
            "lda_n_x": lda.n_x.tolist(),
            "estimator": "mixed coordinate diagnostic",
        },
        "lost_out_of_grid_sample_count_total": int(
            sum(summary.lost_out_of_grid_sample_count for summary in seed_summaries)
        ),
        "diagnostics": diagnostics,
        "stationarity_failure_audit": audit,
        "seed_summaries": [
            _seed_row(seed, summary) for seed, summary in zip(seeds, seed_summaries, strict=True)
        ],
        "trace_artifacts": [],
    }
    uncertainty_by_name = {
        "energy": energy_uncertainty,
        "r2": r2_uncertainty,
        "rms": rms_uncertainty,
    }
    _add_diagnostic_fields(
        result,
        values_by_name={"energy": energy, "r2": r2, "rms": rms},
        uncertainty_by_name=uncertainty_by_name,
        diagnostics=diagnostics,
        audit=audit,
    )
    _add_energy_diagnostic_summary(result, energy_uncertainty, diagnostics, audit)
    return result


def _add_diagnostic_fields(
    result: dict[str, Any],
    *,
    values_by_name: dict[str, np.ndarray],
    uncertainty_by_name: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    for name, values in values_by_name.items():
        uncertainty = uncertainty_by_name[name]
        prefix = "mixed_energy" if name == "energy" else f"{name}_radius"
        result[f"{prefix}_seed_stderr"] = _stderr(values)
        result[f"{prefix}_blocking_stderr"] = uncertainty["blocking_stderr"]
        result[f"{prefix}_correlated_stderr"] = uncertainty["blocking_stderr"]
        result[f"{prefix}_conservative_stderr"] = uncertainty["conservative_stderr"]
        result[f"{prefix}_uncertainty_status"] = uncertainty["status"]
        result[f"{prefix}_error_estimator_status"] = uncertainty["status"]
        result[f"rhat_{name}"] = diagnostics[name]["rhat"]
        result[f"neff_{name}"] = diagnostics[name]["min_effective_independent_samples"]
        result[f"stationarity_{name}"] = diagnostics[name]["classification"]
        result[f"stationarity_reason_{name}"] = audit[name]["reason"]
        result[f"stationarity_failing_seeds_{name}"] = ",".join(
            str(seed) for seed in audit[name]["failing_seeds"]
        )
        result[f"stationarity_slope_z_max_{name}"] = audit[name]["slope_z_max"]
        result[f"stationarity_quarter_z_max_{name}"] = audit[name]["first_last_quarter_z_max"]
        result[f"stationarity_late_z_max_{name}"] = audit[name]["late_cumulative_z_max"]
        result[f"stationarity_block_z_max_{name}"] = audit[name]["first_last_blocking_z_max"]


def _add_energy_diagnostic_summary(
    result: dict[str, Any],
    energy_uncertainty: dict[str, Any],
    diagnostics: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    result["uncertainty_status"] = energy_uncertainty["status"]
    result["max_spread_blocking_z"] = max(
        row["spread_blocking_z"]
        for diagnostic in diagnostics.values()
        for row in diagnostic["chain_diagnostics"]
    )
    result["blocking_plateau_energy"] = np.isfinite(energy_uncertainty["blocking_stderr"])
    result["blocked_zscore_max_energy"] = audit["energy"]["first_last_blocking_z_max"]
    result["robust_zscore_max_energy"] = max(
        audit["energy"][key]
        for key in (
            "slope_z_max",
            "first_last_quarter_z_max",
            "late_cumulative_z_max",
            "first_last_blocking_z_max",
        )
    )


def classify_grid(rows: list[dict[str, Any]]) -> str:
    if all(row.get("validation_passed") is True for row in rows):
        return (
            "accepted"
            if all(row.get("classification") == "accepted" for row in rows)
            else "accepted_with_warnings"
        )
    return "grid_contains_unresolved_case"


def _run_seeds(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    grid: np.ndarray,
    workers: int,
    progress: ProgressBar | None,
    initialization: InitializationControls,
    guide_family: str,
) -> tuple[list[DMCStreamingSummary], int]:
    return run_seed_batch(
        seeds,
        worker_count=workers,
        progress=progress,
        submit_seed=lambda executor, seed, queue: executor.submit(
            _seed_worker,
            case,
            controls,
            seed,
            grid,
            queue,
            initialization,
            guide_family,
        ),
        run_serial_seed=lambda seed: run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=grid,
            progress=progress,
            initialization=initialization,
            guide_family=guide_family,
        ),
    )


def _seed_worker(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    grid: np.ndarray,
    queue: Any | None,
    initialization: InitializationControls,
    guide_family: str,
) -> tuple[int, DMCStreamingSummary]:
    progress = QueuedProgress(queue) if queue is not None else None
    try:
        return seed, run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=grid,
            progress=progress,
            initialization=initialization,
            guide_family=guide_family,
        )
    finally:
        if progress is not None:
            progress.flush()


def _diagnose(summaries: list[DMCStreamingSummary]) -> dict[str, dict[str, Any]]:
    times = [_trace(summary.trace_times, "times") for summary in summaries]
    energy = [_trace(summary.mixed_energy_trace, "energy") for summary in summaries]
    r2 = [_trace(summary.r2_radius_trace, "R2") for summary in summaries]
    return {
        "energy": diagnose_chains(times, energy).to_dict(),
        "r2": diagnose_chains(times, r2).to_dict(),
        "rms": diagnose_chains(
            times, [np.sqrt(np.maximum(values, 0.0)) for values in r2]
        ).to_dict(),
    }


def _audit(seeds: list[int], diagnostics: dict[str, Any]) -> dict[str, Any]:
    rows = diagnostics["chain_diagnostics"]
    per_seed = []
    failing = []
    for seed, row in zip(seeds, rows, strict=True):
        failures = [
            name
            for name, key in (
                ("trend_detected", "trend_clean"),
                ("cumulative_drift", "cumulative_drift_clean"),
                ("block_drift", "blocking_clean"),
            )
            if not bool(row.get(key))
        ]
        if row.get("spread_warning"):
            failures.append("spread_warning")
        if failures:
            failing.append(seed)
        per_seed.append({"seed": seed, "failures": failures, **row})

    def finite_max(key: str) -> float:
        return max(
            (abs(float(row[key])) for row in rows if np.isfinite(float(row[key]))),
            default=float("nan"),
        )

    reasons = sorted({reason for row in per_seed for reason in row["failures"]})
    return {
        "reason": "+".join(reasons) if reasons else "accepted",
        "failing_seeds": failing,
        "failing_seed_count": len(failing),
        "slope_z_max": finite_max("slope_z_autocorr_adjusted"),
        "first_last_quarter_z_max": finite_max("first_last_quarter_z"),
        "late_cumulative_z_max": finite_max("late_cumulative_z"),
        "first_last_blocking_z_max": finite_max("first_last_blocking_z"),
        "per_seed": per_seed,
    }


def _uncertainty(values: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any]:
    seed_error = _stderr(values)
    chain_errors = [
        float(row["blocking_stderr"])
        for row in diagnostics["chain_diagnostics"]
        if np.isfinite(float(row["blocking_stderr"]))
    ]
    blocking = (
        float(np.sqrt(np.sum(np.square(chain_errors))) / len(chain_errors))
        if chain_errors
        else float("nan")
    )
    conservative = max(value for value in (seed_error, blocking) if np.isfinite(value))
    return {
        "blocking_stderr": blocking,
        "conservative_stderr": conservative,
        "status": "accepted" if chain_errors else "correlated_error_unavailable",
    }


def _seed_row(seed: int, summary: DMCStreamingSummary) -> dict[str, Any]:
    fields = [
        "finite_local_energy_fraction",
        "valid_snapshot_fraction",
        "killed_count",
        "resample_count",
        "ess_min",
        "ess_mean",
        "ess_fraction_min",
        "log_weight_span_max",
        "invalid_proposal_fraction_max",
        "local_acceptance_fraction_mean",
        "metropolis_rejection_fraction_max",
        "configuration_esjd_mean",
        "r2_esjd_mean",
        "weighted_free_gap_esjd_mean",
        "drift_limiter",
        "relative_alpha",
        "guide_family",
        "resolved_guide_family",
    ]
    return {
        "seed": seed,
        "mixed_energy": summary.mixed_energy,
        "r2_radius": summary.r2_radius,
        "rms_radius": summary.rms_radius,
        "density_integral": summary.density_integral,
        **{field: summary.metadata.get(field) for field in fields},
    }


def _weight_status(
    ess_min: float,
    span: float,
    warning_fraction: float,
    invalid_fraction: float,
    warning_span: float,
) -> str:
    if ess_min < invalid_fraction:
        return "weight_collapse"
    if ess_min < warning_fraction or span > warning_span:
        return "weight_warning"
    return "accepted"


def _trace(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"DMC summary is missing the {name} trace")
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} trace must be finite and one-dimensional")
    return array


def _metadata(summary: DMCStreamingSummary, key: str) -> float:
    value = float(summary.metadata[key])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite DMC metadata: {key}")
    return value


def _stderr(values: np.ndarray) -> float:
    return 0.0 if values.size == 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))
