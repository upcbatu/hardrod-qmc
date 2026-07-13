from __future__ import annotations

from typing import Any, cast

import numpy as np

from hrdmc.analysis import (
    CHAIN_ACCEPTED,
    CHAIN_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM,
    CHAIN_RHAT_ABOVE_LIMIT,
    CHAIN_SPREAD_WARNING,
    CHAIN_TRACE_NONSTATIONARY,
    CORRELATED_ERROR_AGREEMENT,
    CORRELATED_ERROR_DISAGREEMENT,
    CORRELATED_ERROR_UNAVAILABLE,
    blocking_curve,
    detect_blocking_plateau,
    diagnose_chains,
    relative_density_l2_error,
    triangulated_error_estimate,
)
from hrdmc.artifacts import ensure_dir
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.monte_carlo.dmc.local import DMCStreamingSummary
from hrdmc.runners import run_seed_batch
from hrdmc.theory import lda_density_profile, lda_rms_radius, lda_total_energy
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    build_case_geometry,
    make_grid,
    resolve_parallel_workers,
    run_streaming_seed,
)

ACCEPTED_CHAIN_CLASSIFICATIONS = {CHAIN_ACCEPTED, CHAIN_SPREAD_WARNING}
TRIANGULATED_PRECISION_WARNING = "blocking_plateau_unresolved_correlated_error_available"
MIXED_OBSERVABLE_WARNING = "mixed_coordinate_precision_warning"
MIXED_COORDINATE_DIAGNOSTIC = "energy_accepted_mixed_coordinate_diagnostic_unresolved"
BLOCKING_CURVE_MIN_BLOCKS = 16
BLOCKING_PLATEAU_MIN_BLOCKS = 32
BLOCKING_PLATEAU_WINDOW = 3
BLOCKING_PLATEAU_REL_TOL = 0.10
BLOCKING_PLATEAU_SIGMA_TOL = 1.0


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
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    initialization = InitializationControls() if initialization is None else initialization
    grid = make_grid(controls, case)
    worker_count = resolve_parallel_workers(len(seeds), parallel_workers)
    seed_summaries, actual_worker_count = run_stationarity_seeds(
        case,
        controls,
        seeds,
        grid,
        worker_count=worker_count,
        progress=progress,
        initialization=initialization,
        collective_rn=collective_rn,
        guide_family=guide_family,
    )
    return summarize_stationarity_from_seed_summaries(
        case,
        controls,
        seeds,
        grid,
        seed_summaries,
        actual_worker_count,
        requested_worker_count=worker_count,
        trace_output_dir=trace_output_dir,
        ess_warning_fraction=ess_warning_fraction,
        ess_invalid_fraction=ess_invalid_fraction,
        log_weight_span_warning=log_weight_span_warning,
        initialization=initialization,
        collective_rn=collective_rn,
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
    collective_rn: CollectiveRNControls | None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Aggregate stationarity diagnostics from already-run seed summaries."""

    if requested_worker_count is None:
        requested_worker_count = actual_worker_count

    system, trap = build_case_geometry(case)
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
    correlated_errors = correlated_error_diagnostics(seeds, seed_summaries)
    stationarity_audit = stationarity_failure_audit(seeds, diagnostics)
    trace_artifacts = (
        write_seed_trace_artifacts(trace_output_dir, case.case_id, seeds, seed_summaries)
        if trace_output_dir is not None
        else []
    )
    density_l2_values = np.asarray(
        [relative_density_l2_error(grid, summary.density, lda.n_x) for summary in seed_summaries],
        dtype=float,
    )
    energy_uncertainty = uncertainty_summary(
        energy_values,
        diagnostics["energy"],
        correlated_errors["energy"],
        plateau_all=spread["energy"]["plateau_all"],
    )
    rms_uncertainty = uncertainty_summary(
        rms_values,
        diagnostics["rms"],
        correlated_errors["rms"],
        plateau_all=spread["rms"]["plateau_all"],
    )
    r2_uncertainty = uncertainty_summary(
        r2_values,
        diagnostics["r2"],
        correlated_errors["r2"],
        plateau_all=spread["r2"]["plateau_all"],
    )
    density_integral = float(np.sum(density) * (grid[1] - grid[0]))
    density_accounting_clean = abs(density_integral - system.n_particles) <= 5e-3
    valid_finite_clean = all(
        summary.metadata["finite_local_energy_fraction"] == 1.0
        and summary.metadata["valid_snapshot_fraction"] == 1.0
        for summary in seed_summaries
    )
    ess_fraction_min = min_seed_metadata(seed_summaries, "ess_fraction_min")
    log_weight_span_max = max_seed_metadata(seed_summaries, "log_weight_span_max")
    population_weight_status = classify_population_weight_status(
        ess_fraction_min=ess_fraction_min,
        log_weight_span_max=log_weight_span_max,
        ess_warning_fraction=ess_warning_fraction,
        ess_invalid_fraction=ess_invalid_fraction,
        log_weight_span_warning=log_weight_span_warning,
    )
    population_weights_controlled = population_weight_status != "weight_collapse"
    base_numerics_valid = (
        density_accounting_clean and valid_finite_clean and population_weights_controlled
    )
    final_classification = classify_blocked_case(
        base_numerics_valid=base_numerics_valid,
        population_weight_status=population_weight_status,
        energy=spread["energy"],
        rms=spread["rms"],
        r2=spread["r2"],
        diagnostics=diagnostics,
        stationarity_audit=stationarity_audit,
        correlated_errors=correlated_errors,
    )
    validation_passed = final_classification in {
        "accepted",
        "spread_warning",
        MIXED_OBSERVABLE_WARNING,
        MIXED_COORDINATE_DIAGNOSTIC,
        TRIANGULATED_PRECISION_WARNING,
    }
    classification = classify_case(
        validation_passed=validation_passed,
        density_accounting_clean=density_accounting_clean,
        valid_finite_clean=valid_finite_clean,
        population_weights_controlled=population_weights_controlled,
        diagnostics=diagnostics,
        final_classification=final_classification,
    )
    lda_energy = lda_total_energy(lda, rod_length=system.rod_length)
    lda_rms = lda_rms_radius(lda, center=trap.center)
    return {
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        **case.unit_metadata(),
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "collective_rn_controls": (None if collective_rn is None else collective_rn.to_metadata()),
        "guide_family": guide_family,
        "collective_rn_enabled": collective_rn is not None,
        "resolved_guide_family": ",".join(
            sorted(
                {str(summary.metadata.get("resolved_guide_family")) for summary in seed_summaries}
            )
        ),
        "target_initial_rms": seed_summaries[0].metadata.get("target_initial_rms", float("nan")),
        "initializer_scope": seed_summaries[0].metadata.get("initializer_scope", ""),
        "validation_passed": validation_passed,
        "base_numerics_valid": base_numerics_valid,
        "classification": classification,
        "final_classification": final_classification,
        "method_status": method_status(
            density_accounting_clean=density_accounting_clean,
            valid_finite_clean=valid_finite_clean,
            population_weight_status=population_weight_status,
            diagnostics=diagnostics,
            stationarity_audit=stationarity_audit,
        ),
        "precision_status": precision_status(final_classification),
        "diagnostic_status": final_classification,
        "energy_estimator": "mixed local-energy Hamiltonian estimator",
        "mixed_coordinate_observable_status": (
            "diagnostic only; pure R2/RMS require Hellmann-Feynman energy response "
            "or transported auxiliary forward walking"
        ),
        "mixed_coordinate_diagnostic_status": mixed_coordinate_diagnostic_status(
            rms=spread["rms"],
            r2=spread["r2"],
            diagnostics=diagnostics,
            stationarity_audit=stationarity_audit,
        ),
        "pure_r2_estimator_status": "not_evaluated_hf_or_fw_required",
        "pure_rms_estimator_status": "not_evaluated_hf_or_fw_required",
        "pure_density_estimator_status": "not_evaluated_transported_fw_required",
        "pure_pair_structure_estimator_status": "not_evaluated_transported_fw_required",
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_worker_count,
        "parallel_workers_requested": requested_worker_count,
        "effective_grid_extent": float(max(abs(grid[0]), abs(grid[-1]))),
        "mixed_energy": mean(energy_values),
        "mixed_energy_seed_stderr": stderr(energy_values),
        "mixed_energy_blocking_stderr": energy_uncertainty["blocking_stderr"],
        "mixed_energy_correlated_stderr": energy_uncertainty["correlated_stderr"],
        "mixed_energy_conservative_stderr": energy_uncertainty["conservative_stderr"],
        "mixed_energy_uncertainty_status": energy_uncertainty["status"],
        "mixed_energy_error_estimator_status": correlated_errors["energy"]["status"],
        "rms_radius": mean(rms_values),
        "rms_radius_seed_stderr": stderr(rms_values),
        "rms_radius_blocking_stderr": rms_uncertainty["blocking_stderr"],
        "rms_radius_correlated_stderr": rms_uncertainty["correlated_stderr"],
        "rms_radius_conservative_stderr": rms_uncertainty["conservative_stderr"],
        "rms_radius_uncertainty_status": rms_uncertainty["status"],
        "rms_radius_error_estimator_status": correlated_errors["rms"]["status"],
        "r2_radius": mean(r2_values),
        "r2_radius_seed_stderr": stderr(r2_values),
        "r2_radius_blocking_stderr": r2_uncertainty["blocking_stderr"],
        "r2_radius_correlated_stderr": r2_uncertainty["correlated_stderr"],
        "r2_radius_conservative_stderr": r2_uncertainty["conservative_stderr"],
        "r2_radius_uncertainty_status": r2_uncertainty["status"],
        "r2_radius_error_estimator_status": correlated_errors["r2"]["status"],
        "density_integral": density_integral,
        "density_accounting_clean": density_accounting_clean,
        "valid_finite_clean": valid_finite_clean,
        "population_weights_controlled": population_weights_controlled,
        "lost_out_of_grid_sample_count_total": int(
            sum(summary.lost_out_of_grid_sample_count for summary in seed_summaries)
        ),
        "lda_total_energy": lda_energy,
        "energy_dmc_minus_lda": mean(energy_values) - lda_energy,
        "lda_rms_radius": lda_rms,
        "rms_dmc_minus_lda": mean(rms_values) - lda_rms,
        "density_relative_l2": relative_density_l2_error(grid, density, lda.n_x),
        "density_relative_l2_seed_stderr": stderr(density_l2_values),
        "density_profile": {
            "x": grid.tolist(),
            "mixed_n_x": density.tolist(),
            "lda_n_x": lda.n_x.tolist(),
            "estimator": "mixed coordinate diagnostic; pure density requires transported FW",
        },
        "uncertainty_status": case_uncertainty_status([energy_uncertainty]),
        "mixed_coordinate_uncertainty_status": case_uncertainty_status(
            [rms_uncertainty, r2_uncertainty]
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
        "stationarity_reason_energy": stationarity_audit["energy"]["reason"],
        "stationarity_reason_rms": stationarity_audit["rms"]["reason"],
        "stationarity_reason_r2": stationarity_audit["r2"]["reason"],
        "stationarity_failing_seeds_energy": ",".join(
            str(seed) for seed in stationarity_audit["energy"]["failing_seeds"]
        ),
        "stationarity_failing_seeds_rms": ",".join(
            str(seed) for seed in stationarity_audit["rms"]["failing_seeds"]
        ),
        "stationarity_failing_seeds_r2": ",".join(
            str(seed) for seed in stationarity_audit["r2"]["failing_seeds"]
        ),
        "stationarity_slope_z_max_energy": stationarity_audit["energy"]["slope_z_max"],
        "stationarity_slope_z_max_rms": stationarity_audit["rms"]["slope_z_max"],
        "stationarity_slope_z_max_r2": stationarity_audit["r2"]["slope_z_max"],
        "stationarity_quarter_z_max_energy": stationarity_audit["energy"][
            "first_last_quarter_z_max"
        ],
        "stationarity_quarter_z_max_rms": stationarity_audit["rms"]["first_last_quarter_z_max"],
        "stationarity_quarter_z_max_r2": stationarity_audit["r2"]["first_last_quarter_z_max"],
        "stationarity_late_z_max_energy": stationarity_audit["energy"]["late_cumulative_z_max"],
        "stationarity_late_z_max_rms": stationarity_audit["rms"]["late_cumulative_z_max"],
        "stationarity_late_z_max_r2": stationarity_audit["r2"]["late_cumulative_z_max"],
        "stationarity_block_z_max_energy": stationarity_audit["energy"][
            "first_last_blocking_z_max"
        ],
        "stationarity_block_z_max_rms": stationarity_audit["rms"]["first_last_blocking_z_max"],
        "stationarity_block_z_max_r2": stationarity_audit["r2"]["first_last_blocking_z_max"],
        "correlated_error_energy": correlated_errors["energy"]["status"],
        "correlated_error_rms": correlated_errors["rms"]["status"],
        "correlated_error_r2": correlated_errors["r2"]["status"],
        "correlated_error_energy_triangulated_seed_count": correlated_errors["energy"][
            "triangulated_seed_count"
        ],
        "correlated_error_rms_triangulated_seed_count": correlated_errors["rms"][
            "triangulated_seed_count"
        ],
        "correlated_error_r2_triangulated_seed_count": correlated_errors["r2"][
            "triangulated_seed_count"
        ],
        "spread_warning_count": int(diagnostics["energy"]["spread_warning_count"]),
        "mixed_coordinate_spread_warning_count": int(
            diagnostics["rms"]["spread_warning_count"] + diagnostics["r2"]["spread_warning_count"]
        ),
        "ess_fraction_min": ess_fraction_min,
        "log_weight_span_max": log_weight_span_max,
        "population_weight_status": population_weight_status,
        "population_weight_ess_warning_fraction": ess_warning_fraction,
        "population_weight_ess_invalid_fraction": ess_invalid_fraction,
        "population_weight_log_weight_span_warning": log_weight_span_warning,
        "initial_to_production_rms_ratio": float(
            np.mean(
                [
                    summary.metadata.get("initial_to_production_rms_ratio", float("nan"))
                    for summary in seed_summaries
                ]
            )
        ),
        "guide_batch_backend": ",".join(
            sorted({str(summary.metadata["guide_batch_backend"]) for summary in seed_summaries})
        ),
        "target_backend": ",".join(
            sorted({str(summary.metadata.get("target_backend", "")) for summary in seed_summaries})
        ),
        "proposal_backend": ",".join(
            sorted(
                {str(summary.metadata.get("proposal_backend", "")) for summary in seed_summaries}
            )
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
                "ess_fraction_min": summary.metadata["ess_fraction_min"],
                "log_weight_span_max": summary.metadata.get("log_weight_span_max", float("nan")),
                "invalid_proposal_fraction_max": summary.metadata.get(
                    "invalid_proposal_fraction_max",
                    float("nan"),
                ),
                "local_acceptance_fraction_mean": summary.metadata.get(
                    "local_acceptance_fraction_mean",
                    float("nan"),
                ),
                "metropolis_rejection_fraction_max": summary.metadata.get(
                    "metropolis_rejection_fraction_max",
                    float("nan"),
                ),
                "drift_norm_max": summary.metadata.get("drift_norm_max", float("nan")),
                "configuration_esjd_mean": summary.metadata.get(
                    "configuration_esjd_mean",
                    float("nan"),
                ),
                "r2_esjd_mean": summary.metadata.get("r2_esjd_mean", float("nan")),
                "weighted_free_gap_esjd_mean": summary.metadata.get(
                    "weighted_free_gap_esjd_mean",
                    float("nan"),
                ),
                "free_gap_min": summary.metadata.get("free_gap_min", float("nan")),
                "free_gap_p01_min": summary.metadata.get(
                    "free_gap_p01_min",
                    float("nan"),
                ),
                "initialization_mode": summary.metadata["initialization_mode"],
                "target_initial_rms": summary.metadata["target_initial_rms"],
                "initial_spacing_mean": summary.metadata["initial_spacing_mean"],
                "initial_spacing_std": summary.metadata["initial_spacing_std"],
                "initial_rms_mean": summary.metadata["initial_rms_mean"],
                "initial_rms_std": summary.metadata["initial_rms_std"],
                "initial_gap_min": summary.metadata["initial_gap_min"],
                "breathing_preburn_steps": summary.metadata["breathing_preburn_steps"],
                "breathing_preburn_log_step": summary.metadata["breathing_preburn_log_step"],
                "breathing_preburn_acceptance_rate": summary.metadata[
                    "breathing_preburn_acceptance_rate"
                ],
                "breathing_preburn_jacobian_dimension": summary.metadata[
                    "breathing_preburn_jacobian_dimension"
                ],
                "preburn_rms_mean": summary.metadata["preburn_rms_mean"],
                "preburn_rms_std": summary.metadata["preburn_rms_std"],
                "preburn_gap_min": summary.metadata["preburn_gap_min"],
                "initial_to_production_rms_ratio": summary.metadata[
                    "initial_to_production_rms_ratio"
                ],
                **breathing_trace_summary(summary),
                "guide_batch_backend": summary.metadata["guide_batch_backend"],
                "target_backend": summary.metadata.get("target_backend", ""),
                "proposal_backend": summary.metadata.get("proposal_backend", ""),
                "local_step_method": summary.metadata.get("local_step_method", ""),
                "relative_alpha": summary.metadata.get("relative_alpha"),
                "contact_beta": summary.metadata.get("contact_beta"),
                "response_lambda": summary.metadata.get("response_lambda"),
                "collective_rn": summary.metadata.get("collective_rn"),
                "guide_family": summary.metadata.get("guide_family", ""),
                "resolved_guide_family": summary.metadata.get("resolved_guide_family", ""),
            }
            for seed, summary in zip(seeds, seed_summaries, strict=True)
        ],
        "diagnostics": diagnostics,
        "correlated_error_diagnostics": correlated_errors,
        "stationarity_failure_audit": stationarity_audit,
    }


def run_stationarity_seeds(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    *,
    worker_count: int,
    progress: ProgressBar | None,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> tuple[list[DMCStreamingSummary], int]:
    initialization = InitializationControls() if initialization is None else initialization
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
            initialization,
            collective_rn,
            guide_family,
        ),
        run_serial_seed=lambda seed: run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=progress,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=guide_family,
        ),
    )


def run_seed_worker(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    density_grid: np.ndarray,
    progress_queue: Any | None = None,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> tuple[int, DMCStreamingSummary]:
    initialization = InitializationControls() if initialization is None else initialization
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=worker_progress,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def diagnose_stationarity(seed_summaries: list[DMCStreamingSummary]) -> dict[str, dict]:
    times_by_seed = [require_trace(summary.trace_times) for summary in seed_summaries]
    energy_by_seed = [require_trace(summary.mixed_energy_trace) for summary in seed_summaries]
    r2_by_seed = [require_trace(summary.r2_radius_trace) for summary in seed_summaries]
    rms_by_seed = [np.sqrt(np.maximum(values, 0.0)) for values in r2_by_seed]
    return {
        "energy": diagnose_chains(times_by_seed, energy_by_seed).to_dict(),
        "rms": diagnose_chains(times_by_seed, rms_by_seed).to_dict(),
        "r2": diagnose_chains(times_by_seed, r2_by_seed).to_dict(),
    }


def correlated_error_diagnostics(
    seeds: list[int],
    seed_summaries: list[DMCStreamingSummary],
) -> dict[str, dict[str, Any]]:
    metric_traces = {
        "energy": [optional_trace(summary.mixed_energy_trace) for summary in seed_summaries],
        "rms": [optional_trace(summary.rms_radius_trace) for summary in seed_summaries],
        "r2": [optional_trace(summary.r2_radius_trace) for summary in seed_summaries],
    }
    spacings = [
        trace_spacing_tau(optional_trace(summary.trace_times)) for summary in seed_summaries
    ]
    return {
        metric: summarize_correlated_error_metric(seeds, traces, spacings)
        for metric, traces in metric_traces.items()
    }


def stationarity_failure_audit(
    seeds: list[int],
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        metric: stationarity_metric_audit(seeds, diagnostics[metric])
        for metric in ("energy", "rms", "r2")
    }


def stationarity_metric_audit(
    seeds: list[int],
    metric_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    rows = metric_diagnostics["chain_diagnostics"]
    per_seed = []
    failing_seeds: list[int] = []
    reasons: list[str] = []
    for seed, row in zip(seeds, rows, strict=True):
        failures = stationarity_row_failures(row)
        if failures:
            failing_seeds.append(int(seed))
            reasons.extend(failures)
        per_seed.append({"seed": int(seed), "failures": failures, **stationarity_row_values(row)})
    unique_reasons = sorted(set(reasons))
    return {
        "reason": "+".join(unique_reasons) if unique_reasons else "accepted",
        "failing_seeds": failing_seeds,
        "failing_seed_count": len(failing_seeds),
        "slope_z_max": max_finite(row["slope_z_autocorr_adjusted"] for row in rows),
        "first_last_quarter_z_max": max_finite(row["first_last_quarter_z"] for row in rows),
        "late_cumulative_z_max": max_finite(row["late_cumulative_z"] for row in rows),
        "first_last_blocking_z_max": max_finite(row["first_last_blocking_z"] for row in rows),
        "spread_blocking_z_max": max_finite(row["spread_blocking_z"] for row in rows),
        "per_seed": per_seed,
    }


def stationarity_row_failures(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not bool(row.get("trend_clean", False)):
        failures.append("trend_detected")
    if not bool(row.get("cumulative_drift_clean", False)):
        failures.append("cumulative_drift")
    if not bool(row.get("blocking_clean", False)):
        failures.append("block_drift")
    if bool(row.get("spread_warning", False)):
        failures.append("spread_warning")
    return failures


def stationarity_row_values(row: dict[str, Any]) -> dict[str, float]:
    return {
        "slope_z_autocorr_adjusted": float(row["slope_z_autocorr_adjusted"]),
        "first_last_quarter_z": float(row["first_last_quarter_z"]),
        "late_cumulative_z": float(row["late_cumulative_z"]),
        "first_last_blocking_z": float(row["first_last_blocking_z"]),
        "spread_blocking_z": float(row["spread_blocking_z"]),
    }


def max_finite(values: Any) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return max(finite) if finite else float("nan")


def summarize_correlated_error_metric(
    seeds: list[int],
    traces: list[np.ndarray],
    spacings: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    finite_stderrs: list[float] = []
    statuses: list[str] = []
    for seed, trace, spacing in zip(seeds, traces, spacings, strict=True):
        if trace.size == 0:
            row = {
                "seed": int(seed),
                "trace_spacing_tau": float(spacing),
                "status": CORRELATED_ERROR_UNAVAILABLE,
                "conservative_stderr": float("nan"),
                "overlap_pair_count": 0,
                "estimates": [],
            }
            rows.append(row)
            statuses.append(CORRELATED_ERROR_UNAVAILABLE)
        else:
            result = triangulated_error_estimate(trace, trace_spacing_tau=spacing)
            rows.append(
                {
                    "seed": int(seed),
                    "trace_spacing_tau": float(spacing),
                    **result.to_dict(),
                }
            )
            statuses.append(result.status)
            if np.isfinite(result.conservative_stderr):
                finite_stderrs.append(float(result.conservative_stderr))

    if not finite_stderrs:
        status = CORRELATED_ERROR_UNAVAILABLE
        case_stderr = float("nan")
    elif any(value == CORRELATED_ERROR_UNAVAILABLE for value in statuses):
        status = CORRELATED_ERROR_UNAVAILABLE
        case_stderr = float("nan")
    elif all(value == CORRELATED_ERROR_AGREEMENT for value in statuses):
        status = CORRELATED_ERROR_AGREEMENT
        case_stderr = float(np.sqrt(np.sum(np.square(finite_stderrs))) / len(seeds))
    else:
        status = CORRELATED_ERROR_DISAGREEMENT
        case_stderr = float(np.sqrt(np.sum(np.square(finite_stderrs))) / len(seeds))

    return {
        "status": status,
        "case_correlated_stderr": case_stderr,
        "seed_count": len(seeds),
        "triangulated_seed_count": int(
            sum(value == CORRELATED_ERROR_AGREEMENT for value in statuses)
        ),
        "disagree_seed_count": int(
            sum(value == CORRELATED_ERROR_DISAGREEMENT for value in statuses)
        ),
        "unavailable_seed_count": int(
            sum(value == CORRELATED_ERROR_UNAVAILABLE for value in statuses)
        ),
        "per_seed": rows,
    }


def trace_spacing_tau(times: np.ndarray) -> float:
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 1.0
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.median(diffs)) if diffs.size else 1.0


def require_trace(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        raise ValueError("DMC streaming summary does not contain trace data")
    return np.asarray(values, dtype=float)


def optional_trace(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def classify_case(
    *,
    validation_passed: bool,
    density_accounting_clean: bool,
    valid_finite_clean: bool,
    population_weights_controlled: bool,
    diagnostics: dict[str, dict],
    final_classification: str,
) -> str:
    if validation_passed:
        if final_classification == TRIANGULATED_PRECISION_WARNING:
            return TRIANGULATED_PRECISION_WARNING
        if final_classification in {"spread_warning", MIXED_OBSERVABLE_WARNING}:
            return final_classification
        if final_classification == MIXED_COORDINATE_DIAGNOSTIC:
            return MIXED_COORDINATE_DIAGNOSTIC
        if any(diagnostics[name]["classification"] == CHAIN_SPREAD_WARNING for name in ("energy",)):
            return CHAIN_SPREAD_WARNING
        return "accepted"
    if not density_accounting_clean:
        return "density_normalization_mismatch"
    if not valid_finite_clean:
        return "nonfinite_samples"
    if not population_weights_controlled:
        return "weight_collapse"
    return "trace_nonstationary"


def classify_grid(rows: list[dict[str, Any]]) -> str:
    if all(row["validation_passed"] for row in rows):
        if any(row["classification"] != "accepted" for row in rows):
            return "accepted_with_warnings"
        return "accepted"
    return "grid_contains_unresolved_case"


def blocked_spread_diagnostics(
    seeds: list[int],
    seed_summaries: list[DMCStreamingSummary],
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
    means = np.asarray(
        [float(np.mean(trace[np.isfinite(trace)])) for trace in traces],
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    plateau_flags: list[bool] = []
    blocked_zscores: list[float] = []
    robust_zscores: list[float] = []
    for index, (seed, trace) in enumerate(zip(seeds, traces, strict=True)):
        trace = np.asarray(trace, dtype=float)
        trace = trace[np.isfinite(trace)]
        curve = blocking_curve(trace, min_blocks=BLOCKING_CURVE_MIN_BLOCKS)
        plateau = detect_blocking_plateau(
            curve.block_sizes,
            curve.n_blocks,
            curve.stderr,
            min_blocks=BLOCKING_PLATEAU_MIN_BLOCKS,
            window=BLOCKING_PLATEAU_WINDOW,
            rel_tol=BLOCKING_PLATEAU_REL_TOL,
            sigma_tol=BLOCKING_PLATEAU_SIGMA_TOL,
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
        "blocked_zscore_within_threshold": bool(
            plateau_all and blocked_z_max <= blocked_z_threshold
        ),
        "per_seed": rows,
    }


def classify_blocked_case(
    *,
    base_numerics_valid: bool,
    population_weight_status: str,
    energy: dict[str, Any],
    rms: dict[str, Any],
    r2: dict[str, Any],
    diagnostics: dict[str, dict],
    stationarity_audit: dict[str, dict[str, Any]],
    correlated_errors: dict[str, dict[str, Any]],
) -> str:
    if population_weight_status == "weight_collapse":
        return "weight_collapse"
    if not base_numerics_valid:
        return "base_numerics_invalid"
    if chain_has_failure(diagnostics, stationarity_audit, metrics=("energy",)):
        return "trace_nonstationary"

    energy_problem = metric_precision_problem(energy, diagnostics["energy"])
    rms_problem = metric_precision_problem(rms, diagnostics["rms"])
    r2_problem = metric_precision_problem(r2, diagnostics["r2"])

    energy_repairable = metric_precision_repairable(
        energy_problem,
        correlated_errors["energy"],
    )

    if energy_problem and not energy_repairable:
        if not energy["plateau_all"]:
            return "plateau_unresolved"
        if not energy["blocked_zscore_within_threshold"]:
            return "seed_disagreement"
        return "plateau_unresolved"

    if energy_problem:
        return TRIANGULATED_PRECISION_WARNING

    if mixed_coordinate_chain_has_failure(diagnostics, stationarity_audit):
        return MIXED_COORDINATE_DIAGNOSTIC
    if rms_problem or r2_problem:
        coordinate_repairable = metric_precision_repairable(
            rms_problem,
            correlated_errors["rms"],
        ) and metric_precision_repairable(
            r2_problem,
            correlated_errors["r2"],
        )
        return MIXED_COORDINATE_DIAGNOSTIC if coordinate_repairable else MIXED_OBSERVABLE_WARNING

    if any(diagnostics[name]["classification"] == CHAIN_SPREAD_WARNING for name in ("energy",)):
        return "spread_warning"
    return "accepted"


def metric_precision_problem(metric: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    del diagnostics
    return bool(not metric["plateau_all"] or not metric["blocked_zscore_within_threshold"])


def metric_precision_repairable(problem: bool, correlated_error: dict[str, Any]) -> bool:
    return (not problem) or correlated_error_available(correlated_error)


def correlated_error_available(metric: dict[str, Any]) -> bool:
    return str(metric["status"]) in {
        CORRELATED_ERROR_AGREEMENT,
        CORRELATED_ERROR_DISAGREEMENT,
    } and np.isfinite(float(metric["case_correlated_stderr"]))


def chain_has_failure(
    diagnostics: dict[str, dict],
    stationarity_audit: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...] = ("energy", "rms", "r2"),
) -> bool:
    if any(
        diagnostics[name]["classification"]
        in {CHAIN_RHAT_ABOVE_LIMIT, CHAIN_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM}
        for name in metrics
    ):
        return True
    return any(
        stationarity_audit[name]["reason"] not in {"accepted", "spread_warning"}
        and diagnostics[name]["classification"] == CHAIN_TRACE_NONSTATIONARY
        for name in metrics
    )


def mixed_coordinate_chain_has_failure(
    diagnostics: dict[str, dict],
    stationarity_audit: dict[str, dict[str, Any]],
) -> bool:
    """Return whether diagnostic mixed coordinate traces failed their own checks.

    Mixed coordinate observables are not pure R2/RMS/density estimators.
    Their stationarity failures must stay visible, but they do not veto the
    Hamiltonian energy corridor.
    """

    return chain_has_failure(diagnostics, stationarity_audit, metrics=("rms", "r2"))


def mixed_coordinate_diagnostic_status(
    *,
    rms: dict[str, Any],
    r2: dict[str, Any],
    diagnostics: dict[str, dict],
    stationarity_audit: dict[str, dict[str, Any]],
) -> str:
    """Classify mixed coordinate traces without changing the energy result."""

    if mixed_coordinate_chain_has_failure(diagnostics, stationarity_audit):
        return "mixed_coordinate_trace_nonstationary"
    if metric_precision_problem(rms, diagnostics["rms"]) or metric_precision_problem(
        r2,
        diagnostics["r2"],
    ):
        return "mixed_coordinate_precision_warning"
    if any(diagnostics[name]["classification"] == CHAIN_SPREAD_WARNING for name in ("rms", "r2")):
        return "mixed_coordinate_spread_warning"
    return "accepted"


def method_status(
    *,
    density_accounting_clean: bool,
    valid_finite_clean: bool,
    population_weight_status: str,
    diagnostics: dict[str, dict],
    stationarity_audit: dict[str, dict[str, Any]],
) -> str:
    if not density_accounting_clean:
        return "density_normalization_mismatch"
    if not valid_finite_clean:
        return "nonfinite_samples"
    if population_weight_status == "weight_collapse":
        return "weight_collapse"
    if chain_has_failure(diagnostics, stationarity_audit, metrics=("energy",)):
        return "trace_nonstationary"
    return "accepted"


def precision_status(final_classification: str) -> str:
    if final_classification == "accepted":
        return "accepted"
    if final_classification == TRIANGULATED_PRECISION_WARNING:
        return TRIANGULATED_PRECISION_WARNING
    if final_classification == MIXED_COORDINATE_DIAGNOSTIC:
        return "accepted"
    if final_classification == MIXED_OBSERVABLE_WARNING:
        return MIXED_OBSERVABLE_WARNING
    if final_classification == "spread_warning":
        return "spread_warning"
    return "not_evaluated_due_to_invalid_numerics"


def classify_population_weight_status(
    *,
    ess_fraction_min: float,
    log_weight_span_max: float,
    ess_warning_fraction: float,
    ess_invalid_fraction: float,
    log_weight_span_warning: float,
) -> str:
    if not np.isfinite(ess_fraction_min) or ess_fraction_min < ess_invalid_fraction:
        return "weight_collapse"
    if ess_fraction_min < ess_warning_fraction:
        return "weight_dispersion_warning"
    if np.isfinite(log_weight_span_max) and log_weight_span_max > log_weight_span_warning:
        return "weight_dispersion_warning"
    return "accepted"


def write_seed_trace_artifacts(
    output_dir: Any,
    case_id: str,
    seeds: list[int],
    seed_summaries: list[DMCStreamingSummary],
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


def seed_trace_dict(summary: DMCStreamingSummary) -> dict[str, np.ndarray]:
    trace_times = require_trace(summary.trace_times)
    return {
        "tau": trace_times,
        "mixed_energy": require_trace(summary.mixed_energy_trace),
        "rms_radius": require_trace(summary.rms_radius_trace),
        "r2_radius": require_trace(summary.r2_radius_trace),
        "local_energy_variance": require_trace(summary.local_energy_variance_trace),
        "local_energy_median": require_trace(summary.local_energy_median_trace),
        "local_energy_mad": require_trace(summary.local_energy_mad_trace),
        "local_energy_p001": require_trace(summary.local_energy_p001_trace),
        "local_energy_p01": require_trace(summary.local_energy_p01_trace),
        "local_energy_p99": require_trace(summary.local_energy_p99_trace),
        "local_energy_p999": require_trace(summary.local_energy_p999_trace),
        "log_weight_span": require_trace(summary.log_weight_span_trace),
        "ess_fraction": require_trace(summary.ess_fraction_trace),
        "invalid_proposal_fraction": require_trace(summary.invalid_proposal_fraction_trace),
        "hard_wall_kill_fraction": require_trace(summary.hard_wall_kill_fraction_trace),
        "local_acceptance_fraction": require_trace(summary.local_acceptance_fraction_trace),
        "metropolis_rejection_fraction": require_trace(summary.metropolis_rejection_fraction_trace),
        "drift_norm_max": require_trace(summary.drift_norm_max_trace),
        "configuration_esjd": require_trace(summary.configuration_esjd_trace),
        "r2_esjd": require_trace(summary.r2_esjd_trace),
        "weighted_free_gap_esjd": require_trace(summary.weighted_free_gap_esjd_trace),
        "weighted_free_gap_mean": require_trace(summary.weighted_free_gap_mean_trace),
        "free_gap_min": require_trace(summary.free_gap_min_trace),
        "free_gap_p01": require_trace(summary.free_gap_p01_trace),
        "zero_weight_excluded_fraction": require_trace(summary.zero_weight_excluded_fraction_trace),
        "scheduled_log_target_mean": require_trace(summary.scheduled_log_target_mean_trace),
        "scheduled_log_proposal_mean": require_trace(summary.scheduled_log_proposal_mean_trace),
        "scheduled_log_weight_increment_mean": require_trace(
            summary.scheduled_log_weight_increment_mean_trace
        ),
        "scheduled_log_weight_increment_variance": require_trace(
            summary.scheduled_log_weight_increment_variance_trace
        ),
        "retained_fraction": require_trace(summary.retained_fraction_trace),
    }


def breathing_trace_summary(summary: DMCStreamingSummary) -> dict[str, float]:
    values = require_trace(summary.rms_radius_trace)
    first, last, z_value = first_last_quarter_summary(values)
    return {
        "first_quarter_rms": first,
        "last_quarter_rms": last,
        "first_last_rms_z": z_value,
    }


def first_last_quarter_summary(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.size < 4:
        return float("nan"), float("nan"), float("nan")
    width = max(1, data.size // 4)
    first = data[:width]
    last = data[-width:]
    first_mean = float(np.mean(first))
    last_mean = float(np.mean(last))
    first_var = float(np.var(first, ddof=1)) if first.size > 1 else 0.0
    last_var = float(np.var(last, ddof=1)) if last.size > 1 else 0.0
    stderr_value = np.sqrt(first_var / first.size + last_var / last.size)
    z_value = (last_mean - first_mean) / stderr_value if stderr_value > 0.0 else float("nan")
    return first_mean, last_mean, float(z_value)


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

    curve = blocking_curve(values, min_blocks=BLOCKING_CURVE_MIN_BLOCKS)
    plateau = detect_blocking_plateau(
        curve.block_sizes,
        curve.n_blocks,
        curve.stderr,
        min_blocks=BLOCKING_PLATEAU_MIN_BLOCKS,
        window=BLOCKING_PLATEAU_WINDOW,
        rel_tol=BLOCKING_PLATEAU_REL_TOL,
        sigma_tol=BLOCKING_PLATEAU_SIGMA_TOL,
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


def uncertainty_summary(
    values: np.ndarray,
    diagnostics: dict[str, Any],
    correlated_error: dict[str, Any],
    *,
    plateau_all: bool,
) -> dict[str, Any]:
    seed_stderr = stderr(np.asarray(values, dtype=float))
    blocking_stderr = combined_blocking_stderr(diagnostics)
    correlated_stderr = float(correlated_error["case_correlated_stderr"])
    conservative = finite_max([seed_stderr, blocking_stderr, correlated_stderr])
    spread_warning = int(diagnostics["spread_warning_count"]) > 0
    correlated_available = correlated_error_available(correlated_error)
    status = "accepted"
    if not plateau_all and not correlated_available:
        status = "plateau_unresolved"
    elif correlated_available and (
        spread_warning
        or not plateau_all
        or blocking_stderr_exceeds_seed(blocking_stderr, seed_stderr)
        or blocking_stderr_exceeds_seed(correlated_stderr, seed_stderr)
    ):
        status = TRIANGULATED_PRECISION_WARNING
    elif spread_warning or blocking_stderr_exceeds_seed(blocking_stderr, seed_stderr):
        status = "conservative_error_inflated"
    if not np.isfinite(conservative):
        status = "uncertainty_unavailable"
    return {
        "seed_stderr": float(seed_stderr),
        "blocking_stderr": float(blocking_stderr),
        "correlated_stderr": float(correlated_stderr),
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
    if "uncertainty_unavailable" in statuses:
        return "uncertainty_unavailable"
    if "plateau_unresolved" in statuses:
        return "plateau_unresolved"
    if TRIANGULATED_PRECISION_WARNING in statuses:
        return TRIANGULATED_PRECISION_WARNING
    if "conservative_error_inflated" in statuses:
        return "conservative_error_inflated"
    return "accepted"


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


def min_seed_metadata(seed_summaries: list[DMCStreamingSummary], key: str) -> float:
    values = [
        float(summary.metadata.get(key, float("nan")))
        for summary in seed_summaries
        if np.isfinite(float(summary.metadata.get(key, float("nan"))))
    ]
    return min(values) if values else float("nan")


def max_seed_metadata(seed_summaries: list[DMCStreamingSummary], key: str) -> float:
    values = [
        float(summary.metadata.get(key, float("nan")))
        for summary in seed_summaries
        if np.isfinite(float(summary.metadata.get(key, float("nan"))))
    ]
    return max(values) if values else float("nan")
