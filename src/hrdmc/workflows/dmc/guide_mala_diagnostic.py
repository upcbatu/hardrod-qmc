from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts import build_run_provenance, write_json, write_run_manifest
from hrdmc.io.progress import ProgressBar
from hrdmc.monte_carlo.dmc.common.guide_api import evaluate_guide
from hrdmc.monte_carlo.dmc.local import (
    metropolis_drift_diffusion_step,
)
from hrdmc.monte_carlo.dmc.local.mobility import (
    free_gap_batch_diagnostics,
    local_step_mobility,
    weighted_free_gap_coordinate,
)
from hrdmc.workflows.dmc.initial_conditions.lattice import (
    initial_walkers_with_metadata,
)
from hrdmc.workflows.dmc.trapped import (
    DMCRunControls,
    TrappedCase,
    build_case_geometry,
    build_guide,
    lda_target_rms,
    make_grid,
)

GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION = "guide_mala_diagnostic_v2"


@dataclass(frozen=True)
class GuideMALADiagnosticControls:
    dt: float
    walkers: int
    duration_tau: float
    store_every: int
    start_scale: float
    grid_extent: float
    n_bins: int
    relative_alpha: float | None = None
    guide_family: str = "reduced-tg"
    contact_beta: float | None = None
    tail_tau: float = 20.0

    @property
    def steps(self) -> int:
        return int(round(self.duration_tau / self.dt))

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.walkers <= 0:
            raise ValueError("walkers must be positive")
        if self.duration_tau <= 0.0:
            raise ValueError("duration_tau must be positive")
        if self.store_every <= 0:
            raise ValueError("store_every must be positive")
        if self.start_scale <= 0.0:
            raise ValueError("start_scale must be positive")
        if self.grid_extent <= 0.0:
            raise ValueError("grid_extent must be positive")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least two")
        if self.relative_alpha is not None and self.relative_alpha <= 0.0:
            raise ValueError("relative_alpha must be positive when provided")
        if self.guide_family not in {
            "reduced-tg",
            "contact-corrected-reduced-tg",
        }:
            raise ValueError("unsupported guide_family for guide-MALA diagnostic")
        if self.guide_family == "contact-corrected-reduced-tg" and self.contact_beta is None:
            raise ValueError("contact-corrected guide requires contact_beta")
        if self.tail_tau <= 0.0 or self.tail_tau > self.duration_tau:
            raise ValueError("tail_tau must lie in (0, duration_tau]")
        reconstructed_tau = self.steps * self.dt
        if not np.isclose(reconstructed_tau, self.duration_tau, rtol=0.0, atol=1e-12):
            raise ValueError("duration_tau must be an integer multiple of dt")


def run_guide_mala_diagnostic(
    case: TrappedCase,
    controls: GuideMALADiagnosticControls,
    seed: int,
    *,
    guide_parameter_source: dict[str, Any] | None = None,
    progress: ProgressBar | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Run guide-squared MALA without DMC branching or population control."""

    controls.validate()
    rng = np.random.default_rng(seed)
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=controls.guide_family,
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
    )
    run_controls = DMCRunControls(
        dt=controls.dt,
        walkers=controls.walkers,
        burn_tau=0.0,
        production_tau=controls.duration_tau,
        store_every=controls.store_every,
        grid_extent=controls.grid_extent,
        n_bins=controls.n_bins,
        local_step_method="metropolis",
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
    )
    grid = make_grid(run_controls, case)
    target_rms = lda_target_rms(case, run_controls, grid)
    initial = initial_walkers_with_metadata(
        system,
        controls.walkers,
        rng,
        initialization_mode="lda-rms-lattice",
        target_initial_rms=target_rms,
        init_width_log_sigma=0.0,
    )
    positions = scale_reduced_free_gaps(
        initial.positions,
        rod_length=system.rod_length,
        scale=controls.start_scale,
    )
    local_energies, valid = evaluate_guide(guide, positions)
    if not np.all(valid):
        raise RuntimeError("scaled guide-MALA start contains invalid walkers")

    trace: list[dict[str, float]] = []
    interval = _IntervalAccumulator()
    initial_metrics = _configuration_metrics(
        positions,
        local_energies,
        rod_length=system.rod_length,
        center=system.center,
    )
    for step_index in range(1, controls.steps + 1):
        old_positions = positions
        result = metropolis_drift_diffusion_step(
            rng,
            positions,
            guide,
            controls.dt,
            local_energies,
        )
        positions = np.asarray(result.positions, dtype=float)
        local_energies = np.asarray(result.local_energies, dtype=float)
        mobility = local_step_mobility(
            old_positions,
            positions,
            center=system.center,
            rod_length=system.rod_length,
        )
        interval.update(result, mobility)
        if progress is not None:
            progress.update(1)
        if step_index % controls.store_every == 0 or step_index == controls.steps:
            row = {
                "tau": float(step_index * controls.dt),
                **_configuration_metrics(
                    positions,
                    local_energies,
                    rod_length=system.rod_length,
                    center=system.center,
                ),
                **interval.finalize(),
            }
            trace.append(row)
            interval = _IntervalAccumulator()

    tail_start = controls.duration_tau - controls.tail_tau
    tail_rows = [row for row in trace if row["tau"] > tail_start]
    final_valid = np.asarray(guide.valid_batch(positions), dtype=bool)
    tail_means = _mean_rows(tail_rows)
    final_metrics = _configuration_metrics(
        positions,
        local_energies,
        rod_length=system.rod_length,
        center=system.center,
    )
    all_final_walkers_valid = bool(np.all(final_valid))
    all_metrics_finite = all(
        np.isfinite(value)
        for metrics in (initial_metrics, final_metrics, tail_means, *trace)
        for value in metrics.values()
    )
    if not all_final_walkers_valid:
        status = "hard_core_validity_failure"
    elif not all_metrics_finite:
        status = "nonfinite_diagnostic"
    else:
        status = "diagnostic_completed"
    summary: dict[str, Any] = {
        "schema_version": GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION,
        "status": status,
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "seed": int(seed),
        "controls": {
            "dt": controls.dt,
            "walkers": controls.walkers,
            "duration_tau": controls.duration_tau,
            "steps": controls.steps,
            "store_every": controls.store_every,
            "start_scale": controls.start_scale,
            "grid_extent": controls.grid_extent,
            "n_bins": controls.n_bins,
            "relative_alpha": controls.relative_alpha,
            "guide_family": controls.guide_family,
            "contact_beta": controls.contact_beta,
            "tail_tau": controls.tail_tau,
        },
        "physics": {
            "branching_enabled": False,
            "population_resampling_enabled": False,
            "walker_weights_enabled": False,
            "stationary_target": "guide_squared",
            "proposal": "whole_configuration_metropolis_drift_diffusion",
        },
        "guide_parameter_source": guide_parameter_source or {"kind": "explicit"},
        "initialization": {
            **initial.metadata,
            "reduced_free_gap_scale": controls.start_scale,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "tail_means": tail_means,
        "trace_rows": len(trace),
        "tail_trace_rows": len(tail_rows),
        "all_final_walkers_valid": all_final_walkers_valid,
    }
    return summary, trace


def guide_mala_manifest_config(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic run identity duplicated into the generic manifest."""

    return {
        "case_id": summary["case_id"],
        "seed": summary["seed"],
        "controls": summary["controls"],
        "physics": summary["physics"],
        "guide_parameter_source": summary["guide_parameter_source"],
    }


def write_guide_mala_diagnostic_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    trace: list[dict[str, float]],
    command: list[str] | None,
) -> dict[str, Path]:
    """Persist one immutable, source-bound guide-MALA calibration packet."""

    if summary.get("schema_version") != GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("guide-MALA summary has an unsupported schema")
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.csv"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"guide-MALA output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    write_guide_mala_trace(trace_path, trace)
    written_manifest = write_run_manifest(
        output_dir,
        run_name="guide_mala_diagnostic",
        config=guide_mala_manifest_config(summary),
        artifacts=[summary_path, trace_path],
        schema_version=GUIDE_MALA_DIAGNOSTIC_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=str(summary["status"]),
    )
    return {
        "summary": summary_path,
        "trace": trace_path,
        "run_manifest": written_manifest,
    }


def scale_reduced_free_gaps(
    positions: np.ndarray,
    *,
    rod_length: float,
    scale: float,
) -> np.ndarray:
    """Scale internal reduced coordinates while preserving each walker COM."""

    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("positions must have shape (n_walkers, n_particles)")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    n_particles = values.shape[1]
    offsets = rod_length * (np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1))
    reduced = values - offsets[np.newaxis, :]
    center_of_mass = np.mean(reduced, axis=1, keepdims=True)
    scaled_reduced = center_of_mass + scale * (reduced - center_of_mass)
    return scaled_reduced + offsets[np.newaxis, :]


def write_guide_mala_trace(path: Path, rows: list[dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("guide-MALA trace must contain at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class _IntervalAccumulator:
    def __init__(self) -> None:
        self.steps = 0
        self.acceptance_sum = 0.0
        self.invalid_sum = 0.0
        self.rejection_sum = 0.0
        self.drift_max = float("-inf")
        self.configuration_esjd_sum = 0.0
        self.r2_esjd_sum = 0.0
        self.weighted_free_gap_esjd_sum = 0.0

    def update(self, result: Any, mobility: Any) -> None:
        self.steps += 1
        self.acceptance_sum += float(np.mean(result.accepted))
        self.invalid_sum += float(np.mean(result.invalid_proposal))
        self.rejection_sum += float(np.mean(result.metropolis_rejected))
        self.drift_max = max(self.drift_max, float(result.drift_norm_max))
        self.configuration_esjd_sum += float(mobility.configuration_esjd)
        self.r2_esjd_sum += float(mobility.r2_esjd)
        self.weighted_free_gap_esjd_sum += float(mobility.weighted_free_gap_esjd)

    def finalize(self) -> dict[str, float]:
        if self.steps <= 0:
            raise RuntimeError("cannot finalize an empty guide-MALA interval")
        return {
            "acceptance_fraction": self.acceptance_sum / self.steps,
            "invalid_proposal_fraction": self.invalid_sum / self.steps,
            "metropolis_rejection_fraction": self.rejection_sum / self.steps,
            "drift_norm_max": self.drift_max,
            "configuration_esjd": self.configuration_esjd_sum / self.steps,
            "r2_esjd": self.r2_esjd_sum / self.steps,
            "weighted_free_gap_esjd": self.weighted_free_gap_esjd_sum / self.steps,
        }


def _configuration_metrics(
    positions: np.ndarray,
    local_energies: np.ndarray,
    *,
    rod_length: float,
    center: float,
) -> dict[str, float]:
    weights = np.full(positions.shape[0], 1.0 / positions.shape[0], dtype=float)
    gap = free_gap_batch_diagnostics(
        positions,
        weights,
        rod_length=rod_length,
    )
    median = float(np.median(local_energies))
    return {
        "local_energy_mean": float(np.mean(local_energies)),
        "local_energy_median": median,
        "local_energy_mad": float(np.median(np.abs(local_energies - median))),
        "local_energy_p99": float(np.quantile(local_energies, 0.99)),
        "local_energy_p999": float(np.quantile(local_energies, 0.999)),
        "r2_mean": float(np.mean((positions - center) ** 2)),
        "weighted_free_gap_mean": float(
            np.mean(weighted_free_gap_coordinate(positions, rod_length=rod_length))
        ),
        "free_gap_min": gap["free_gap_min"],
        "free_gap_p01": gap["free_gap_p01"],
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("guide-MALA tail window contains no stored rows")
    keys = [key for key in rows[0] if key != "tau"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
