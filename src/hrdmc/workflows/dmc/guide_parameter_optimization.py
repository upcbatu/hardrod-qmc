from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from hrdmc.artifacts import (
    build_run_provenance,
    verify_run_manifest,
    write_json,
    write_run_manifest,
)
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.io.progress import ProgressBar
from hrdmc.monte_carlo.dmc.common.guide_api import evaluate_guide
from hrdmc.monte_carlo.dmc.local import metropolis_drift_diffusion_step
from hrdmc.wavefunctions import ContactTGSufficientStatistics
from hrdmc.wavefunctions.guides import ContactCorrectedReducedTGHardRodGuide
from hrdmc.workflows.dmc.guide_mala_diagnostic import scale_reduced_free_gaps
from hrdmc.workflows.dmc.initial_conditions.lattice import (
    initial_walkers_with_metadata,
)
from hrdmc.workflows.dmc.trapped import (
    DMCRunControls,
    TrappedCase,
    build_case_geometry,
    lda_target_rms,
    make_grid,
)

FloatArray = NDArray[np.float64]
CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION = "contact_guide_correlated_optimization_v2"


@dataclass(frozen=True)
class ContactGuideOptimizationControls:
    dt: float
    walkers: int
    burn_tau: float
    sample_tau: float
    sample_stride_steps: int
    start_scale: float
    grid_extent: float
    n_bins: int
    reference_relative_alpha: float
    reference_contact_beta: float = 0.0
    alpha_log_half_width: float = 0.2
    alpha_grid_points: int = 31
    beta_initial_max: float = 0.25
    beta_grid_points: int = 26
    min_reweight_ess_fraction: float = 0.10
    max_configurations: int = 100_000
    drift_limiter: str = "none"

    @property
    def burn_steps(self) -> int:
        return int(round(self.burn_tau / self.dt))

    @property
    def sample_steps(self) -> int:
        return int(round(self.sample_tau / self.dt))

    @property
    def total_steps(self) -> int:
        return self.burn_steps + self.sample_steps

    @property
    def stored_configuration_count(self) -> int:
        return (self.sample_steps // self.sample_stride_steps) * self.walkers

    def validate(self) -> None:
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be finite and positive")
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")
        if self.walkers <= 0:
            raise ValueError("walkers must be positive")
        if self.burn_tau <= 0.0 or self.sample_tau <= 0.0:
            raise ValueError("burn_tau and sample_tau must be positive")
        if self.sample_stride_steps <= 0:
            raise ValueError("sample_stride_steps must be positive")
        if self.sample_steps < self.sample_stride_steps:
            raise ValueError("sample_tau must contain at least one stored sample")
        for tau_name, tau, steps in (
            ("burn_tau", self.burn_tau, self.burn_steps),
            ("sample_tau", self.sample_tau, self.sample_steps),
        ):
            if not np.isclose(steps * self.dt, tau, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"{tau_name} must be an integer multiple of dt")
        if self.start_scale <= 0.0:
            raise ValueError("start_scale must be positive")
        if self.grid_extent <= 0.0 or self.n_bins < 2:
            raise ValueError("grid extent and bin count are invalid")
        if self.reference_relative_alpha <= 0.0:
            raise ValueError("reference_relative_alpha must be positive")
        if not 0.0 <= self.reference_contact_beta <= 1.0:
            raise ValueError("reference_contact_beta must lie in [0, 1]")
        if self.alpha_log_half_width <= 0.0:
            raise ValueError("alpha_log_half_width must be positive")
        if self.alpha_grid_points < 5 or self.beta_grid_points < 3:
            raise ValueError("optimization grids are too small")
        if not 0.0 < self.beta_initial_max <= 1.0:
            raise ValueError("beta_initial_max must lie in (0, 1]")
        if not 0.0 < self.min_reweight_ess_fraction <= 1.0:
            raise ValueError("min_reweight_ess_fraction must lie in (0, 1]")
        if self.stored_configuration_count > self.max_configurations:
            raise ValueError(
                "requested sufficient-statistics sample exceeds max_configurations; "
                "increase sample_stride_steps or max_configurations"
            )


def run_contact_guide_optimization(
    case: TrappedCase,
    controls: ContactGuideOptimizationControls,
    seed: int,
    *,
    progress: ProgressBar | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]], ContactTGSufficientStatistics]:
    """Optimize split width and partial contact correction by correlated sampling."""

    controls.validate()
    system, trap = build_case_geometry(case)
    guide = ContactCorrectedReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=case.omega,
        relative_alpha=controls.reference_relative_alpha,
        contact_beta=controls.reference_contact_beta,
    )
    rng = np.random.default_rng(seed)
    run_controls = DMCRunControls(
        dt=controls.dt,
        walkers=controls.walkers,
        burn_tau=controls.burn_tau,
        production_tau=controls.sample_tau,
        store_every=controls.sample_stride_steps,
        grid_extent=controls.grid_extent,
        n_bins=controls.n_bins,
        local_step_method="metropolis",
        drift_limiter=controls.drift_limiter,
        relative_alpha=controls.reference_relative_alpha,
        contact_beta=controls.reference_contact_beta,
    )
    grid = make_grid(run_controls, case)
    initial = initial_walkers_with_metadata(
        system,
        controls.walkers,
        rng,
        initialization_mode="lda-rms-lattice",
        target_initial_rms=lda_target_rms(case, run_controls, grid),
        init_width_log_sigma=0.0,
    )
    positions = scale_reduced_free_gaps(
        initial.positions,
        rod_length=system.rod_length,
        scale=controls.start_scale,
    )
    local_energies, valid = evaluate_guide(guide, positions)
    if not np.all(valid):
        raise RuntimeError("contact-guide optimization start contains invalid walkers")

    accepted = 0
    invalid = 0
    proposed = 0
    stored: dict[str, list[FloatArray]] = {
        name: []
        for name in (
            "base_local_energy",
            "internal_norm2",
            "correction_log_sum",
            "correction_lap_sum",
            "harmonic_contact_cross",
            "internal_contact_cross",
            "contact_gradient_norm2",
        )
    }
    for step_index in range(1, controls.total_steps + 1):
        result = metropolis_drift_diffusion_step(
            rng,
            positions,
            guide,
            controls.dt,
            local_energies,
            drift_limiter=controls.drift_limiter,
        )
        positions = np.asarray(result.positions, dtype=float)
        local_energies = np.asarray(result.local_energies, dtype=float)
        accepted += int(np.sum(result.accepted)) if result.accepted is not None else 0
        invalid += (
            int(np.sum(result.invalid_proposal)) if result.invalid_proposal is not None else 0
        )
        proposed += controls.walkers
        if progress is not None:
            progress.update(1)
        sample_index = step_index - controls.burn_steps
        if sample_index <= 0 or sample_index % controls.sample_stride_steps != 0:
            continue
        stats = guide.sufficient_statistics_batch(positions)
        if not np.all(stats.finite):
            raise RuntimeError("non-finite contact-guide sufficient statistics")
        for name in stored:
            stored[name].append(np.asarray(getattr(stats, name), dtype=float).copy())

    concatenated = {name: np.concatenate(chunks) for name, chunks in stored.items()}
    sample = ContactTGSufficientStatistics(
        **concatenated,
        finite=np.ones(concatenated["base_local_energy"].size, dtype=bool),
    )
    if _sample_size(sample) != controls.stored_configuration_count:
        raise RuntimeError("stored sufficient-statistics count does not match run plan")
    rows, optimum, search = optimize_contact_guide_from_sample(
        sample,
        omega=case.omega,
        n_particles=case.n_particles,
        reference_relative_alpha=controls.reference_relative_alpha,
        reference_contact_beta=controls.reference_contact_beta,
        alpha_log_half_width=controls.alpha_log_half_width,
        alpha_grid_points=controls.alpha_grid_points,
        beta_initial_max=controls.beta_initial_max,
        beta_grid_points=controls.beta_grid_points,
        min_reweight_ess_fraction=controls.min_reweight_ess_fraction,
    )
    reference = candidate_metrics(
        sample,
        omega=case.omega,
        n_particles=case.n_particles,
        reference_relative_alpha=controls.reference_relative_alpha,
        reference_contact_beta=controls.reference_contact_beta,
        relative_alpha=controls.reference_relative_alpha,
        contact_beta=controls.reference_contact_beta,
    )
    overlap_ok = optimum["reweight_ess_fraction"] >= controls.min_reweight_ess_fraction
    alpha_low = controls.reference_relative_alpha * math.exp(-controls.alpha_log_half_width)
    alpha_high = controls.reference_relative_alpha * math.exp(controls.alpha_log_half_width)
    alpha_at_boundary = bool(
        optimum["relative_alpha"] <= alpha_low * 1.001
        or optimum["relative_alpha"] >= alpha_high / 1.001
    )
    beta_search_upper = float(search["beta_max_sequence"][-1])
    beta_boundary_tolerance = max(1.0e-7, 1.0e-3 * beta_search_upper)
    beta_at_boundary = bool(optimum["contact_beta"] >= beta_search_upper - beta_boundary_tolerance)
    status = "optimization_candidate"
    if not overlap_ok:
        status = "reweight_overlap_insufficient"
    elif alpha_at_boundary:
        status = "reference_recenter_required"
    elif beta_at_boundary:
        status = "contact_beta_boundary_reached"
    summary: dict[str, Any] = {
        "schema_version": CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION,
        "status": status,
        "case_id": case.case_id,
        "seed": int(seed),
        "controls": {
            **controls.__dict__,
            "burn_steps": controls.burn_steps,
            "sample_steps": controls.sample_steps,
            "total_steps": controls.total_steps,
        },
        "physics": {
            "branching_enabled": False,
            "population_resampling_enabled": False,
            "stationary_target": "reference_guide_squared",
            "objective": "correlated_sampling_weighted_local_energy_variance",
            "com_width": case.omega,
            "contact_beta_bounds": [0.0, 1.0],
            "overlap_metric_scope": (
                "self-normalized importance-weight overlap over correlated stored "
                "configurations; not an independent-sample MCMC ESS"
            ),
        },
        "sample": {
            "configuration_count": _sample_size(sample),
            "snapshot_count": _sample_size(sample) // controls.walkers,
            "acceptance_fraction": accepted / proposed,
            "invalid_proposal_fraction": invalid / proposed,
            "initialization": initial.metadata,
        },
        "reference_metrics": reference,
        "recommended_parameters": {
            "relative_alpha": optimum["relative_alpha"],
            "contact_beta": optimum["contact_beta"],
        },
        "candidate_metrics": optimum,
        "parameter_boundary": {
            "relative_alpha_at_boundary": alpha_at_boundary,
            "contact_beta_at_upper_search_boundary": beta_at_boundary,
            "contact_beta_search_upper": beta_search_upper,
        },
        "overlap_assessment": {
            "observed_reweight_ess_fraction": optimum["reweight_ess_fraction"],
            "minimum_reweight_ess_fraction": controls.min_reweight_ess_fraction,
            "threshold_role": (
                "conservative correlated-sampling overlap heuristic requiring direct "
                "MALA validation; not a universal acceptance threshold"
            ),
            "passed": overlap_ok,
        },
        "variance_reduction_fraction": 1.0
        - optimum["local_energy_variance"] / reference["local_energy_variance"],
        "search": search,
        "requires_direct_mala_validation": True,
        "next_validation": (
            "independent compact and expanded guide-squared MALA runs before DMC branching"
        ),
    }
    return summary, rows, sample


def optimize_contact_guide_from_sample(
    sample: ContactTGSufficientStatistics,
    *,
    omega: float,
    n_particles: int,
    reference_relative_alpha: float,
    reference_contact_beta: float,
    alpha_log_half_width: float,
    alpha_grid_points: int,
    beta_initial_max: float,
    beta_grid_points: int,
    min_reweight_ess_fraction: float,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, Any]]:
    alpha_low = reference_relative_alpha * math.exp(-alpha_log_half_width)
    alpha_high = reference_relative_alpha * math.exp(alpha_log_half_width)
    alpha_values = np.exp(np.linspace(math.log(alpha_low), math.log(alpha_high), alpha_grid_points))
    beta_max = beta_initial_max
    rows: list[dict[str, float]] = []
    expansions: list[float] = []
    best: dict[str, float] | None = None
    while True:
        beta_values = np.linspace(0.0, beta_max, beta_grid_points)
        current_rows = [
            candidate_metrics(
                sample,
                omega=omega,
                n_particles=n_particles,
                reference_relative_alpha=reference_relative_alpha,
                reference_contact_beta=reference_contact_beta,
                relative_alpha=float(alpha),
                contact_beta=float(beta),
            )
            for alpha in alpha_values
            for beta in beta_values
        ]
        rows.extend(current_rows)
        eligible = [
            row for row in current_rows if row["reweight_ess_fraction"] >= min_reweight_ess_fraction
        ]
        pool = eligible if eligible else current_rows
        best = min(pool, key=lambda row: row["local_energy_variance"])
        expansions.append(beta_max)
        at_upper = best["contact_beta"] >= beta_max - 0.5 * beta_max / (beta_grid_points - 1)
        if not at_upper or beta_max >= 1.0:
            break
        beta_max = min(1.0, 2.0 * beta_max)

    assert best is not None

    def objective(theta: FloatArray) -> float:
        metrics = candidate_metrics(
            sample,
            omega=omega,
            n_particles=n_particles,
            reference_relative_alpha=reference_relative_alpha,
            reference_contact_beta=reference_contact_beta,
            relative_alpha=float(math.exp(theta[0])),
            contact_beta=float(theta[1]),
        )
        variance = max(metrics["local_energy_variance"], np.finfo(float).tiny)
        shortfall = max(0.0, min_reweight_ess_fraction - metrics["reweight_ess_fraction"])
        return float(math.log(variance) + 1.0e4 * shortfall * shortfall)

    refined = minimize(
        objective,
        x0=np.asarray([math.log(best["relative_alpha"]), best["contact_beta"]]),
        method="Powell",
        bounds=((math.log(alpha_low), math.log(alpha_high)), (0.0, beta_max)),
        options={"xtol": 1.0e-7, "ftol": 1.0e-9, "maxiter": 800},
    )
    refined_alpha = float(np.clip(math.exp(refined.x[0]), alpha_low, alpha_high))
    refined_beta = float(np.clip(refined.x[1], 0.0, beta_max))
    optimum = candidate_metrics(
        sample,
        omega=omega,
        n_particles=n_particles,
        reference_relative_alpha=reference_relative_alpha,
        reference_contact_beta=reference_contact_beta,
        relative_alpha=refined_alpha,
        contact_beta=refined_beta,
    )
    return (
        rows,
        optimum,
        {
            "alpha_bounds": [alpha_low, alpha_high],
            "beta_max_sequence": expansions,
            "grid_row_count": len(rows),
            "refinement_success": bool(refined.success),
            "refinement_message": str(refined.message),
            "minimum_reweight_ess_fraction": min_reweight_ess_fraction,
            "minimum_reweight_ess_fraction_scope": (
                "importance-weight overlap heuristic over correlated configurations"
            ),
        },
    )


def candidate_metrics(
    sample: ContactTGSufficientStatistics,
    *,
    omega: float,
    n_particles: int,
    reference_relative_alpha: float,
    reference_contact_beta: float,
    relative_alpha: float,
    contact_beta: float,
) -> dict[str, float]:
    log_weights = 2.0 * sample.log_amplitude_difference(
        relative_alpha=relative_alpha,
        contact_beta=contact_beta,
        reference_relative_alpha=reference_relative_alpha,
        reference_contact_beta=reference_contact_beta,
    )
    shifted = log_weights - float(np.max(log_weights))
    raw_weights = np.exp(shifted)
    weights = raw_weights / float(np.sum(raw_weights))
    ess = float(1.0 / np.sum(weights * weights))
    local_energy = sample.local_energy(
        relative_alpha=relative_alpha,
        contact_beta=contact_beta,
        omega=omega,
        n_particles=n_particles,
    )
    mean = float(np.sum(weights * local_energy))
    centered = local_energy - mean
    variance = float(np.sum(weights * centered * centered))
    median = _weighted_quantile(local_energy, weights, 0.5)
    mad = _weighted_quantile(np.abs(local_energy - median), weights, 0.5)
    q001 = _weighted_quantile(local_energy, weights, 0.001)
    q01 = _weighted_quantile(local_energy, weights, 0.01)
    q99 = _weighted_quantile(local_energy, weights, 0.99)
    q999 = _weighted_quantile(local_energy, weights, 0.999)
    return {
        "relative_alpha": float(relative_alpha),
        "contact_beta": float(contact_beta),
        "local_energy_mean": mean,
        "local_energy_variance": variance,
        "local_energy_std": float(math.sqrt(max(variance, 0.0))),
        "local_energy_median": median,
        "local_energy_mad": mad,
        "local_energy_q99_q01_span": q99 - q01,
        "local_energy_q999_q001_span": q999 - q001,
        "reweight_ess": ess,
        "reweight_ess_fraction": ess / _sample_size(sample),
        "max_normalized_weight": float(np.max(weights)),
    }


def load_contact_optimization_candidate(
    path: Path,
    *,
    case: TrappedCase,
) -> tuple[float, float]:
    """Load a manifest-bound candidate for guide-MALA calibration only."""

    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("optimization summary has no run manifest")
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        raise ValueError(f"optimization run manifest failed verification: {'; '.join(errors)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != "contact_guide_correlated_optimization":
        raise ValueError("optimization run manifest has the wrong owner")
    if manifest.get("result_schema_version") != CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION:
        raise ValueError("optimization run manifest has the wrong result schema")
    if manifest.get("status") != "optimization_candidate":
        raise ValueError("optimization run manifest does not identify a usable candidate")
    try:
        summary_relative = path.resolve().relative_to(path.parent.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("optimization summary is outside its manifest directory") from exc
    artifact_paths = {str(entry.get("path", "")) for entry in manifest.get("artifacts", [])}
    if summary_relative not in artifact_paths:
        raise ValueError("optimization summary is not listed by its run manifest")
    implementation = manifest.get("provenance", {}).get("implementation", {})
    if implementation.get("status") != "identified":
        raise ValueError("optimization run manifest has no identified source tree")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION:
        raise ValueError("optimization summary has an unsupported schema")
    if payload.get("status") != "optimization_candidate":
        raise ValueError("optimization summary does not contain a usable candidate")
    if payload.get("case_id") != case.case_id:
        raise ValueError(
            f"optimization summary is for {payload.get('case_id')}, expected {case.case_id}"
        )
    expected_config = {
        "case_id": payload.get("case_id"),
        "seed": payload.get("seed"),
        "controls": to_jsonable(payload.get("controls")),
        "physics": to_jsonable(payload.get("physics")),
    }
    if manifest.get("config") != expected_config:
        raise ValueError("optimization run manifest config does not match summary semantics")
    parameters = payload.get("recommended_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("optimization summary has no recommended_parameters")
    alpha = float(parameters["relative_alpha"])
    beta = float(parameters["contact_beta"])
    if alpha <= 0.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("optimization summary contains invalid guide parameters")
    return alpha, beta


def write_contact_guide_optimization_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, float]],
    sample: ContactTGSufficientStatistics,
    command: list[str] | None,
) -> dict[str, Path]:
    """Persist a source-bound correlated-sampling candidate packet."""

    if summary.get("schema_version") != CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION:
        raise ValueError("contact-guide optimization summary has an unsupported schema")
    summary_path = output_dir / "summary.json"
    grid_path = output_dir / "candidate_grid.csv"
    sample_path = output_dir / "sufficient_statistics.npz"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"contact-guide optimization output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    write_contact_optimization_grid(grid_path, rows)
    write_contact_sufficient_sample(sample_path, sample)
    written_manifest = write_run_manifest(
        output_dir,
        run_name="contact_guide_correlated_optimization",
        config={
            "case_id": summary["case_id"],
            "seed": summary["seed"],
            "controls": to_jsonable(summary["controls"]),
            "physics": to_jsonable(summary["physics"]),
        },
        artifacts=[summary_path, grid_path, sample_path],
        schema_version=CONTACT_GUIDE_OPTIMIZATION_SCHEMA_VERSION,
        provenance=build_run_provenance(command),
        status=str(summary["status"]),
    )
    return {
        "summary": summary_path,
        "candidate_grid": grid_path,
        "sufficient_statistics": sample_path,
        "run_manifest": written_manifest,
    }


def write_contact_optimization_grid(path: Path, rows: list[dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_contact_sufficient_sample(
    path: Path,
    sample: ContactTGSufficientStatistics,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _sufficient_statistics_arrays(sample)
    with path.open("wb") as handle:
        save_compressed = cast(Any, np.savez_compressed)
        save_compressed(handle, **arrays)
    return path


def _sample_size(sample: ContactTGSufficientStatistics) -> int:
    return int(sample.base_local_energy.size)


def _sufficient_statistics_arrays(
    sample: ContactTGSufficientStatistics,
) -> dict[str, np.ndarray]:
    return {
        "base_local_energy": sample.base_local_energy,
        "internal_norm2": sample.internal_norm2,
        "correction_log_sum": sample.correction_log_sum,
        "correction_lap_sum": sample.correction_lap_sum,
        "harmonic_contact_cross": sample.harmonic_contact_cross,
        "internal_contact_cross": sample.internal_contact_cross,
        "contact_gradient_norm2": sample.contact_gradient_norm2,
        "finite": sample.finite,
    }


def _weighted_quantile(values: FloatArray, weights: FloatArray, probability: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, probability, side="left")), values.size - 1)
    return float(ordered_values[index])
