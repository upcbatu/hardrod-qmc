from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.schema import to_jsonable
from hrdmc.sampling.initial_conditions import initial_walkers_with_metadata
from hrdmc.sampling.vmc.engine import run_vmc_streaming
from hrdmc.sampling.vmc.results import VMCTransitionEvent
from hrdmc.sampling.vmc.transitions import VMCConfig
from hrdmc.system.settings import TrappedCase, build_case_geometry
from hrdmc.theory.lda import lda_density_profile, lda_rms_radius
from hrdmc.trial.guide import ReducedTGHardRodGuide
from hrdmc.trial.relative_width import (
    RelativeWidthSample,
    optimize_relative_alpha,
    relative_alpha_metrics,
    relative_width_sample,
)


@dataclass(frozen=True)
class AlphaOptimizationControls:
    dt: float
    walkers: int
    burn_in_steps: int
    production_steps: int
    sample_stride_steps: int
    drift_limiter: str
    grid_extent: float
    n_bins: int
    reference_relative_alpha: float
    alpha_log_half_width: float
    alpha_grid_points: int
    min_reweight_ess_fraction: float
    max_configurations: int

    @property
    def stored_configuration_count(self) -> int:
        return (self.production_steps // self.sample_stride_steps) * self.walkers

    def validate(self) -> None:
        VMCConfig(
            walkers=self.walkers,
            burn_in_steps=self.burn_in_steps,
            production_steps=self.production_steps,
            method="mala",
            dt=self.dt,
            drift_limiter=self.drift_limiter,
        ).validate()
        if self.sample_stride_steps <= 0:
            raise ValueError("sample_stride_steps must be positive")
        if self.production_steps < self.sample_stride_steps:
            raise ValueError("production_steps must contain at least one stored sample")
        if self.grid_extent <= 0.0 or self.n_bins < 2:
            raise ValueError("grid_extent and n_bins must define a valid LDA grid")
        if self.reference_relative_alpha <= 0.0:
            raise ValueError("reference_relative_alpha must be positive")
        if self.alpha_log_half_width <= 0.0 or self.alpha_grid_points < 5:
            raise ValueError("alpha search window is invalid")
        if not 0.0 < self.min_reweight_ess_fraction <= 1.0:
            raise ValueError("min_reweight_ess_fraction must lie in (0, 1]")
        if self.stored_configuration_count > self.max_configurations:
            raise ValueError(
                "stored sample exceeds max_configurations; increase "
                "sample_stride_steps or max_configurations"
            )


class _AlphaSampleCollector:
    def __init__(self, *, rod_length: float, omega: float, stride: int) -> None:
        self._rod_length = float(rod_length)
        self._omega = float(omega)
        self._stride = int(stride)
        self._base_local_energy: list[np.ndarray] = []
        self._internal_norm2: list[np.ndarray] = []

    def record_vmc_transition(self, event: VMCTransitionEvent) -> None:
        if event.production_step % self._stride:
            return
        sample = relative_width_sample(
            event.positions,
            rod_length=self._rod_length,
            omega=self._omega,
        )
        self._base_local_energy.append(sample.base_local_energy)
        self._internal_norm2.append(sample.internal_norm2)

    def finish(self) -> RelativeWidthSample:
        if not self._base_local_energy:
            raise RuntimeError("alpha optimization collected no configurations")
        return RelativeWidthSample(
            base_local_energy=np.concatenate(self._base_local_energy),
            internal_norm2=np.concatenate(self._internal_norm2),
        )


def run_alpha_optimization(
    case: TrappedCase,
    controls: AlphaOptimizationControls,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, float]], RelativeWidthSample]:
    started = time.perf_counter()
    controls.validate()
    if case.rod_length == 0.0:
        raise ValueError("the exact hard-point guide does not require alpha optimization")
    system, trap = build_case_geometry(case)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=case.omega,
        relative_alpha=controls.reference_relative_alpha,
    )
    initial_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    grid = np.linspace(-controls.grid_extent, controls.grid_extent, controls.n_bins)
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(case.n_particles),
        rod_length=case.rod_length,
    )
    initial = initial_walkers_with_metadata(
        system,
        controls.walkers,
        initial_rng,
        initialization_mode="lda-rms-lattice",
        target_initial_rms=lda_rms_radius(lda, center=trap.center),
        init_width_log_sigma=0.0,
    )
    collector = _AlphaSampleCollector(
        rod_length=case.rod_length,
        omega=case.omega,
        stride=controls.sample_stride_steps,
    )
    vmc = run_vmc_streaming(
        initial_positions=initial.positions,
        guide=guide,
        config=VMCConfig(
            walkers=controls.walkers,
            burn_in_steps=controls.burn_in_steps,
            production_steps=controls.production_steps,
            method="mala",
            dt=controls.dt,
            drift_limiter=controls.drift_limiter,
        ),
        seed=seed,
        observer=collector,
    )
    sample = collector.finish()
    if sample.size != controls.stored_configuration_count:
        raise RuntimeError("collected alpha sample does not match the run plan")
    rows, optimum, search = optimize_relative_alpha(
        sample,
        omega=case.omega,
        n_particles=case.n_particles,
        reference_relative_alpha=controls.reference_relative_alpha,
        alpha_log_half_width=controls.alpha_log_half_width,
        alpha_grid_points=controls.alpha_grid_points,
        min_reweight_ess_fraction=controls.min_reweight_ess_fraction,
    )
    reference = relative_alpha_metrics(
        sample,
        omega=case.omega,
        n_particles=case.n_particles,
        reference_relative_alpha=controls.reference_relative_alpha,
        relative_alpha=controls.reference_relative_alpha,
    )
    summary: dict[str, Any] = {
        "schema_version": "reduced_tg_relative_alpha_optimization_v1",
        "status": search["status"],
        "case_id": case.case_id,
        "seed": int(seed),
        "controls": asdict(controls),
        "sample": {
            "configuration_count": sample.size,
            "snapshot_count": sample.size // controls.walkers,
            "acceptance_rate": vmc.production_attempts.acceptance_rate,
            "initialization": initial.metadata,
        },
        "objective": "correlated_sampling_weighted_local_energy_variance",
        "reference_metrics": reference,
        "recommended_relative_alpha": optimum["relative_alpha"],
        "candidate_metrics": optimum,
        "variance_reduction_fraction": (
            1.0 - optimum["local_energy_variance"] / reference["local_energy_variance"]
        ),
        "search": search,
        "requires_independent_vmc_validation": True,
        "calculation_wall_seconds": time.perf_counter() - started,
        "vmc_sampling_wall_seconds": vmc.wall_seconds,
    }
    return summary, rows, sample


def write_alpha_optimization_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, float]],
    sample: RelativeWidthSample,
) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"alpha optimization output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    grid_path = output_dir / "candidate_grid.csv"
    sample_path = output_dir / "sufficient_statistics.npz"
    summary_path.write_text(
        json.dumps(to_jsonable(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with grid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        sample_path,
        base_local_energy=sample.base_local_energy,
        internal_norm2=sample.internal_norm2,
    )
    return {
        "summary": summary_path,
        "candidate_grid": grid_path,
        "sufficient_statistics": sample_path,
    }
