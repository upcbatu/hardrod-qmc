from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_requested
from hrdmc.io.progress import QueuedProgress
from hrdmc.monte_carlo.dmc.rn_block import (
    RNBlockDMCConfig,
    RNBlockStreamingSummary,
    run_rn_block_dmc_streaming,
)
from hrdmc.runners import run_seed_batch
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
    OpenLineHardRodSystem,
    OrderedHarmonicMehlerKernel,
)
from hrdmc.wavefunctions import ReducedTGHardRodGuide
from hrdmc.workflows.dmc.rn_block import (
    RNRunControls,
    controls_to_dict,
    initial_walkers,
    resolve_parallel_workers,
    rn_progress_bar,
    rn_run_config,
    write_rn_run_artifacts,
)


@dataclass(frozen=True)
class ExactTGTrapConfig:
    n_particles: int
    omega: float

    @property
    def exact_energy_total(self) -> float:
        return float(self.n_particles * self.n_particles * self.omega / np.sqrt(2.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate RN-block DMC against exact trapped TG.")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--seeds", default="301,302,303,304")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.005)
    parser.add_argument("--burn-tau", type=float, default=1.0)
    parser.add_argument("--production-tau", type=float, default=2.0)
    parser.add_argument("--store-every", type=int, default=10)
    parser.add_argument("--grid-extent", type=float, default=12.0)
    parser.add_argument("--n-bins", type=int, default=160)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--energy-tolerance", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    exact_config = ExactTGTrapConfig(n_particles=args.n_particles, omega=args.omega)
    controls = RNRunControls(
        dt=args.dt,
        walkers=args.walkers,
        tau_block=args.tau,
        rn_cadence_tau=args.rn_cadence,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
    )
    seeds = _parse_seeds(args.seeds)
    requested_workers = resolve_parallel_workers(len(seeds), args.parallel_workers)
    density_grid = np.linspace(-controls.grid_extent, controls.grid_extent, controls.n_bins)
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label=f"RN exact TG N{args.n_particles}",
        enabled=progress_requested(args.progress),
    ) as bar:
        seed_summaries, actual_workers = _run_exact_seed_summaries(
            exact_config,
            controls,
            seeds,
            density_grid,
            worker_count=requested_workers,
            progress=bar,
        )

    energy_values = np.asarray([summary.mixed_energy for summary in seed_summaries], dtype=float)
    density_integrals = np.asarray([summary.density_integral for summary in seed_summaries])
    abs_error = abs(float(np.mean(energy_values)) - exact_config.exact_energy_total)
    status = "passed" if abs_error <= args.energy_tolerance else "failed"
    case_id = f"N{args.n_particles}_a0_omega{args.omega:g}"
    payload = {
        "schema_version": "rn_block_exact_tg_trap_v1",
        "status": status,
        "benchmark_tier": "exact trapped TG RN-block DMC validation",
        "claim_boundary": "exact a=0 harmonic TG anchor; not a finite-rod trapped benchmark",
        "exact_solution": {
            "model": "zero-length hard rods in a harmonic trap",
            "formula": "E0 = N^2 * omega / sqrt(2)",
            "n_particles": exact_config.n_particles,
            "omega": exact_config.omega,
            "energy_total": exact_config.exact_energy_total,
        },
        "controls": controls_to_dict(controls),
        "energy_tolerance": args.energy_tolerance,
        "mixed_energy": float(np.mean(energy_values)),
        "mixed_energy_seed_stderr": _stderr(energy_values),
        "absolute_energy_error": abs_error,
        "relative_energy_error": abs_error / exact_config.exact_energy_total,
        "density_integral_mean": float(np.mean(density_integrals)),
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "parallel_workers_requested": requested_workers,
        "guide_batch_backend": ",".join(
            sorted({str(summary.metadata["guide_batch_backend"]) for summary in seed_summaries})
        ),
        "seed_summaries": [
            {
                "seed": seed,
                "mixed_energy": summary.mixed_energy,
                "absolute_energy_error": abs(
                    summary.mixed_energy - exact_config.exact_energy_total
                ),
                "density_integral": summary.density_integral,
                "lost_out_of_grid_sample_count": summary.lost_out_of_grid_sample_count,
                "killed_count": summary.metadata["killed_count"],
                "resample_count": summary.metadata["resample_count"],
                "ess_min": summary.metadata["ess_min"],
                "ess_mean": summary.metadata["ess_mean"],
                "guide_batch_backend": summary.metadata["guide_batch_backend"],
            }
            for seed, summary in zip(seeds, seed_summaries, strict=True)
        ],
    }
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root, ArtifactRoute("dmc", "rn_block", "exact_tg_trap")
        )
        write_rn_run_artifacts(
            output_dir,
            payload=payload,
            rows=[],
            run_name="rn_block_exact_tg_trap",
            config=rn_run_config(
                run_kind="rn_block_exact_tg_trap",
                cases=[case_id],
                seeds=seeds,
                controls=controls,
                parallel_workers=args.parallel_workers,
            ),
            command=sys.argv,
        )
    print(json.dumps(payload, indent=2))


def _run_exact_seed_summaries(
    exact_config: ExactTGTrapConfig,
    controls: RNRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    *,
    worker_count: int,
    progress: Any,
) -> tuple[list[RNBlockStreamingSummary], int]:
    return run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            _exact_seed_worker,
            exact_config,
            controls,
            seed,
            density_grid,
            progress_queue,
        ),
        run_serial_seed=lambda seed: _run_exact_seed(
            exact_config,
            controls,
            seed,
            density_grid,
            progress=progress,
        ),
    )


def _exact_seed_worker(
    exact_config: ExactTGTrapConfig,
    controls: RNRunControls,
    seed: int,
    density_grid: np.ndarray,
    progress_queue: Any | None = None,
) -> tuple[int, RNBlockStreamingSummary]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, _run_exact_seed(
            exact_config,
            controls,
            seed,
            density_grid,
            progress=worker_progress,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def _run_exact_seed(
    exact_config: ExactTGTrapConfig,
    controls: RNRunControls,
    seed: int,
    density_grid: np.ndarray,
    *,
    progress: Any | None = None,
) -> RNBlockStreamingSummary:
    system = OpenLineHardRodSystem(n_particles=exact_config.n_particles, rod_length=0.0)
    trap = HarmonicTrap(omega=exact_config.omega)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=exact_config.omega / np.sqrt(2.0),
    )
    rng = np.random.default_rng(seed)
    return run_rn_block_dmc_streaming(
        initial_walkers=initial_walkers(system, controls.walkers, rng),
        guide=guide,
        system=system,
        target_kernel=OrderedHarmonicMehlerKernel(system=system, trap=trap),
        proposal_kernel=HarmonicMehlerKernel(trap=trap),
        density_grid=density_grid,
        config=RNBlockDMCConfig(
            tau_block=controls.tau_block,
            rn_cadence_tau=controls.rn_cadence_tau,
            component_log_scales=(-0.02, 0.0, 0.02),
            component_probabilities=(0.25, 0.5, 0.25),
        ),
        rng=rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
    )


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _stderr(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


if __name__ == "__main__":
    main()
