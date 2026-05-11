from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io.artifacts import build_run_provenance, ensure_dir, write_json, write_run_manifest
from hrdmc.monte_carlo.dmc.rn_block import (
    RNBlockDMCConfig,
    run_rn_block_dmc,
    run_rn_block_dmc_streaming,
)
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
    OpenHardRodTrapPrimitiveKernel,
    OpenLineHardRodSystem,
)
from hrdmc.wavefunctions import ReducedTGHardRodGuide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small RN-block DMC smoke test.")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--tau-block", type=float, default=0.01)
    parser.add_argument("--rn-cadence-tau", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--production", type=int, default=40)
    parser.add_argument("--store-every", type=int, default=5)
    parser.add_argument("--summary-mode", choices=("streaming", "raw"), default="streaming")
    parser.add_argument("--grid-extent", type=float, default=8.0)
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))

    system = OpenLineHardRodSystem(n_particles=args.n_particles, rod_length=args.rod_length)
    trap = HarmonicTrap(omega=args.omega)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=args.omega / np.sqrt(2.0),
    )
    target_kernel = OpenHardRodTrapPrimitiveKernel(system=system, trap=trap)
    proposal_kernel = HarmonicMehlerKernel(trap=trap)
    config = RNBlockDMCConfig(
        tau_block=args.tau_block,
        rn_cadence_tau=args.rn_cadence_tau,
        component_log_scales=(-0.02, 0.0, 0.02),
        component_probabilities=(0.25, 0.5, 0.25),
    )
    rng = np.random.default_rng(args.seed)
    initial_walkers = _initial_walkers(system, args.walkers, rng)

    grid = np.linspace(-args.grid_extent, args.grid_extent, args.n_bins)
    if args.summary_mode == "streaming":
        result = run_rn_block_dmc_streaming(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            target_kernel=target_kernel,
            proposal_kernel=proposal_kernel,
            density_grid=grid,
            config=config,
            rng=rng,
            dt=args.dt,
            burn_in_steps=args.burn_in,
            production_steps=args.production,
            store_every=args.store_every,
        )
        observable_summary = {
            "summary_mode": "streaming",
            "n_samples": result.sample_count,
            "stored_batch_count": result.stored_batch_count,
            "mixed_energy": result.mixed_energy,
            "rms_radius": result.rms_radius,
            "density_integral": result.density_integral,
            "lost_out_of_grid_sample_count": result.lost_out_of_grid_sample_count,
            "weight_sum": None,
        }
        run_metadata = result.metadata
        valid_snapshot_fraction = 1.0
    else:
        result = run_rn_block_dmc(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            target_kernel=target_kernel,
            proposal_kernel=proposal_kernel,
            config=config,
            rng=rng,
            dt=args.dt,
            burn_in_steps=args.burn_in,
            production_steps=args.production,
            store_every=args.store_every,
        )
        valid_mask = np.array(
            [system.is_valid(snapshot) for snapshot in result.snapshots],
            dtype=bool,
        )
        observable_summary = {
            "summary_mode": "raw",
            "n_samples": int(result.snapshots.shape[0]),
            "stored_batch_count": int(result.metadata["stored_batch_count"]),
            "mixed_energy": float(np.average(result.local_energies, weights=result.weights)),
            "rms_radius": None,
            "density_integral": None,
            "lost_out_of_grid_sample_count": None,
            "weight_sum": float(np.sum(result.weights)),
        }
        run_metadata = result.metadata
        valid_snapshot_fraction = float(np.mean(valid_mask))

    rn_block_config = {
        "seed": args.seed,
        "summary_mode": args.summary_mode,
        "dt": args.dt,
        "tau_block": config.tau_block,
        "rn_cadence_tau": config.rn_cadence_tau,
        "walkers": args.walkers,
        "burn_in_steps": args.burn_in,
        "production_steps": args.production,
        "store_every": args.store_every,
        "rn_event_count": run_metadata["rn_event_count"],
        "local_step_count": run_metadata["local_step_count"],
        "killed_count": run_metadata["killed_count"],
    }
    schema_version = "rn_block_smoke_v1"
    summary = {
        "schema_version": schema_version,
        "status": "completed",
        "benchmark_tier": "RN-block DMC smoke",
        "claim_boundary": "API smoke only; not a benchmark run",
        "system": {
            "geometry": "open_line",
            "n_particles": system.n_particles,
            "rod_length": system.rod_length,
        },
        "trap": {
            "type": "harmonic",
            "omega": trap.omega,
            "center": trap.center,
        },
        "guide": {
            "type": "ReducedTGHardRodGuide",
            "alpha": guide.alpha,
            "pair_power": guide.pair_power,
        },
        "rn_block": rn_block_config,
        "valid_snapshot_fraction": valid_snapshot_fraction,
        **observable_summary,
    }

    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or artifact_dir(
            repo_root, ArtifactRoute("dmc", "rn_block", "smoke")
        ))
        summary_path = out_dir / "summary.json"
        write_json(summary_path, summary)
        write_run_manifest(
            out_dir,
            run_name="rn_block_smoke",
            config=rn_block_config,
            artifacts=[summary_path],
            schema_version=schema_version,
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(summary, indent=2))


def _initial_walkers(
    system: OpenLineHardRodSystem,
    walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if walkers <= 0:
        raise ValueError("walkers must be positive")
    return np.vstack(
        [
            system.initial_lattice(
                spacing=max(1.25, 2.5 * system.rod_length),
                jitter=0.05,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for _ in range(walkers)
        ]
    )


if __name__ == "__main__":
    main()
