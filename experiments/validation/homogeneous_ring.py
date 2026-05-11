from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.estimators import estimate_local_energy
from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.monte_carlo.vmc import MetropolisVMC
from hrdmc.systems import excluded_length
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import (
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
)
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial


@dataclass(frozen=True)
class ValidationCase:
    n_particles: int
    packing_fraction: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the homogeneous hard-rod ring validation benchmark."
    )
    parser.add_argument("--steps", type=int, default=2_500, help="Metropolis steps per case")
    parser.add_argument("--burn-in", type=int, default=500, help="Burn-in steps per case")
    parser.add_argument("--thinning", type=int, default=10, help="Snapshot thinning interval")
    parser.add_argument("--seed", type=int, default=1729, help="Base RNG seed")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute per-particle energy tolerance for pass/fail",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to results/homogeneous_validation",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run and print the benchmark without writing summary.json",
    )
    return parser


def default_cases() -> list[ValidationCase]:
    return [
        ValidationCase(n_particles=n_particles, packing_fraction=eta)
        for n_particles in (4, 8, 16)
        for eta in (0.10, 0.30, 0.50)
    ]


def run_case(
    case: ValidationCase,
    *,
    rod_length: float,
    n_steps: int,
    burn_in: int,
    thinning: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    density = case.packing_fraction / rod_length
    length = case.n_particles / density
    system = HardRodSystem(
        n_particles=case.n_particles,
        length=length,
        rod_length=rod_length,
    )
    trial = HardRodJastrowTrial(system=system, power=1.0, nearest_neighbor_only=False)
    free_gap = length / case.n_particles - rod_length
    step_size = 0.25 * free_gap
    sampler = MetropolisVMC(system=system, trial=trial, step_size=step_size, seed=seed)
    result = sampler.run(n_steps=n_steps, burn_in=burn_in, thinning=thinning)
    energy = estimate_local_energy(result.snapshots, trial)

    exact_per_particle = hard_rod_finite_ring_energy_per_particle(
        n_particles=system.n_particles,
        length=system.length,
        rod_length=system.rod_length,
    )
    exact_total = exact_per_particle * system.n_particles
    estimated_per_particle = energy.mean / system.n_particles
    energy_error = estimated_per_particle - exact_per_particle
    valid_fraction = float(np.mean([system.is_valid(x) for x in result.snapshots]))
    passed = (
        np.isfinite(energy.mean)
        and abs(energy_error) <= tolerance
        and valid_fraction == 1.0
        and 0.0 < result.acceptance_rate < 1.0
    )

    return {
        "passed": bool(passed),
        "n_particles": system.n_particles,
        "length": system.length,
        "density": system.density,
        "rod_length": system.rod_length,
        "packing_fraction": system.packing_fraction,
        "excluded_length": excluded_length(
            system.n_particles,
            system.length,
            system.rod_length,
        ),
        "step_size": step_size,
        "n_snapshots": int(result.snapshots.shape[0]),
        "valid_snapshot_fraction": valid_fraction,
        "acceptance_rate": result.acceptance_rate,
        "cpu_seconds": result.cpu_seconds,
        "energy_total_estimate": energy.mean,
        "energy_total_stderr": energy.stderr,
        "energy_total_exact_finite_N": exact_total,
        "energy_per_particle_estimate": estimated_per_particle,
        "energy_per_particle_exact_finite_N": exact_per_particle,
        "energy_per_particle_thermodynamic": hard_rod_energy_per_particle(
            system.density,
            system.rod_length,
        ),
        "energy_per_particle_error": energy_error,
        "energy_per_particle_abs_error": abs(energy_error),
        "vmc": result.metadata,
    }


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    cases = [
        run_case(
            case,
            rod_length=0.5,
            n_steps=args.steps,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=args.seed + index,
            tolerance=args.tolerance,
        )
        for index, case in enumerate(default_cases())
    ]
    max_error = max(float(case["energy_per_particle_abs_error"]) for case in cases)
    passed = all(bool(case["passed"]) for case in cases)
    summary = {
        "status": "passed" if passed else "failed",
        "benchmark": "homogeneous hard-rod ring exact-wavefunction validation",
        "benchmark_tier": "exact_trial_local_energy",
        "units": "hbar^2/(2m)=1",
        "tolerance_energy_per_particle_abs": args.tolerance,
        "max_energy_per_particle_abs_error": max_error,
        "case_count": len(cases),
        "cases": cases,
    }
    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or repo_root / "results" / "homogeneous_validation")
        write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
