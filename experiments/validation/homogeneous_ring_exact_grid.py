from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.systems import excluded_length
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import (
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
)
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial


@dataclass(frozen=True)
class HomogeneousExactCase:
    n_particles: int
    packing_fraction: float

    @property
    def case_id(self) -> str:
        return f"N{self.n_particles}_eta{self.packing_fraction:g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate homogeneous finite-a hard-rod ring exact energies."
    )
    parser.add_argument("--n-values", default="4,8,16,32,64")
    parser.add_argument("--eta-values", default="0.05,0.1,0.3,0.5,0.7")
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--samples-per-case", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cases = [
        HomogeneousExactCase(n_particles=n_particles, packing_fraction=eta)
        for n_particles in _parse_ints(args.n_values)
        for eta in _parse_floats(args.eta_values)
    ]
    rows = [
        run_case(
            case,
            rod_length=args.rod_length,
            samples_per_case=args.samples_per_case,
            seed=args.seed + index,
            tolerance=args.tolerance,
        )
        for index, case in enumerate(cases)
    ]
    max_abs_error = max(float(row["max_energy_per_particle_abs_error"]) for row in rows)
    max_relative_error = max(float(row["max_energy_per_particle_relative_error"]) for row in rows)
    passed = all(bool(row["passed"]) for row in rows)
    payload = {
        "status": "passed" if passed else "failed",
        "benchmark_tier": "homogeneous finite-a hard-rod exact ring validation",
        "claim_boundary": (
            "exact homogeneous ring local-energy anchor; not a trapped benchmark "
            "and not an LDA validation"
        ),
        "source_basis": "Mazzanti2008HardRods finite-N reduced-length mapping",
        "formula": "E_N/N = pi^2 * (N^2 - 1) / (3 * (L - N*a)^2)",
        "units": "hbar^2/(2m)=1",
        "rod_length": args.rod_length,
        "samples_per_case": args.samples_per_case,
        "tolerance_energy_per_particle_abs": args.tolerance,
        "max_energy_per_particle_abs_error": max_abs_error,
        "max_energy_per_particle_relative_error": max_relative_error,
        "case_count": len(rows),
        "cases": rows,
    }
    if not args.no_write:
        output_dir = ensure_dir(
            args.output_dir or repo_root / "results" / "homogeneous_ring_exact_grid"
        )
        write_json(output_dir / "summary.json", payload)
        write_case_table(output_dir / "case_table.csv", rows)
        if not args.skip_plots:
            payload["plots"] = write_plots(output_dir, rows)
            write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def run_case(
    case: HomogeneousExactCase,
    *,
    rod_length: float,
    samples_per_case: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    if samples_per_case <= 0:
        raise ValueError("samples_per_case must be positive")
    if not 0.0 < case.packing_fraction < 1.0:
        raise ValueError("packing fractions must satisfy 0 < eta < 1")
    density = case.packing_fraction / rod_length
    length = case.n_particles / density
    system = HardRodSystem(
        n_particles=case.n_particles,
        length=length,
        rod_length=rod_length,
    )
    trial = HardRodJastrowTrial(system=system, power=1.0, nearest_neighbor_only=False)
    exact_per_particle = hard_rod_finite_ring_energy_per_particle(
        n_particles=system.n_particles,
        length=system.length,
        rod_length=system.rod_length,
    )
    energies = np.asarray(
        [
            trial.local_kinetic_energy(positions)
            for positions in sample_valid_ring_configurations(
                system,
                samples_per_case=samples_per_case,
                seed=seed,
            )
        ],
        dtype=float,
    )
    per_particle = energies / system.n_particles
    abs_errors = np.abs(per_particle - exact_per_particle)
    max_abs_error = float(np.max(abs_errors))
    relative_denominator = max(1.0, abs(exact_per_particle))
    max_relative_error = float(max_abs_error / relative_denominator)
    return {
        "passed": bool(max_abs_error <= tolerance),
        "case_id": case.case_id,
        "n_particles": system.n_particles,
        "rod_length": system.rod_length,
        "length": system.length,
        "density": system.density,
        "packing_fraction": system.packing_fraction,
        "excluded_length": excluded_length(
            system.n_particles,
            system.length,
            system.rod_length,
        ),
        "samples": samples_per_case,
        "energy_total_mean": float(np.mean(energies)),
        "energy_total_min": float(np.min(energies)),
        "energy_total_max": float(np.max(energies)),
        "energy_total_exact_finite_N": exact_per_particle * system.n_particles,
        "energy_per_particle_mean": float(np.mean(per_particle)),
        "energy_per_particle_exact_finite_N": exact_per_particle,
        "energy_per_particle_thermodynamic": hard_rod_energy_per_particle(
            system.density,
            system.rod_length,
        ),
        "max_energy_per_particle_abs_error": max_abs_error,
        "max_energy_per_particle_relative_error": max_relative_error,
    }


def sample_valid_ring_configurations(
    system: HardRodSystem,
    *,
    samples_per_case: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    free_gap = system.length / system.n_particles - system.rod_length
    jitter = 0.08 * free_gap
    samples: list[np.ndarray] = []
    for _ in range(samples_per_case):
        positions = system.initial_lattice(
            jitter=jitter,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        positions = system.wrap(positions + rng.uniform(0.0, system.length))
        if not system.is_valid(positions):
            raise RuntimeError("generated invalid homogeneous ring configuration")
        samples.append(positions)
    return samples


def write_case_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "n_particles",
        "packing_fraction",
        "length",
        "excluded_length",
        "samples",
        "energy_per_particle_mean",
        "energy_per_particle_exact_finite_N",
        "energy_per_particle_thermodynamic",
        "max_energy_per_particle_abs_error",
        "max_energy_per_particle_relative_error",
        "passed",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plt = load_pyplot(output_dir)
    plot_dir = ensure_dir(output_dir / "plots")
    paths = [
        plot_energy_error_heatmap(plt, rows, plot_dir / "energy_error_heatmap.png"),
        plot_finite_vs_thermodynamic_energy(
            plt,
            rows,
            plot_dir / "finite_vs_thermodynamic_energy.png",
        ),
        plot_relative_error_scatter(plt, rows, plot_dir / "relative_error_by_case.png"),
    ]
    return [str(path.relative_to(output_dir)) for path in paths]


def load_pyplot(output_dir: Path):
    scratch_dir = ensure_dir(output_dir / "mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(scratch_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_energy_error_heatmap(plt, rows: list[dict[str, Any]], output_path: Path) -> Path:
    n_values = sorted({int(row["n_particles"]) for row in rows})
    eta_values = sorted({float(row["packing_fraction"]) for row in rows})
    heatmap = np.full((len(n_values), len(eta_values)), np.nan)
    for row in rows:
        n_index = n_values.index(int(row["n_particles"]))
        eta_index = eta_values.index(float(row["packing_fraction"]))
        heatmap[n_index, eta_index] = np.log10(
            max(float(row["max_energy_per_particle_abs_error"]), 1e-18)
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    image = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(eta_values)), [f"{eta:g}" for eta in eta_values])
    ax.set_yticks(np.arange(len(n_values)), [str(n) for n in n_values])
    ax.set_xlabel("packing fraction eta = rho a")
    ax.set_ylabel("N")
    ax.set_title("Homogeneous exact-ring energy error")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10 max |E/N - exact|")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_finite_vs_thermodynamic_energy(plt, rows: list[dict[str, Any]], output_path: Path) -> Path:
    n_values = sorted({int(row["n_particles"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for n_particles in n_values:
        subset = sorted(
            [row for row in rows if int(row["n_particles"]) == n_particles],
            key=lambda row: float(row["packing_fraction"]),
        )
        eta = np.asarray([float(row["packing_fraction"]) for row in subset])
        finite = np.asarray([float(row["energy_per_particle_exact_finite_N"]) for row in subset])
        ax.plot(eta, finite, marker="o", linewidth=1.4, label=f"N={n_particles}")

    thermo_rows = sorted(
        {float(row["packing_fraction"]): float(row["energy_per_particle_thermodynamic"])
         for row in rows}.items()
    )
    ax.plot(
        [item[0] for item in thermo_rows],
        [item[1] for item in thermo_rows],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="thermodynamic EOS",
    )
    ax.set_xlabel("packing fraction eta = rho a")
    ax.set_ylabel("energy per particle")
    ax.set_title("Finite-N exact ring energy vs thermodynamic EOS")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_relative_error_scatter(plt, rows: list[dict[str, Any]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = np.asarray([float(row["packing_fraction"]) for row in rows])
    y = np.asarray([int(row["n_particles"]) for row in rows])
    errors = np.asarray(
        [max(float(row["max_energy_per_particle_relative_error"]), 1e-18) for row in rows]
    )
    scatter = ax.scatter(x, y, c=np.log10(errors), s=85, cmap="plasma", edgecolor="black")
    ax.set_xlabel("packing fraction eta = rho a")
    ax.set_ylabel("N")
    ax.set_yscale("log", base=2)
    ax.set_yticks(sorted(set(y)), [str(int(value)) for value in sorted(set(y))])
    ax.set_title("Relative exact-energy error by case")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10 relative error")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _parse_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one float value is required")
    return values


if __name__ == "__main__":
    main()
