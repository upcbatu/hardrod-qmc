from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from hrdmc.artifacts import repo_root_from
from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.workflows.dmc.rn_block import RNRunControls, make_grid, parse_case, parse_seeds

DEFAULT_CASES = "N10_A0.1,N10_A1,N10_A10,N20_A0.1,N20_A1,N20_A10"
DEFAULT_SEEDS = "7001,7002"
DEFAULT_LAGS = "0,400,800,1200,1600,2000,2800"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final trapped hard-rod DMC matrix by dispatching "
            "benchmark-packet cases with shared settings."
        )
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument(
        "--local-step-method",
        choices=("euler", "metropolis"),
        default="metropolis",
    )
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
    parser.add_argument("--store-every", type=int, default=10)
    parser.add_argument(
        "--grid-extent",
        type=float,
        default=35.0,
        help="Minimum half-width of the density grid in oscillator units.",
    )
    parser.add_argument(
        "--excluded-volume-margin",
        type=float,
        default=35.0,
        help=(
            "Extra half-width reserved beyond N*a/2 for finite-rod cases. "
            "The final LDA grid may grow further when the cloud requires it."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument(
        "--max-density-bin-width",
        type=float,
        default=0.20,
        help="Largest density-bin width in oscillator units after grid planning.",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument("--pure-fw-lags", default=DEFAULT_LAGS)
    parser.add_argument("--pure-fw-block-size-steps", type=int, default=1)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=20)
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/dmc/final_matrix/local_mala_dt0025"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    cases = _parse_cases(args.cases)
    seeds = parse_seeds(args.seeds)
    output_root = ensure_dir(args.output_root)

    records: list[dict[str, Any]] = []
    for case_id in cases:
        case_output_dir = output_root / case_id
        grid_plan = _case_grid_plan(args, case_id)
        record: dict[str, Any] = {
            "case": case_id,
            "output_dir": str(case_output_dir),
            "summary": str(case_output_dir / "summary.json"),
            "grid_plan": grid_plan,
        }
        summary_path = case_output_dir / "summary.json"
        if summary_path.exists() and not args.force:
            record["status"] = "skipped_existing_summary"
            record["grid_plan"] = _existing_grid_plan(summary_path, grid_plan)
            record["run_manifest"] = str(case_output_dir / "run_manifest.json")
            record["rerun_command"] = _benchmark_command(
                args,
                case_id,
                case_output_dir,
                grid_plan,
            )
            records.append(record)
            continue
        command = _benchmark_command(args, case_id, case_output_dir, grid_plan)
        record["command"] = command
        if args.dry_run:
            record["status"] = "planned"
            records.append(record)
            continue
        ensure_dir(case_output_dir)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=_subprocess_env(repo_root),
            check=False,
        )
        record["returncode"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        records.append(record)
        _write_matrix_manifest(output_root, args, cases, seeds, records)
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)

    manifest_path = _write_matrix_manifest(output_root, args, cases, seeds, records)
    print(json.dumps({"manifest": str(manifest_path), "rows": records}, indent=2))


def _parse_cases(value: str) -> list[str]:
    cases = [item.strip() for item in value.split(",") if item.strip()]
    if not cases:
        raise ValueError("at least one case is required")
    for case_id in cases:
        parse_case(case_id)
    return cases


def _case_grid_plan(args: argparse.Namespace, case_id: str) -> dict[str, float | int]:
    """Plan a finite-support density grid before dispatching one matrix row."""
    case = parse_case(case_id)
    minimum_extent = 0.5 * case.n_particles * case.rod_length
    requested_extent = max(args.grid_extent, minimum_extent + args.excluded_volume_margin)
    controls = RNRunControls(
        dt=args.dt,
        walkers=args.walkers,
        tau_block=args.tau,
        rn_cadence_tau=args.burn_tau + args.production_tau + args.dt,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=requested_extent,
        n_bins=args.n_bins,
        local_step_method=args.local_step_method,
    )
    grid = make_grid(controls, case)
    planned_extent = float(max(abs(grid[0]), abs(grid[-1])))
    planned_bins = max(
        args.n_bins,
        math.ceil((2.0 * planned_extent) / args.max_density_bin_width),
    )
    return {
        "minimum_excluded_volume_extent": minimum_extent,
        "requested_grid_extent": requested_extent,
        "grid_extent": planned_extent,
        "n_bins": planned_bins,
        "density_bin_width": (2.0 * planned_extent) / planned_bins,
    }


def _existing_grid_plan(
    summary_path: Path,
    planned_grid: dict[str, float | int],
) -> dict[str, float | int]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    controls = payload.get("controls", {})
    grid_extent = controls.get("grid_extent")
    n_bins = controls.get("n_bins")
    if not isinstance(grid_extent, (int, float)) or not isinstance(n_bins, int):
        return planned_grid
    density = payload.get("paper_values", {}).get("density", {})
    density_x = density.get("x") if isinstance(density, dict) else None
    if isinstance(density_x, list) and len(density_x) >= 2:
        actual_extent = float(max(abs(density_x[0]), abs(density_x[-1])))
        actual_bins = len(density_x)
        return {
            **planned_grid,
            "configured_grid_extent": float(grid_extent),
            "grid_extent": actual_extent,
            "n_bins": actual_bins,
            "density_bin_width": (2.0 * actual_extent) / actual_bins,
        }
    return {
        **planned_grid,
        "grid_extent": float(grid_extent),
        "n_bins": n_bins,
        "density_bin_width": (2.0 * float(grid_extent)) / n_bins,
    }


def _benchmark_command(
    args: argparse.Namespace,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
) -> list[str]:
    command = [
        sys.executable,
        "experiments/dmc/rn_block/benchmark_packet.py",
        "--case",
        case_id,
        "--seeds",
        args.seeds,
        "--dt",
        _format_number(args.dt),
        "--walkers",
        str(args.walkers),
        "--local-step-method",
        args.local_step_method,
        "--tau",
        _format_number(args.tau),
        "--burn-tau",
        _format_number(args.burn_tau),
        "--production-tau",
        _format_number(args.production_tau),
        "--store-every",
        str(args.store_every),
        "--grid-extent",
        _format_number(float(grid_plan["grid_extent"])),
        "--n-bins",
        str(grid_plan["n_bins"]),
        "--initialization-mode",
        args.initialization_mode,
        "--init-width-log-sigma",
        _format_number(args.init_width_log_sigma),
        "--breathing-preburn-steps",
        str(args.breathing_preburn_steps),
        "--breathing-preburn-log-step",
        _format_number(args.breathing_preburn_log_step),
        "--proposal-family",
        "harmonic-mehler",
        "--guide-family",
        "reduced-tg",
        "--target-family",
        "primitive",
        "--component-log-scales",
        "0.0",
        "--component-probabilities",
        "1.0",
        "--pure-fw-lags",
        args.pure_fw_lags,
        "--pure-fw-block-size-steps",
        str(args.pure_fw_block_size_steps),
        "--pure-fw-collection-stride-steps",
        str(args.pure_fw_collection_stride_steps),
        "--pure-fw-min-block-count",
        str(args.pure_fw_min_block_count),
        "--pure-fw-min-walker-weight-ess",
        _format_number(args.pure_fw_min_walker_weight_ess),
        "--parallel-workers",
        str(args.parallel_workers),
        "--plot-formats",
        args.plot_formats,
        "--output-dir",
        str(output_dir),
        "--disable-rn",
    ]
    if args.progress:
        command.append("--progress")
    return command


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _write_matrix_manifest(
    output_root: Path,
    args: argparse.Namespace,
    cases: list[str],
    seeds: list[int],
    records: list[dict[str, Any]],
) -> Path:
    path = output_root / "final_matrix_manifest.json"
    existing_records = _existing_matrix_records(path)
    discovered_records = _discover_completed_rows(output_root)
    merged_records = {
        record["case"]: record for record in [*existing_records, *discovered_records, *records]
    }
    write_json(
        path,
        {
            "schema_version": "hardrod_final_matrix_v2",
            "requested_cases": cases,
            "seeds": seeds,
            "invocation_settings": {
                "dt": args.dt,
                "walkers": args.walkers,
                "tau": args.tau,
                "burn_tau": args.burn_tau,
                "production_tau": args.production_tau,
                "store_every": args.store_every,
                "grid_extent": args.grid_extent,
                "excluded_volume_margin": args.excluded_volume_margin,
                "n_bins": args.n_bins,
                "max_density_bin_width": args.max_density_bin_width,
                "initialization_mode": args.initialization_mode,
                "init_width_log_sigma": args.init_width_log_sigma,
                "breathing_preburn_steps": args.breathing_preburn_steps,
                "breathing_preburn_log_step": args.breathing_preburn_log_step,
                "pure_fw_lags": args.pure_fw_lags,
                "pure_fw_block_size_steps": args.pure_fw_block_size_steps,
                "pure_fw_collection_stride_steps": args.pure_fw_collection_stride_steps,
                "pure_fw_min_block_count": args.pure_fw_min_block_count,
                "pure_fw_min_walker_weight_ess": args.pure_fw_min_walker_weight_ess,
                "parallel_workers": args.parallel_workers,
                "plot_formats": args.plot_formats,
                "calculation": "metropolis_corrected_drift_diffusion_dmc",
            },
            "rows": [merged_records[case_id] for case_id in sorted(merged_records)],
        },
    )
    return path


def _existing_matrix_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("rows", payload.get("records", []))
    return [record for record in records if isinstance(record, dict) and "case" in record]


def _discover_completed_rows(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("N*_A*/summary.json")):
        case_id = summary_path.parent.name
        try:
            parse_case(case_id)
        except ValueError:
            continue
        records.append(
            {
                "case": case_id,
                "output_dir": str(summary_path.parent),
                "summary": str(summary_path),
                "run_manifest": str(summary_path.parent / "run_manifest.json"),
                "status": "completed_existing",
                "grid_plan": _existing_grid_plan(summary_path, {}),
            }
        )
    return records


def _format_number(value: float) -> str:
    return f"{value:g}"


if __name__ == "__main__":
    main()
