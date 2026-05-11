from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_bar, progress_requested
from hrdmc.io.artifacts import build_run_provenance, ensure_dir, write_json, write_run_manifest
from hrdmc.workflows.dmc.rn_block import (
    RNRunControls,
    parse_case,
    parse_seeds,
)
from hrdmc.workflows.dmc.rn_block_stationarity import summarize_stationarity_case

DEFAULT_CASES = (
    "N4_a0.5_omega0.05,"
    "N4_a0.5_omega0.10,"
    "N8_a0.5_omega0.20,"
    "N8_a0.5_omega0.54204"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small RN-DMC blocking-aware timestep/population audit sweep."
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--seeds", default="301,302,303,304,305,306,307,308,309,310")
    parser.add_argument("--base-dt", type=float, default=0.00125)
    parser.add_argument("--base-walkers", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.005)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=20.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-traces", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    output_dir = args.output_dir or artifact_dir(
        repo_root,
        ArtifactRoute("dmc", "rn_block", "gate_audit_sweep"),
    )
    cases = [parse_case(item) for item in args.cases.split(",") if item.strip()]
    seeds = parse_seeds(args.seeds)
    dt_values = [args.base_dt, args.base_dt / 2.0, args.base_dt / 4.0]
    walker_values = [args.base_walkers, 2 * args.base_walkers]
    rows: list[dict[str, Any]] = []
    total_steps = sum(
        len(cases)
        * len(seeds)
        * (
            RNRunControls(
                dt=dt,
                walkers=walkers,
                tau_block=args.tau,
                rn_cadence_tau=args.rn_cadence,
                burn_tau=args.burn_tau,
                production_tau=args.production_tau,
                store_every=args.store_every,
                grid_extent=args.grid_extent,
                n_bins=args.n_bins,
            ).burn_in_steps
            + RNRunControls(
                dt=dt,
                walkers=walkers,
                tau_block=args.tau,
                rn_cadence_tau=args.rn_cadence,
                burn_tau=args.burn_tau,
                production_tau=args.production_tau,
                store_every=args.store_every,
                grid_extent=args.grid_extent,
                n_bins=args.n_bins,
            ).production_steps
        )
        for walkers in walker_values
        for dt in dt_values
    )
    with progress_bar(
        total=total_steps,
        label="RN gate audit sweep",
        enabled=progress_requested(args.progress),
    ) as bar:
        for case in cases:
            for walkers in walker_values:
                for dt in dt_values:
                    controls = RNRunControls(
                        dt=dt,
                        walkers=walkers,
                        tau_block=args.tau,
                        rn_cadence_tau=args.rn_cadence,
                        burn_tau=args.burn_tau,
                        production_tau=args.production_tau,
                        store_every=args.store_every,
                        grid_extent=args.grid_extent,
                        n_bins=args.n_bins,
                    )
                    trace_dir = (
                        None
                        if args.no_write or args.skip_traces
                        else output_dir / "trace_artifacts"
                    )
                    row = summarize_stationarity_case(
                        case,
                        controls,
                        seeds,
                        parallel_workers=args.parallel_workers,
                        progress=bar,
                        trace_output_dir=trace_dir,
                    )
                    row["dt"] = dt
                    row["walkers"] = walkers
                    rows.append(row)

    payload = {
        "schema_version": "rn_block_gate_audit_sweep_v1",
        "status": "completed",
        "classification": classify_sweep(rows),
        "benchmark_tier": "RN-DMC gate-audit sweep",
        "claim_boundary": (
            "statistical-control and timestep/population diagnostic only; "
            "not final benchmark by itself"
        ),
        "case_count": len(cases),
        "row_count": len(rows),
        "dt_values": dt_values,
        "walker_values": walker_values,
        "seeds": seeds,
        "fits": fit_by_case_and_population(rows),
        "rows": rows,
    }
    if not args.no_write:
        ensure_dir(output_dir)
        summary_path = output_dir / "rn_dmc_sweep_summary.json"
        table_path = output_dir / "rn_dmc_sweep_table.csv"
        write_json(summary_path, payload)
        write_sweep_table(table_path, rows)
        write_run_manifest(
            output_dir,
            run_name="rn_block_gate_audit_sweep",
            config={
                "cases": [case.case_id for case in cases],
                "seeds": seeds,
                "base_dt": args.base_dt,
                "base_walkers": args.base_walkers,
                "dt_values": dt_values,
                "walker_values": walker_values,
                "parallel_workers": args.parallel_workers,
            },
            artifacts=[summary_path, table_path],
            schema_version="rn_block_gate_audit_sweep_v1",
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(payload, indent=2))


def classify_sweep(rows: list[dict[str, Any]]) -> str:
    if all(row["case_gate"] for row in rows):
        return "RN_DMC_SWEEP_PASS_CANDIDATE"
    if any(row["final_classification"] == "HYGIENE_NO_GO" for row in rows):
        return "RN_DMC_SWEEP_HYGIENE_NO_GO"
    return "RN_DMC_SWEEP_STATISTICAL_WARNING"


def fit_by_case_and_population(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["case_id"], int(row["walkers"])) for row in rows})
    for case_id, walkers in keys:
        subset = [
            row
            for row in rows
            if row["case_id"] == case_id and int(row["walkers"]) == walkers
        ]
        subset = sorted(subset, key=lambda row: float(row["dt"]))
        dt = np.asarray([float(row["dt"]) for row in subset], dtype=float)
        energy = np.asarray([float(row["mixed_energy"]) for row in subset], dtype=float)
        if dt.size < 2:
            continue
        linear = np.polyfit(dt, energy, deg=1)
        fit = {
            "case_id": case_id,
            "walkers": walkers,
            "linear_E0": float(linear[-1]),
            "linear_slope": float(linear[-2]),
        }
        if dt.size >= 3:
            quadratic = np.polyfit(dt, energy, deg=2)
            fit["quadratic_E0"] = float(quadratic[-1])
            fit["quadratic_c1"] = float(quadratic[-2])
            fit["quadratic_c2"] = float(quadratic[-3])
            qdmc_like = np.polyfit(dt * dt, energy, deg=1)
            fit["second_order_E0"] = float(qdmc_like[-1])
            fit["second_order_c2"] = float(qdmc_like[-2])
        out.append(fit)
    return out


def write_sweep_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "dt",
        "walkers",
        "case_gate",
        "final_classification",
        "hygiene_gate",
        "mixed_energy",
        "mixed_energy_conservative_stderr",
        "rms_radius",
        "rms_radius_conservative_stderr",
        "density_relative_l2",
        "blocking_plateau_energy",
        "blocked_zscore_max_energy",
        "rhat_energy",
        "neff_energy",
        "ess_fraction_min",
        "log_weight_span_max",
        "rn_weight_status",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
