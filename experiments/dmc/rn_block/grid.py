from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_requested
from hrdmc.workflows.dmc.rn_block import (
    RN_GRID_SCHEMA_VERSION,
    RNRunControls,
    load_completed_case_rows,
    parse_case,
    parse_seeds,
    rn_progress_bar,
    rn_run_config,
    summarize_case,
    write_case_checkpoint,
    write_rn_run_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an RN-block DMC trapped hard-rod grid.")
    parser.add_argument(
        "--cases",
        default="N4_a0.5_omega0.05,N4_a0.5_omega0.1,N4_a0.5_omega0.2,"
        "N8_a0.5_omega0.05,N8_a0.5_omega0.1,N8_a0.5_omega0.2",
    )
    parser.add_argument("--seeds", default="301,302,303,304")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.005)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=20.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=5000,
        help="Write per-seed engine checkpoints every N steps when artifacts are enabled.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Seed workers per case; 0 chooses an automatic local default.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json in the output directory after completed cases.",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
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
    seeds = parse_seeds(args.seeds)
    case_ids = [case_id for case_id in args.cases.split(",") if case_id.strip()]
    if args.resume and args.no_write:
        raise SystemExit("--resume requires artifact writes")
    out_dir = args.output_dir or artifact_dir(repo_root, ArtifactRoute("dmc", "rn_block", "grid"))
    config = rn_run_config(
        run_kind="rn_block_grid",
        cases=case_ids,
        seeds=seeds,
        controls=controls,
        parallel_workers=args.parallel_workers,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    checkpoint_path = out_dir / "checkpoint.json"
    rows_by_case = (
        load_completed_case_rows(checkpoint_path, expected_config=config)
        if args.resume and not args.no_write
        else {}
    )
    if not args.no_write and not rows_by_case:
        write_case_checkpoint(
            checkpoint_path,
            status="running",
            config=config,
            completed_cases=[],
            pending_cases=case_ids,
        )
    remaining_case_ids = [case_id for case_id in case_ids if case_id not in rows_by_case]
    if remaining_case_ids:
        with rn_progress_bar(
            controls=controls,
            seed_count=len(seeds) * len(remaining_case_ids),
            label="RN grid",
            enabled=progress_requested(args.progress),
        ) as bar:
            for case_id in remaining_case_ids:
                rows_by_case[case_id] = summarize_case(
                    parse_case(case_id),
                    controls,
                    seeds,
                    parallel_workers=args.parallel_workers,
                    progress=bar,
                    checkpoint_dir=None if args.no_write else out_dir / "seed_checkpoints",
                    checkpoint_every_steps=(
                        None if args.no_write else args.checkpoint_every_steps
                    ),
                    resume_seed_checkpoints=args.resume,
                )
                if not args.no_write:
                    write_case_checkpoint(
                        checkpoint_path,
                        status="running",
                        config=config,
                        completed_cases=[
                            rows_by_case[item] for item in case_ids if item in rows_by_case
                        ],
                        pending_cases=[
                            item for item in case_ids if item not in rows_by_case
                        ],
                    )
    rows = [rows_by_case[case_id] for case_id in case_ids]
    payload = {
        "schema_version": RN_GRID_SCHEMA_VERSION,
        "status": "completed",
        "benchmark_tier": "RN-block DMC candidate grid",
        "claim_boundary": "candidate streaming grid; verify controls before thesis figures",
        "run_config": config,
        "case_count": len(rows),
        "cases": rows,
    }
    if not args.no_write:
        write_case_checkpoint(
            checkpoint_path,
            status="completed",
            config=config,
            completed_cases=rows,
            pending_cases=[],
        )
        write_rn_run_artifacts(
            out_dir,
            payload=payload,
            rows=rows,
            run_name="rn_block_grid",
            config=config,
            command=sys.argv,
            extra_artifacts=[checkpoint_path],
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
