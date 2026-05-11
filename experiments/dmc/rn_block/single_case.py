from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_requested
from hrdmc.workflows.dmc.rn_block import (
    RN_SINGLE_CASE_SCHEMA_VERSION,
    RNRunControls,
    parse_case,
    parse_seeds,
    rn_progress_bar,
    rn_run_config,
    summarize_case,
    write_rn_run_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one RN-block DMC trapped hard-rod case.")
    parser.add_argument("--case", default="N4_a0.5_omega0.1")
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
        help="Seed workers; 0 chooses an automatic local default.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    out_dir = args.output_dir or artifact_dir(
        repo_root, ArtifactRoute("dmc", "rn_block", "single_case")
    )
    if args.resume and args.no_write:
        raise SystemExit("--resume requires artifact writes")
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label=f"RN single {args.case}",
        enabled=progress_requested(args.progress),
    ) as bar:
        case_summary = summarize_case(
            parse_case(args.case),
            controls,
            seeds,
            parallel_workers=args.parallel_workers,
            progress=bar,
            checkpoint_dir=None if args.no_write else out_dir / "seed_checkpoints",
            checkpoint_every_steps=None if args.no_write else args.checkpoint_every_steps,
            resume_seed_checkpoints=args.resume,
        )
    run_config = rn_run_config(
        run_kind="rn_block_single_case",
        cases=[args.case],
        seeds=seeds,
        controls=controls,
        parallel_workers=args.parallel_workers,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    payload = {
        "schema_version": RN_SINGLE_CASE_SCHEMA_VERSION,
        "status": "completed",
        "benchmark_tier": "RN-block DMC candidate case",
        "claim_boundary": "candidate streaming summary; verify controls before thesis figures",
        "run_config": run_config,
        "case": case_summary,
    }
    if not args.no_write:
        write_rn_run_artifacts(
            out_dir,
            payload=payload,
            rows=[case_summary],
            run_name="rn_block_single_case",
            config=run_config,
            command=sys.argv,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
