from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_requested
from hrdmc.workflows.dmc.rn_block import (
    RNRunControls,
    controls_to_dict,
    parse_case,
    rn_progress_bar,
    rn_run_config,
    validate_streaming_against_raw,
    write_rn_run_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate RN-block streaming summaries.")
    parser.add_argument("--case", default="N4_a0.5_omega0.1")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=0.04)
    parser.add_argument("--production-tau", type=float, default=0.08)
    parser.add_argument("--store-every", type=int, default=5)
    parser.add_argument("--grid-extent", type=float, default=8.0)
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
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
    with rn_progress_bar(
        controls=controls,
        seed_count=1,
        label=f"RN validate {args.case}",
        enabled=progress_requested(args.progress),
        raw_validation=True,
    ) as bar:
        result = validate_streaming_against_raw(
            parse_case(args.case),
            controls,
            args.seed,
            progress=bar,
        )
    payload = {
        "schema_version": "rn_block_streaming_validation_v1",
        "status": "completed",
        "benchmark_tier": "RN-block DMC validation",
        "claim_boundary": "streaming equivalence validation only; not a physics benchmark",
        "controls": controls_to_dict(controls),
        "validation": result,
    }
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root, ArtifactRoute("dmc", "rn_block", "validate_streaming")
        )
        write_rn_run_artifacts(
            output_dir,
            payload=payload,
            rows=[],
            run_name="rn_block_streaming_validation",
            config=rn_run_config(
                run_kind="rn_block_streaming_validation",
                cases=[args.case],
                seeds=[args.seed],
                controls=controls,
                parallel_workers=1,
            ),
            command=sys.argv,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
