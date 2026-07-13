from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import (
    ArtifactRoute,
    artifact_dir,
    repo_root_from,
)
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.energy_response import (
    run_energy_response_reanalysis,
)
from hrdmc.workflows.dmc.trapped import parse_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a five-point DMC trap-coupling ladder for R2 and RMS."
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--base-case", default="N8_A0.1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    base_case = parse_case(args.base_case)
    output_dir = None
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root,
            ArtifactRoute("dmc", "local", "energy_response_reanalysis"),
        )
    payload = run_energy_response_reanalysis(
        base_case=base_case,
        summary_paths=args.summaries,
        output_dir=output_dir,
        command=sys.argv,
        write=not args.no_write,
    )
    estimator = payload.get("energy_response", {})
    print_run_summary(
        run="reanalyze_energy_response",
        status=str(payload.get("status", payload.get("point_validity_status", "completed"))),
        summary={
            "case": base_case.case_id,
            "point_count": payload.get("point_count"),
            "seed_count": payload.get("seed_count"),
            "r2": estimator.get("pure_r2"),
            "rms_radius": estimator.get("rms_radius"),
        },
        artifacts=payload["artifacts"],
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
