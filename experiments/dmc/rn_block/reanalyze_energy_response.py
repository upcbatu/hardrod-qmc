from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io.artifacts import build_run_provenance, write_json, write_run_manifest
from hrdmc.io.schema import to_jsonable
from hrdmc.workflows.dmc.energy_response import (
    reanalyze_trap_r2_energy_response,
    write_response_point_table,
)
from hrdmc.workflows.dmc.rn_block import parse_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Hellmann-Feynman trap R2/RMS reanalysis from RN-DMC summaries."
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--base-case", default="N8_a0.5_omega0.05")
    parser.add_argument("--polynomial-degree", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    base_case = parse_case(args.base_case)
    payload = reanalyze_trap_r2_energy_response(
        base_case=base_case,
        summary_paths=args.summaries,
        degree=args.polynomial_degree,
    )
    payload["plan"] = {
        "runner": "reanalyze_energy_response",
        "base_case": base_case.case_id,
        "summary_paths": [str(path) for path in args.summaries],
        "polynomial_degree": args.polynomial_degree,
        "claim_boundary": "offline HF response reanalysis only; no RN-DMC sampling performed",
    }
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root,
            ArtifactRoute("dmc", "rn_block", "energy_response_reanalysis"),
        )
        summary_path = output_dir / "summary.json"
        write_json(summary_path, payload)
        point_table = write_response_point_table(output_dir, payload["points"])
        write_run_manifest(
            output_dir,
            run_name="rn_block_energy_response_reanalysis",
            config=payload["plan"],
            artifacts=[summary_path, point_table],
            schema_version=str(payload["schema_version"]),
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
