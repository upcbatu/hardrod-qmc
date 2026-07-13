from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.guide_validation import (
    validate_contact_guide_calibrations,
    write_contact_guide_validation_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed contact-guide parameters against independent compact "
            "and expanded guide-squared MALA calibration artifacts."
        )
    )
    parser.add_argument("--compact-summary", type=Path, required=True)
    parser.add_argument("--expanded-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root_from(Path(__file__))
    payload = validate_contact_guide_calibrations(
        compact_summary=args.compact_summary,
        expanded_summary=args.expanded_summary,
    )
    artifacts = write_contact_guide_validation_output(
        args.output_dir,
        payload=payload,
        command=sys.argv,
    )
    print_run_summary(
        run="validate_contact_guide",
        status=str(payload["status"]),
        summary={
            "case": payload.get("case_id"),
            "failed_checks": payload.get("failed_checks", []),
            "validated_parameters": payload.get("validated_parameters"),
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )
    if payload["status"] != "validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
