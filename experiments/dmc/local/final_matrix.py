from __future__ import annotations

import argparse
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.final_matrix import (
    DEFAULT_CASES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLATEAU_EQUIVALENCE_CONFIDENCE_LEVEL,
    DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE,
    DEFAULT_SEEDS,
    FinalMatrixConfig,
    run_final_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed thesis trapped hard-rod DMC matrix by dispatching "
            "benchmark-packet cases with verified restart metadata."
        )
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
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
        default=0.10,
        help="Largest density-bin width in oscillator units after grid planning.",
    )
    parser.add_argument("--ess-resample-fraction", type=float, default=0.35)
    parser.add_argument("--pure-fw-block-size-steps", type=int, default=1)
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-min-source-ancestor-ess", type=float, default=50.0)
    parser.add_argument("--pure-fw-max-source-family-fraction", type=float, default=0.10)
    parser.add_argument(
        "--pure-fw-rms-plateau-relative-tolerance",
        type=float,
        default=DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE,
        help=(
            "Practical relative RMS-radius equivalence margin for the paired-seed "
            "forward-walking lag window."
        ),
    )
    parser.add_argument(
        "--pure-fw-plateau-equivalence-confidence-level",
        type=float,
        default=DEFAULT_PLATEAU_EQUIVALENCE_CONFIDENCE_LEVEL,
        help="Family-wise confidence level for the paired-lag equivalence bound.",
    )
    parser.add_argument("--pure-fw-density-plateau-window-lag-count", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=5)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--n10-a0p1-guide-validation-summary",
        type=Path,
        default=None,
        help=("Validated contact-guide summary required when N10_A0.1 is requested."),
    )
    parser.add_argument(
        "--n20-a0p1-guide-validation-summary",
        type=Path,
        default=None,
        help=("Validated contact-guide summary required when N20_A0.1 is requested."),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = vars(args)
    verbose_json = bool(options.pop("verbose_json"))
    try:
        result = run_final_matrix(
            FinalMatrixConfig(**options),
            repo_root=repo_root_from(Path(__file__)),
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"final_matrix: {exc}") from None
    print_run_summary(
        run="final_matrix",
        status=result.status,
        summary=result.summary,
        artifacts=result.artifacts,
        verbose_payload=result.verbose_payload,
        verbose_json=verbose_json,
    )


if __name__ == "__main__":
    main()
