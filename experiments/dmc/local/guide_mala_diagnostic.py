from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import file_sha256, repo_root_from
from hrdmc.io import print_run_summary, progress_bar, progress_requested
from hrdmc.workflows.dmc.guide_mala_diagnostic import (
    GuideMALADiagnosticControls,
    run_guide_mala_diagnostic,
    write_guide_mala_diagnostic_outputs,
)
from hrdmc.workflows.dmc.guide_parameter_optimization import (
    load_contact_optimization_candidate,
)
from hrdmc.workflows.dmc.trapped import parse_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run branching-free guide-squared MALA from one scaled reduced-gap start.")
    )
    parser.add_argument("--case", default="N20_A1")
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument("--dt", type=float, default=0.000625)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--duration-tau", type=float, default=60.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--start-scale", type=float, required=True)
    parser.add_argument("--relative-alpha", type=float, default=None)
    parser.add_argument(
        "--guide-family",
        choices=("reduced-tg", "contact-corrected-reduced-tg"),
        default="reduced-tg",
    )
    parser.add_argument("--contact-beta", type=float, default=None)
    parser.add_argument(
        "--guide-optimization-summary",
        type=Path,
        default=None,
        help=(
            "Load candidate relative_alpha and contact_beta values for this "
            "branching-free calibration run."
        ),
    )
    parser.add_argument("--grid-extent", type=float, default=48.512)
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument("--tail-tau", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root_from(Path(__file__))
    case = parse_case(args.case)
    relative_alpha = args.relative_alpha
    contact_beta = args.contact_beta
    guide_family = args.guide_family
    guide_parameter_source: dict[str, object] = {"kind": "explicit"}
    if args.guide_optimization_summary is not None:
        if relative_alpha is not None or contact_beta is not None:
            raise ValueError(
                "do not combine explicit guide parameters with --guide-optimization-summary"
            )
        relative_alpha, contact_beta = load_contact_optimization_candidate(
            args.guide_optimization_summary,
            case=case,
        )
        guide_family = "contact-corrected-reduced-tg"
        candidate_path = args.guide_optimization_summary.resolve()
        candidate_manifest = candidate_path.parent / "run_manifest.json"
        guide_parameter_source = {
            "kind": "optimization_candidate",
            "summary_path": str(candidate_path),
            "summary_sha256": file_sha256(candidate_path),
            "manifest_sha256": file_sha256(candidate_manifest),
        }
    controls = GuideMALADiagnosticControls(
        dt=args.dt,
        walkers=args.walkers,
        duration_tau=args.duration_tau,
        store_every=args.store_every,
        start_scale=args.start_scale,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        relative_alpha=relative_alpha,
        guide_family=guide_family,
        contact_beta=contact_beta,
        tail_tau=args.tail_tau,
    )
    with progress_bar(
        total=controls.steps,
        label=(f"Guide MALA {case.case_id} dt={controls.dt:g} scale={controls.start_scale:g}"),
        enabled=progress_requested(args.progress),
    ) as bar:
        summary, trace = run_guide_mala_diagnostic(
            case,
            controls,
            args.seed,
            guide_parameter_source=guide_parameter_source,
            progress=bar,
        )
    artifacts = write_guide_mala_diagnostic_outputs(
        args.output_dir,
        summary=summary,
        trace=trace,
        command=sys.argv,
    )
    print_run_summary(
        run="guide_mala_diagnostic",
        status=str(summary.get("status", "completed")),
        summary={
            "case": case.case_id,
            "seed": args.seed,
            "guide_family": guide_family,
            "acceptance_fraction": summary["tail_means"].get("acceptance_fraction"),
            "local_energy_mad": summary["tail_means"].get("local_energy_mad"),
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=summary,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
