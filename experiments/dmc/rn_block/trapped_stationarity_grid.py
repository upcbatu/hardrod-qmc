from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import progress_requested
from hrdmc.io.artifacts import build_run_provenance, ensure_dir, write_run_manifest
from hrdmc.io.schema import to_jsonable
from hrdmc.workflows.dmc.rn_block import (
    DEFAULT_COMPONENT_LOG_SCALES,
    DEFAULT_COMPONENT_PROBABILITIES,
    DEFAULT_RN_GUIDE_FAMILY,
    DEFAULT_RN_PROPOSAL_FAMILY,
    DEFAULT_RN_TARGET_FAMILY,
    RN_GUIDE_FAMILIES,
    RN_PROPOSAL_FAMILIES,
    RN_TARGET_FAMILIES,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
    parse_case,
    parse_seeds,
    rn_progress_bar,
    write_summary,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls
from hrdmc.workflows.dmc.rn_block_stationarity import classify_grid, summarize_stationarity_case
from hrdmc.workflows.dmc.rn_block_stationarity_outputs import write_case_table, write_plots

DEFAULT_CASES = "N4_a0.5_omega0.05,N8_a0.5_omega0.05,N8_a0.5_omega0.2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trapped RN-DMC stationarity diagnostic grid."
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
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
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-no-go-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument(
        "--proposal-family",
        choices=RN_PROPOSAL_FAMILIES,
        default=DEFAULT_RN_PROPOSAL_FAMILY,
        help="RN base proposal family; RN collective scaling still applies.",
    )
    parser.add_argument(
        "--guide-family",
        choices=RN_GUIDE_FAMILIES,
        default=DEFAULT_RN_GUIDE_FAMILY,
        help=(
            "DMC guide family. auto uses gap-h-corrected guide with gap-h-transform "
            "proposal and reduced-tg otherwise."
        ),
    )
    parser.add_argument(
        "--target-family",
        choices=RN_TARGET_FAMILIES,
        default=DEFAULT_RN_TARGET_FAMILY,
        help="RN target kernel family.",
    )
    parser.add_argument(
        "--component-log-scales",
        default=",".join(f"{value:g}" for value in DEFAULT_COMPONENT_LOG_SCALES),
        help="Comma-separated RN collective log-scale mixture components.",
    )
    parser.add_argument(
        "--component-probabilities",
        default=",".join(f"{value:g}" for value in DEFAULT_COMPONENT_PROBABILITIES),
        help="Comma-separated RN collective mixture probabilities.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Seed workers per case; 0 chooses an automatic local default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to results/dmc/rn_block/trapped_stationarity_grid",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
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
    cases = [parse_case(item) for item in args.cases.split(",") if item.strip()]
    component_log_scales = _parse_float_tuple(args.component_log_scales)
    component_probabilities = _parse_float_tuple(args.component_probabilities)
    initialization = RNInitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    proposal = RNCollectiveProposalControls(
        component_log_scales=component_log_scales,
        component_probabilities=component_probabilities,
    )
    output_dir = args.output_dir or artifact_dir(
        repo_root, ArtifactRoute("dmc", "rn_block", "trapped_stationarity_grid")
    )
    trace_output_dir = None if args.no_write else output_dir
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds) * len(cases),
        label="RN stationarity grid",
        enabled=progress_requested(args.progress),
    ) as bar:
        rows = [
            summarize_stationarity_case(
                case,
                controls,
                seeds,
                parallel_workers=args.parallel_workers,
                progress=bar,
                trace_output_dir=trace_output_dir,
                ess_warning_fraction=args.ess_warning_fraction,
                ess_no_go_fraction=args.ess_no_go_fraction,
                log_weight_span_warning=args.log_weight_span_warning,
                initialization=initialization,
                proposal=proposal,
                proposal_family=args.proposal_family,
                guide_family=args.guide_family,
                target_family=args.target_family,
            )
            for case in cases
        ]

    payload = {
        "schema_version": "rn_block_trapped_stationarity_grid_v1",
        "status": "completed",
        "classification": classify_grid(rows),
        "benchmark_tier": "finite-a trapped RN-DMC statistical-control diagnostic",
        "claim_boundary": (
            "finite-a trapped RN-DMC statistical-control diagnostic; "
            "not final paper benchmark by itself"
        ),
        "controls": controls_to_dict(controls),
        "initialization_mode": args.initialization_mode,
        "init_width_log_sigma": args.init_width_log_sigma,
        "breathing_preburn_steps": args.breathing_preburn_steps,
        "breathing_preburn_log_step": args.breathing_preburn_log_step,
        "component_log_scales": list(component_log_scales),
        "component_probabilities": list(component_probabilities),
        "proposal_family": args.proposal_family,
        "guide_family": args.guide_family,
        "target_family": args.target_family,
        "case_count": len(rows),
        "cases": rows,
    }
    if not args.no_write:
        write_summary(output_dir, payload)
        write_case_table(output_dir, rows)
        plot_paths: list[str] = []
        if not args.skip_plots:
            plot_paths = write_plots(ensure_dir(output_dir), rows)
            payload["plots"] = plot_paths
            write_summary(output_dir, payload)
        artifacts = [output_dir / "summary.json", output_dir / "case_table.csv"]
        if not args.skip_plots:
            artifacts.extend(output_dir / path for path in plot_paths)
        write_run_manifest(
            output_dir,
            run_name="rn_block_trapped_stationarity_grid",
            config={
                "cases": [case.case_id for case in cases],
                "seeds": seeds,
                "controls": controls_to_dict(controls),
                "parallel_workers": args.parallel_workers,
                "initialization_mode": args.initialization_mode,
                "init_width_log_sigma": args.init_width_log_sigma,
                "breathing_preburn_steps": args.breathing_preburn_steps,
                "breathing_preburn_log_step": args.breathing_preburn_log_step,
                "component_log_scales": list(component_log_scales),
                "component_probabilities": list(component_probabilities),
                "proposal_family": args.proposal_family,
                "guide_family": args.guide_family,
                "target_family": args.target_family,
            },
            artifacts=artifacts,
            schema_version="rn_block_trapped_stationarity_grid_v1",
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


if __name__ == "__main__":
    main()
