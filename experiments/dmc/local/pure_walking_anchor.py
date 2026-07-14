from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hrdmc.artifacts import (
    ArtifactRoute,
    artifact_dir,
    build_run_provenance,
    repo_root_from,
    write_json,
    write_run_manifest,
)
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io import print_run_summary, progress_requested
from hrdmc.workflows.dmc.collective_rn import (
    DEFAULT_COMPONENT_LOG_SCALES,
    DEFAULT_COMPONENT_PROBABILITIES,
    DEFAULT_PROPOSAL_FAMILY,
    DEFAULT_TARGET_FAMILY,
    PROPOSAL_FAMILIES,
    TARGET_FAMILIES,
    CollectiveRNControls,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.pure_walking import (
    summarize_pure_walking_case,
    write_pure_walking_seed_table,
)
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    GUIDE_FAMILIES,
    DMCRunControls,
    controls_to_dict,
    dmc_progress_bar,
    parse_case,
    parse_seeds,
)

DEFAULT_CASE = "N8_A0.1"
DEFAULT_LAGS = "0,10,20,30,40,50,100,200"
SUPPORTED_OBSERVABLES = {"r2", "density"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run transported auxiliary forward walking on DMC trajectories."
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=("Case in harmonic-oscillator units, e.g. N8_A0.2 where A=a/a_ho."),
    )
    parser.add_argument("--seeds", default="1001,1002")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=256)
    collective = parser.add_argument_group("optional collective RN move")
    collective.add_argument("--collective-rn", action="store_true")
    collective.add_argument("--collective-cadence-tau", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=120.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=20.0)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=800,
        help=(
            "Density histogram bins. The default is intentionally high enough "
            "to resolve finite-N trapped density peaks in report figures."
        ),
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    collective.add_argument(
        "--proposal-family",
        choices=PROPOSAL_FAMILIES,
        default=DEFAULT_PROPOSAL_FAMILY,
    )
    parser.add_argument(
        "--guide-family",
        choices=GUIDE_FAMILIES,
        default=DEFAULT_GUIDE_FAMILY,
    )
    collective.add_argument(
        "--target-family",
        choices=TARGET_FAMILIES,
        default=DEFAULT_TARGET_FAMILY,
    )
    collective.add_argument(
        "--component-log-scales",
        default=",".join(f"{value:g}" for value in DEFAULT_COMPONENT_LOG_SCALES),
    )
    collective.add_argument(
        "--component-probabilities",
        default=",".join(f"{value:g}" for value in DEFAULT_COMPONENT_PROBABILITIES),
    )
    parser.add_argument("--lags", default=DEFAULT_LAGS)
    parser.add_argument(
        "--observables",
        default="r2",
        help="Comma-separated transported FW observables. Supported: r2,density.",
    )
    parser.add_argument("--observable-source", choices=("raw_r2", "r2_rb"), default="raw_r2")
    parser.add_argument("--block-size-steps", type=int, default=1)
    parser.add_argument("--collection-stride-steps", type=int, default=1)
    parser.add_argument("--min-block-count", type=int, default=30)
    parser.add_argument("--min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--plateau-window-lag-count", type=int, default=4)
    parser.add_argument(
        "--rms-plateau-relative-tolerance",
        type=float,
        default=0.001,
        help=(
            "Declared practical RMS equivalence margin for the FW lag window. "
            "The 0.1%% default is a reporting resolution, not a universal physical "
            "constant."
        ),
    )
    parser.add_argument(
        "--plateau-equivalence-confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = repo_root_from(Path(__file__))
    case = parse_case(args.case)
    seeds = parse_seeds(args.seeds)
    observables = _parse_str_tuple(args.observables)
    unsupported_observables = set(observables) - SUPPORTED_OBSERVABLES
    if unsupported_observables:
        parser.error(
            "unsupported --observables for this runner: "
            + ",".join(sorted(unsupported_observables))
            + "; supported values are r2,density"
        )
    controls = DMCRunControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
    )
    initialization = InitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    collective_rn = _collective_rn_controls(args)
    pure_config = PureWalkingConfig(
        lag_steps=_parse_int_tuple(args.lags),
        observables=observables,
        observable_source=args.observable_source,
        density_bin_edges=(
            np.linspace(-args.grid_extent, args.grid_extent, args.n_bins + 1)
            if "density" in observables
            else None
        ),
        min_block_count=args.min_block_count,
        min_walker_weight_ess=args.min_walker_weight_ess,
        plateau_window_lag_count=args.plateau_window_lag_count,
        rms_plateau_relative_tolerance=args.rms_plateau_relative_tolerance,
        plateau_equivalence_confidence_level=(args.plateau_equivalence_confidence_level),
        block_size_steps=args.block_size_steps,
        collection_stride_steps=args.collection_stride_steps,
        transport_invariant_tests_passed=(
            "lag0_identity",
            "deterministic_parent_map",
            "weight_gauge_shift_cancellation",
        ),
    )
    output_dir = args.output_dir or artifact_dir(
        repo_root, ArtifactRoute("dmc", "local", "pure_walking_anchor")
    )
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label=("Pure walking with collective RN" if collective_rn is not None else "Pure walking"),
        enabled=progress_requested(args.progress),
    ) as bar:
        payload = summarize_pure_walking_case(
            case,
            controls,
            seeds,
            pure_config=pure_config,
            parallel_workers=args.parallel_workers,
            progress=bar,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=args.guide_family,
        )
    if not args.skip_write:
        summary_path = output_dir / "summary.json"
        write_json(summary_path, payload)
        seed_table = write_pure_walking_seed_table(output_dir, payload["seed_results"])
        write_run_manifest(
            output_dir,
            run_name="dmc_pure_walking_anchor",
            config={
                "case": case.case_id,
                "seeds": seeds,
                "controls": controls_to_dict(controls),
                "parallel_workers": args.parallel_workers,
                "initialization_mode": args.initialization_mode,
                "init_width_log_sigma": args.init_width_log_sigma,
                "breathing_preburn_steps": args.breathing_preburn_steps,
                "breathing_preburn_log_step": args.breathing_preburn_log_step,
                "collective_rn": (None if collective_rn is None else collective_rn.to_metadata()),
                "guide_family": args.guide_family,
                "pure_config": payload["pure_config"],
            },
            artifacts=[summary_path, seed_table],
            schema_version="transported_pure_walking_case_v3",
            provenance=build_run_provenance(sys.argv),
        )
    print_run_summary(
        run="pure_walking_anchor",
        status=str(payload.get("status", "completed")),
        summary={
            "case": case.case_id,
            "seed_count": len(seeds),
            "observables": list(observables),
            "classification": payload.get("classification"),
        },
        artifacts={
            "summary": None if args.skip_write else str(output_dir / "summary.json"),
            "seed_table": None if args.skip_write else str(output_dir / "seed_table.csv"),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


def _collective_rn_controls(args: argparse.Namespace) -> CollectiveRNControls | None:
    if not args.collective_rn:
        return None
    return CollectiveRNControls(
        cadence_tau=args.collective_cadence_tau,
        proposal_family=args.proposal_family,
        target_family=args.target_family,
        component_log_scales=_parse_float_tuple(args.component_log_scales),
        component_probabilities=_parse_float_tuple(args.component_probabilities),
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one observable is required")
    return values


if __name__ == "__main__":
    main()
