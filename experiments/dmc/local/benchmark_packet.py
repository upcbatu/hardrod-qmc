from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts.progress import progress_requested
from hrdmc.artifacts.terminal import print_run_summary
from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.production.benchmark.run import run_benchmark_packet_workflow
from hrdmc.sampling.dmc.run import parse_seeds
from hrdmc.sampling.initial_conditions import InitializationControls
from hrdmc.system.guide_registry import (
    load_validated_reduced_tg_guide,
)
from hrdmc.system.settings import DMCRunControls, parse_case
from hrdmc.trial.guide import DEFAULT_GUIDE_FAMILY, GUIDE_FAMILIES

DEFAULT_CASE = "N8_A0.1"
DEFAULT_LAGS = "0,10,20,30,40,50,100,200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one DMC benchmark packet with energy diagnostics and transported FW observables."
        )
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=("Case in harmonic-oscillator units, e.g. N8_A0.2 where A=a/a_ho."),
    )
    parser.add_argument("--seeds", default="1001,1002")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument(
        "--drift-limiter",
        choices=("none", "umrigar"),
        default="none",
        help="Finite-timestep MALA drift limiter; branching and local energy are unchanged.",
    )
    parser.add_argument("--walkers", type=int, default=256)
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
    parser.add_argument(
        "--relative-alpha",
        type=float,
        default=None,
        help=(
            "Optional reduced-coordinate internal Gaussian width for the "
            "reduced-TG guide. The center-of-mass width remains harmonic."
        ),
    )
    parser.add_argument(
        "--guide-validation-summary",
        type=Path,
        default=None,
        help="Load relative_alpha from the validated final-matrix guide registry.",
    )
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument(
        "--guide-family",
        choices=GUIDE_FAMILIES,
        default=None,
        help=(
            f"Importance-sampling guide family; defaults to {DEFAULT_GUIDE_FAMILY}. "
            "Omit when using --guide-validation-summary."
        ),
    )
    parser.add_argument(
        "--ess-resample-fraction",
        type=float,
        default=0.35,
        help="Resample when normalized walker-weight ESS falls below this fraction.",
    )
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-invalid-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument("--pure-fw-lags", default=DEFAULT_LAGS)
    parser.add_argument(
        "--pure-fw-density-lags",
        default=None,
        help=(
            "Optional density-only FW lag ladder. Use this with a coarser "
            "density collection stride when long density transport would be large."
        ),
    )
    parser.add_argument(
        "--pure-fw-observables",
        default="r2,density",
    )
    parser.add_argument(
        "--pure-fw-observable-source",
        choices=("raw_r2", "r2_rb"),
        default="raw_r2",
    )
    parser.add_argument(
        "--pure-fw-density-source",
        choices=("raw_density", "com_rao_blackwell"),
        default="raw_density",
    )
    parser.add_argument(
        "--pure-fw-density-parity-average",
        action="store_true",
        help="Average density bins with their exact trap-parity partners.",
    )
    parser.add_argument("--pure-fw-block-size-steps", type=int, default=1)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=1)
    parser.add_argument("--pure-fw-density-collection-stride-steps", type=int, default=None)
    parser.add_argument("--pure-fw-min-block-count", type=int, default=30)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-min-source-ancestor-ess", type=float, default=50.0)
    parser.add_argument("--pure-fw-max-source-family-fraction", type=float, default=0.10)
    parser.add_argument(
        "--pure-fw-rms-plateau-relative-tolerance",
        type=float,
        default=0.0,
        help=(
            "Practical relative RMS-radius equivalence margin for the aggregate "
            "paired-seed R2 lag window. Zero requires exact equivalence."
        ),
    )
    parser.add_argument(
        "--pure-fw-plateau-equivalence-confidence-level",
        type=float,
        default=0.95,
        help="Family-wise confidence level for aggregate paired-lag equivalence.",
    )
    parser.add_argument("--pure-fw-plateau-window-lag-count", type=int, default=4)
    parser.add_argument("--pure-fw-density-plateau-window-lag-count", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case = parse_case(args.case)
    seeds = parse_seeds(args.seeds)
    guide_family = args.guide_family or DEFAULT_GUIDE_FAMILY
    guide_parameter_source = "explicit"
    if args.guide_validation_summary is not None:
        if args.relative_alpha is not None or args.guide_family is not None:
            raise ValueError(
                "omit explicit guide parameters and --guide-family when using "
                "--guide-validation-summary"
            )
        validated_guide = load_validated_reduced_tg_guide(
            args.guide_validation_summary,
            case=case,
        )
        args.relative_alpha = validated_guide.relative_alpha
        guide_family = "reduced-tg"
        guide_parameter_source = str(validated_guide.summary_path)
    controls = DMCRunControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        ess_resample_fraction=args.ess_resample_fraction,
        drift_limiter=args.drift_limiter,
        relative_alpha=args.relative_alpha,
    )
    initialization = InitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    pure_config = PureWalkingConfig(
        lag_steps=_parse_int_tuple(args.pure_fw_lags),
        observables=_parse_str_tuple(args.pure_fw_observables),
        observable_source=args.pure_fw_observable_source,
        density_source=args.pure_fw_density_source,
        density_parity_average=args.pure_fw_density_parity_average,
        min_block_count=args.pure_fw_min_block_count,
        min_walker_weight_ess=args.pure_fw_min_walker_weight_ess,
        min_source_ancestor_ess=args.pure_fw_min_source_ancestor_ess,
        max_source_family_fraction=args.pure_fw_max_source_family_fraction,
        rms_plateau_relative_tolerance=args.pure_fw_rms_plateau_relative_tolerance,
        plateau_equivalence_confidence_level=(args.pure_fw_plateau_equivalence_confidence_level),
        plateau_window_lag_count=args.pure_fw_plateau_window_lag_count,
        density_lag_steps=(
            None
            if args.pure_fw_density_lags is None
            else _parse_int_tuple(args.pure_fw_density_lags)
        ),
        density_collection_stride_steps=args.pure_fw_density_collection_stride_steps,
        density_plateau_window_lag_count=args.pure_fw_density_plateau_window_lag_count,
        block_size_steps=args.pure_fw_block_size_steps,
        collection_stride_steps=args.pure_fw_collection_stride_steps,
        transport_invariant_tests_passed=(
            "lag0_identity",
            "deterministic_parent_map",
            "composed_parent_map_associativity",
            "weight_gauge_shift_cancellation",
        ),
    )
    plot_formats = ("png", "pdf") if args.skip_write else _parse_str_tuple(args.plot_formats)
    result = run_benchmark_packet_workflow(
        case,
        controls,
        seeds,
        pure_config=pure_config,
        parallel_workers=args.parallel_workers,
        progress=progress_requested(args.progress),
        output_dir=args.output_dir,
        write_artifacts=not args.skip_write,
        write_plots=not args.skip_plots,
        plot_formats=plot_formats,
        command=sys.argv,
        ess_warning_fraction=args.ess_warning_fraction,
        ess_invalid_fraction=args.ess_invalid_fraction,
        log_weight_span_warning=args.log_weight_span_warning,
        initialization=initialization,
        guide_family=guide_family,
        guide_parameter_source=guide_parameter_source,
    )
    print_run_summary(
        run="dmc_benchmark_packet",
        status=result.status,
        summary=result.summary,
        artifacts=result.artifacts,
        verbose_payload=result.payload,
        verbose_json=args.verbose_json,
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
