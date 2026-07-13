from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from hrdmc.workflows.dmc.guide_validation import (
    load_validated_contact_guide,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.stationarity_grid import run_stationarity_grid_workflow
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    GUIDE_FAMILIES,
    DMCRunControls,
    parse_case,
    parse_seeds,
)

DEFAULT_CASES = "N4_A0.1,N8_A0.1,N8_A0.2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trapped DMC stationarity diagnostic grid.")
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help=("Comma-separated harmonic-oscillator-unit cases, e.g. N4_A0.1,N8_A0.2."),
    )
    parser.add_argument("--seeds", default="301,302,303,304")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=512)
    collective = parser.add_argument_group("optional collective RN move")
    collective.add_argument("--collective-rn", action="store_true")
    collective.add_argument("--collective-cadence-tau", type=float, default=0.005)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
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
            "Optional reduced-coordinate Gaussian width for the reduced-TG "
            "guide; the center-of-mass width remains harmonic."
        ),
    )
    parser.add_argument(
        "--guide-validation-summary",
        type=Path,
        default=None,
        help=(
            "Load relative-alpha and contact-beta values from a manifest-bound "
            "contact-guide calibration summary with status validated."
        ),
    )
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-invalid-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    collective.add_argument(
        "--proposal-family",
        choices=PROPOSAL_FAMILIES,
        default=DEFAULT_PROPOSAL_FAMILY,
        help="RN base proposal family; RN collective scaling still applies.",
    )
    parser.add_argument(
        "--guide-family",
        choices=GUIDE_FAMILIES,
        default=None,
        help=(
            f"Importance-sampling guide family; defaults to {DEFAULT_GUIDE_FAMILY}. "
            "Omit when using --guide-validation-summary."
        ),
    )
    collective.add_argument(
        "--target-family",
        choices=TARGET_FAMILIES,
        default=DEFAULT_TARGET_FAMILY,
        help="RN target kernel family.",
    )
    collective.add_argument(
        "--component-log-scales",
        default=",".join(f"{value:g}" for value in DEFAULT_COMPONENT_LOG_SCALES),
        help="Comma-separated RN collective log-scale mixture components.",
    )
    collective.add_argument(
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
        help="Output directory; defaults to results/dmc/local/trapped_stationarity_grid",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    cases = [parse_case(item) for item in args.cases.split(",") if item.strip()]
    guide_family = args.guide_family or DEFAULT_GUIDE_FAMILY
    relative_alpha = args.relative_alpha
    contact_beta: float | None = None
    guide_parameter_source = "explicit"
    guide_parameter_source_sha256: str | None = None
    guide_parameter_source_manifest_sha256: str | None = None
    guide_parameter_source_identity_fingerprint: str | None = None
    if args.guide_validation_summary is not None:
        if relative_alpha is not None or args.guide_family is not None:
            raise ValueError(
                "omit explicit guide parameters and --guide-family when using "
                "--guide-validation-summary"
            )
        if len(cases) != 1:
            raise ValueError("--guide-validation-summary requires exactly one stationarity case")
        validated_guide = load_validated_contact_guide(
            args.guide_validation_summary,
            case=cases[0],
        )
        relative_alpha = validated_guide.relative_alpha
        contact_beta = validated_guide.contact_beta
        guide_family = "contact-corrected-reduced-tg"
        guide_parameter_source = str(validated_guide.summary_path)
        guide_parameter_source_sha256 = validated_guide.summary_sha256
        guide_parameter_source_manifest_sha256 = validated_guide.manifest_sha256
        guide_parameter_source_identity_fingerprint = validated_guide.identity_fingerprint
    elif guide_family == "contact-corrected-reduced-tg":
        raise ValueError(
            "contact-corrected-reduced-tg production requires --guide-validation-summary"
        )
    controls = DMCRunControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        relative_alpha=relative_alpha,
        contact_beta=contact_beta,
    )
    component_log_scales = _parse_float_tuple(args.component_log_scales)
    component_probabilities = _parse_float_tuple(args.component_probabilities)
    initialization = InitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    collective_rn = (
        CollectiveRNControls(
            cadence_tau=args.collective_cadence_tau,
            proposal_family=args.proposal_family,
            target_family=args.target_family,
            component_log_scales=component_log_scales,
            component_probabilities=component_probabilities,
        )
        if args.collective_rn
        else None
    )
    result = run_stationarity_grid_workflow(
        cases,
        controls,
        seeds,
        parallel_workers=args.parallel_workers,
        progress=progress_requested(args.progress),
        output_dir=args.output_dir,
        write_artifacts=not args.no_write,
        write_plots=not args.skip_plots,
        command=sys.argv,
        ess_warning_fraction=args.ess_warning_fraction,
        ess_invalid_fraction=args.ess_invalid_fraction,
        log_weight_span_warning=args.log_weight_span_warning,
        initialization=initialization,
        collective_rn=collective_rn,
        guide_family=guide_family,
        guide_parameter_source=guide_parameter_source,
        guide_parameter_source_sha256=guide_parameter_source_sha256,
        guide_parameter_source_manifest_sha256=guide_parameter_source_manifest_sha256,
        guide_parameter_source_identity_fingerprint=(guide_parameter_source_identity_fingerprint),
    )
    print_run_summary(
        run="trapped_stationarity_grid",
        status=result.status,
        summary=result.summary,
        artifacts=result.artifacts,
        verbose_payload=result.payload,
        verbose_json=args.verbose_json,
    )


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


if __name__ == "__main__":
    main()
