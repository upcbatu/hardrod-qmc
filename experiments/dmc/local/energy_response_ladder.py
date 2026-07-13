from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary, progress_requested
from hrdmc.workflows.dmc.energy_response import (
    EnergyResponseLadderControls,
    parse_five_point_offsets,
    run_energy_response_ladder,
)
from hrdmc.workflows.dmc.guide_validation import (
    load_validated_contact_guide,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import DMCRunControls, TrappedCase, parse_case, parse_seeds

DEFAULT_OFFSETS = "-0.005,-0.0025,0,0.0025,0.005"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a manifest-bound five-point local-DMC trap-coupling ladder "
            "for a paired-seed Hellmann-Feynman R2 estimate."
        )
    )
    parser.add_argument("--case", default="N20_A10")
    parser.add_argument("--relative-lambda-offsets", default=DEFAULT_OFFSETS)
    parser.add_argument("--seeds", default="9501,9502,9503,9504,9505")
    parser.add_argument("--dt", type=float, default=0.00025)
    parser.add_argument("--walkers", type=int, default=512)
    parser.add_argument("--burn-tau", type=float, default=20.0)
    parser.add_argument("--production-tau", type=float, default=60.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=90.0)
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument("--ess-resample-fraction", type=float, default=0.35)
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-invalid-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-lattice",
    )
    parser.add_argument("--parallel-workers", type=int, default=5)
    guide = parser.add_mutually_exclusive_group(required=True)
    guide.add_argument(
        "--guide-validation-summary",
        type=Path,
        help="Use the validated contact-guide parameters in this summary.",
    )
    guide.add_argument(
        "--relative-alpha",
        type=float,
        help="Use an explicit reduced-coordinate width for the reduced-TG guide.",
    )
    parser.add_argument(
        "--guide-family",
        choices=("reduced-tg",),
        default=None,
        help="Required with --relative-alpha; omit with a validation summary.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root_from(Path(__file__))
    case = parse_case(args.case)
    (
        guide_family,
        relative_alpha,
        contact_beta,
        guide_source,
        guide_source_sha256,
        guide_source_manifest_sha256,
        guide_source_identity_fingerprint,
    ) = _resolve_guide(parser, args, case)
    run_controls = DMCRunControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        ess_resample_fraction=args.ess_resample_fraction,
        local_step_method="metropolis",
        relative_alpha=relative_alpha,
        contact_beta=contact_beta,
    )
    controls = EnergyResponseLadderControls(
        run=run_controls,
        seeds=tuple(parse_seeds(args.seeds)),
        relative_lambda_offsets=parse_five_point_offsets(args.relative_lambda_offsets),
        initialization=InitializationControls(mode=args.initialization_mode),
        guide_family=guide_family,
        parallel_workers=args.parallel_workers,
        ess_warning_fraction=args.ess_warning_fraction,
        ess_invalid_fraction=args.ess_invalid_fraction,
        log_weight_span_warning=args.log_weight_span_warning,
        guide_parameter_source=guide_source,
        guide_parameter_source_sha256=guide_source_sha256,
        guide_parameter_source_manifest_sha256=guide_source_manifest_sha256,
        guide_parameter_source_identity_fingerprint=guide_source_identity_fingerprint,
    )
    payload = run_energy_response_ladder(
        case=case,
        controls=controls,
        output_dir=args.output_dir,
        command=sys.argv,
        resume=args.resume,
        dry_run=args.dry_run,
        progress=progress_requested(args.progress),
    )
    estimate = payload.get("energy_response", {})
    artifacts = payload.get("artifacts", {})
    print_run_summary(
        run="energy_response_ladder",
        status=str(payload["status"]),
        summary={
            "case": case.case_id,
            "point_count": payload.get("point_count", 5),
            "seed_count": payload.get("seed_count", len(controls.seeds)),
            "r2": estimate.get("pure_r2"),
            "r2_seed_stderr": estimate.get("pure_r2_seed_stderr"),
            "rms_radius": estimate.get("rms_radius"),
        },
        artifacts={
            "summary": artifacts.get("summary"),
            "point_table": artifacts.get("point_table"),
            "output_dir": artifacts.get("output_dir", str(args.output_dir.resolve())),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


def _resolve_guide(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    case: TrappedCase,
) -> tuple[
    str,
    float,
    float | None,
    str,
    str | None,
    str | None,
    str | None,
]:
    if args.guide_validation_summary is not None:
        if args.guide_family is not None:
            parser.error("omit --guide-family when using --guide-validation-summary")
        validated_guide = load_validated_contact_guide(
            args.guide_validation_summary,
            case=case,
        )
        return (
            "contact-corrected-reduced-tg",
            validated_guide.relative_alpha,
            validated_guide.contact_beta,
            str(validated_guide.summary_path),
            validated_guide.summary_sha256,
            validated_guide.manifest_sha256,
            validated_guide.identity_fingerprint,
        )
    if args.guide_family != "reduced-tg":
        parser.error("--relative-alpha requires --guide-family reduced-tg")
    if args.relative_alpha is None or args.relative_alpha <= 0.0:
        parser.error("--relative-alpha must be positive")
    return "reduced-tg", float(args.relative_alpha), None, "explicit", None, None, None


if __name__ == "__main__":
    main()
