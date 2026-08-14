from __future__ import annotations

import argparse
from pathlib import Path

from hrdmc.artifacts.progress import progress_bar, progress_requested
from hrdmc.artifacts.terminal import print_run_summary
from hrdmc.sampling.dmc.run import parse_seeds
from hrdmc.system.settings import parse_case
from hrdmc.validation.sampler_equivalence.binding import (
    bind_current_reduced_tg_guide,
    bind_validated_reduced_tg_guide,
)
from hrdmc.validation.sampler_equivalence.calibration import load_calibrated_sampler_choices
from hrdmc.validation.sampler_equivalence.models import VMCSamplingControls, VMCValidationPolicy
from hrdmc.validation.sampler_equivalence.run import run_vmc_validation_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the independent VMC versus branching-free DMC "
            "validation packet for one trapped hard-rod case."
        )
    )
    parser.add_argument("--case", choices=("N10_A0", "N10_A1"), required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--guide-validation-summary", type=Path, default=None)
    parser.add_argument("--rwm-seeds", default="71001,71002,71003,71004,71005")
    parser.add_argument("--mala-seeds", default="72001,72002,72003,72004,72005")
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--burn-in-steps", type=int, default=5000)
    parser.add_argument("--production-steps", type=int, default=20000)
    parser.add_argument("--block-steps", type=int, default=20)
    parser.add_argument("--density-bins", type=int, default=840)
    parser.add_argument("--free-gap-bins", type=int, default=640)
    parser.add_argument("--parallel-workers", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case = parse_case(args.case)
    if case.rod_length == 0.0:
        if args.guide_validation_summary is not None:
            raise ValueError("N10_A0 uses the current exact reduced-TG guide binding")
        binding = bind_current_reduced_tg_guide(expected_case=case)
    else:
        if args.guide_validation_summary is None:
            raise ValueError("finite-A validation requires --guide-validation-summary")
        binding = bind_validated_reduced_tg_guide(
            args.guide_validation_summary,
            expected_case=case,
        )
    choices = load_calibrated_sampler_choices(
        args.calibration_summary,
        expected_case=case,
    )
    controls = VMCSamplingControls(
        walkers=args.walkers,
        burn_in_steps=args.burn_in_steps,
        production_steps=args.production_steps,
        block_steps=args.block_steps,
        density_bins=args.density_bins,
        free_gap_bins=args.free_gap_bins,
    )
    total = controls.production_steps * 10
    with progress_bar(
        total=total,
        label=f"VMC validation {case.case_id}",
        enabled=progress_requested(args.progress),
    ) as progress:
        result = run_vmc_validation_case(
            case,
            binding,
            controls=controls,
            policy=VMCValidationPolicy(),
            sampler_choices=choices,
            rwm_seeds=tuple(parse_seeds(args.rwm_seeds)),
            mala_seeds=tuple(parse_seeds(args.mala_seeds)),
            parallel_workers=args.parallel_workers,
            output_dir=args.output_dir,
            progress=progress,
        )
    print_run_summary(
        run="trapped_vmc_validation_case",
        status=result.status,
        summary={
            "case": case.case_id,
            "actual_workers": result.actual_workers,
            "failed_or_unresolved_checks": result.assessment.get("failed_or_unresolved_checks", []),
        },
        artifacts={**result.artifacts, "output_dir": str(args.output_dir)},
        verbose_payload=result.assessment,
        verbose_json=args.verbose_json,
    )
    if result.status not in {"accepted", "accepted_with_warnings"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
