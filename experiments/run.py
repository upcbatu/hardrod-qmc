"""Research commands; numerical work remains in the hrdmc library."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from hrdmc.artifacts.schema import to_jsonable
from hrdmc.production.matrix.final_matrix import FinalMatrixConfig, case_grid_plan
from hrdmc.production.matrix.method import DEFAULT_GUIDE_VALIDATION_SUMMARY, row_method
from hrdmc.production.pure_walking import pure_config_for_case
from hrdmc.production.variational import VariationalRunControls, run_variational_workflow
from hrdmc.sampling.vmc.transitions import VMCConfig
from hrdmc.system.guide_selection import GuideSelection, select_guide
from hrdmc.system.settings import THESIS_CASE_ORDER, TrappedCase, make_grid, parse_case
from hrdmc.system.units import forward_time_steps

ASSESSMENTS = {
    "plot": "report.plot",
    "timestep": "dmc.local.extrapolate_timestep",
    "population": "dmc.local.assess_population_systematics",
    "fw-sensitivity": "dmc.local.assess_fw_sensitivity",
    "uncertainty": "dmc.local.assemble_numerical_systematics",
    "validate-vmc": "vmc.validation_packet",
}


def _case(args: argparse.Namespace) -> TrappedCase:
    if args.case is not None:
        if args.n is not None or args.a is not None:
            raise ValueError("use --case or the pair --n and --a")
        return parse_case(args.case)
    if args.n is None or args.a is None or not math.isfinite(args.a):
        raise ValueError("provide --case or both --n and --a with finite A")
    return TrappedCase(args.n, args.a)


def _case_arguments(parser: argparse.ArgumentParser, *, existing_case: bool = False) -> None:
    if not existing_case:
        parser.add_argument("--case", help="For example N10_A0.1; A=a/a_ho")
    parser.add_argument("--n", type=int, help="Number of rods (at least 2)")
    parser.add_argument("--a", type=float, help="Diameter a/a_ho")


def _guide_arguments(parser: argparse.ArgumentParser, *, existing_alpha: bool = False) -> None:
    parser.add_argument("--preset", choices=("thesis",), help="Use stored thesis guide/settings")
    if existing_alpha:
        parser.add_argument("--alpha", dest="relative_alpha", type=float, default=argparse.SUPPRESS)
    else:
        parser.add_argument("--alpha", dest="relative_alpha", type=float)
    parser.add_argument(
        "--alpha-from", type=Path, help="Optimization summary.json; still a candidate"
    )


def _guide(args: argparse.Namespace, case: TrappedCase) -> GuideSelection:
    registry = getattr(args, "guide_validation_summary", None)
    if args.preset == "thesis" and args.relative_alpha is None and args.alpha_from is None:
        registry = registry or DEFAULT_GUIDE_VALIDATION_SUMMARY
    return select_guide(
        case, alpha=args.relative_alpha, alpha_from=args.alpha_from, registry=registry
    )


def _seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",")]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    return seeds


def _empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"choose an empty output directory: {path}")


def _show(plan: dict) -> None:
    print(json.dumps(to_jsonable(plan), indent=2, allow_nan=False), flush=True)


def dmc_parser() -> argparse.ArgumentParser:
    module = importlib.import_module("dmc.local.benchmark_packet")
    parser = module.build_parser()
    parser.description = "DMC with explicit forward times in units of 1/Omega."
    parser.set_defaults(case=None)
    _case_arguments(parser, existing_case=True)
    _guide_arguments(parser, existing_alpha=True)
    parser.add_argument("--density-fw-times", help="Physical times, including 0; e.g. 0,2,4,7")
    parser.add_argument("--rms-fw-times", help="Physical times, including 0; e.g. 0,5,10,20")
    parser.add_argument("--store-time", type=float, help="Energy recording interval in 1/Omega")
    parser.add_argument(
        "--rms-collection-time", type=float, help="R2 collection interval in 1/Omega"
    )
    parser.add_argument("--density-collection-time", type=float, help="Density interval in 1/Omega")
    parser.add_argument("--workers", dest="parallel_workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output", dest="output_dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--burn-time", dest="burn_tau", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--run-time", dest="production_tau", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Show resolved settings; do not run")
    return parser


def _thesis_dmc_defaults(parser: argparse.ArgumentParser, case: TrappedCase) -> None:
    if case.case_id not in THESIS_CASE_ORDER:
        raise ValueError("the thesis DMC preset supports N=10,20 and A=0,0.1,1,10")
    method = row_method(case.case_id, guide_validation_root=None)
    grid = case_grid_plan(FinalMatrixConfig(), case.case_id, method)
    parser.set_defaults(
        dt=method.dt,
        walkers=method.walkers,
        drift_limiter=method.drift_limiter,
        initialization_mode=method.initialization_mode,
        breathing_preburn_steps=method.breathing_preburn_steps,
        init_width_log_sigma=method.init_width_log_sigma,
        breathing_preburn_log_step=method.breathing_preburn_log_step,
        burn_tau=60,
        production_tau=480,
        seeds="7001,7002,7003,7004,7005",
        grid_extent=grid["grid_extent"],
        n_bins=grid["n_bins"],
        parallel_workers=5,
        rms_fw_times=",".join(str(lag * method.dt) for lag in method.pure_fw_lags),
        density_fw_times=",".join(str(lag * method.dt) for lag in method.pure_fw_density_lags),
        store_time=method.store_every * method.dt,
        rms_collection_time=method.pure_fw_collection_stride_steps * method.dt,
        density_collection_time=method.pure_fw_density_collection_stride_steps * method.dt,
        pure_fw_min_block_count=20,
        pure_fw_rms_plateau_relative_tolerance=0.001,
        pure_fw_density_plateau_window_lag_count=3,
        pure_fw_observable_source="r2_rb",
        pure_fw_density_source="com_rao_blackwell",
        pure_fw_density_parity_average=True,
    )


def _dmc_times(args: argparse.Namespace, argv: list[str]) -> None:
    conversions = (
        ("rms_fw_times", "pure_fw_lags", "--pure-fw-lags", True),
        ("density_fw_times", "pure_fw_density_lags", "--pure-fw-density-lags", True),
        ("store_time", "store_every", "--store-every", False),
        (
            "rms_collection_time",
            "pure_fw_collection_stride_steps",
            "--pure-fw-collection-stride-steps",
            False,
        ),
        (
            "density_collection_time",
            "pure_fw_density_collection_stride_steps",
            "--pure-fw-density-collection-stride-steps",
            False,
        ),
    )
    for source, target, step_flag, sequence in conversions:
        value = getattr(args, source)
        if value is None:
            continue
        if any(item.split("=")[0] == step_flag for item in argv):
            raise ValueError(f"use physical times instead of {step_flag} with this command/preset")
        times = [float(item) for item in value.split(",")] if sequence else [value]
        steps = [forward_time_steps(time, args.dt) for time in times]
        setattr(args, target, ",".join(map(str, steps)) if sequence else steps[0])
    if forward_time_steps(args.burn_tau, args.dt) < 1:
        raise ValueError("DMC burn-time must contain at least one step")
    if forward_time_steps(args.production_tau, args.dt) < 1:
        raise ValueError("DMC run-time must contain at least one step")


def prepare_dmc(argv: list[str]) -> tuple[argparse.Namespace, GuideSelection, dict]:
    parser = dmc_parser()
    args = parser.parse_args(argv)
    case = _case(args)
    if args.preset == "thesis":
        _thesis_dmc_defaults(parser, case)
        args = parser.parse_args(argv)
    args.case = case.case_id
    guide = _guide(args, case)
    args.relative_alpha = guide.relative_alpha
    args.guide_validation_summary = None
    if guide.validation == "validated":
        args.relative_alpha = None
        args.guide_validation_summary = Path(guide.source)
    seeds = _seeds(args.seeds)
    _dmc_times(args, argv)
    if args.output_dir is None:
        raise ValueError("--output is required")
    _empty_output(args.output_dir)
    if args.parallel_workers < 0:
        raise ValueError("workers must be nonnegative (0 selects automatically)")
    module = importlib.import_module("dmc.local.benchmark_packet")
    options = module.workflow_options(args)
    controls, pure = options["controls"], options["pure_config"]
    grid = make_grid(controls, case)
    options["initialization"].validate()
    pure_config_for_case(pure, case=case, grid=grid).validate()
    if max((*pure.lag_steps, *(pure.density_lag_steps or ()))) >= controls.production_steps:
        raise ValueError("run-time must exceed the largest forward time")
    plan = {
        "case": case.case_id,
        "guide": asdict(guide),
        "controls": asdict(controls),
        "seeds": seeds,
        "workers": args.parallel_workers,
        "equilibration_steps": controls.burn_in_steps,
        "production_steps": controls.production_steps,
        "rms_forward_times": [lag * controls.dt for lag in pure.lag_steps],
        "rms_lag_steps": pure.lag_steps,
        "density_forward_times": [
            lag * controls.dt for lag in (pure.density_lag_steps or pure.lag_steps)
        ],
        "density_lag_steps": pure.density_lag_steps or pure.lag_steps,
        "forward_walking": asdict(pure),
        "initialization": asdict(options["initialization"]),
        "actual_grid_extent": float(max(abs(grid[0]), abs(grid[-1]))),
        "grid_point_spacing": float(grid[1] - grid[0]),
        "output": str(args.output_dir),
    }
    # workflow_options loads a registry width into args; restore its exclusive CLI input.
    if guide.validation == "validated":
        args.relative_alpha = None
    return args, guide, plan


def _dmc(argv: list[str]) -> None:
    args, guide, plan = prepare_dmc(argv)
    if args.verbose_json:
        _show(plan)
    else:
        _show_dmc_plan(plan)
    if not args.dry_run:
        module = importlib.import_module("dmc.local.benchmark_packet")
        module.execute(
            args,
            allow_unvalidated_guide=guide.validation == "not_independently_validated",
            guide_provenance=asdict(guide),
        )


def _show_dmc_plan(plan: dict) -> None:
    controls, fw, guide = (plan[name] for name in ("controls", "forward_walking", "guide"))
    print(f"{plan['case']} | DMC | alpha={guide['relative_alpha'] or 1} ({guide['validation']})")
    print(f"Guide source: {guide['candidate_path'] or guide['source']}")
    print(
        f"dt={controls['dt']:g}; walkers={controls['walkers']}; seeds={plan['seeds']}; "
        f"workers={plan['workers']}"
    )
    print(
        f"Equilibration {controls['burn_tau']:g} -> {plan['equilibration_steps']} steps; "
        f"production {controls['production_tau']:g} -> {plan['production_steps']} steps"
    )
    print(f"RMS forward times: {plan['rms_forward_times']} -> lags {plan['rms_lag_steps']}")
    print(
        f"Density forward times: {plan['density_forward_times']} "
        f"-> lags {plan['density_lag_steps']}"
    )
    print(
        f"R2/density collection strides: {fw['collection_stride_steps']}/"
        f"{fw['density_collection_stride_steps'] or fw['collection_stride_steps']} steps"
    )
    print(
        f"Grid half-width={plan['actual_grid_extent']:g}; centers={controls['n_bins']}; "
        f"spacing={plan['grid_point_spacing']:.6g}"
    )
    print(
        f"FW minimum blocks={fw['min_block_count']}; walker ESS={fw['min_walker_weight_ess']:g}; "
        f"ancestor ESS={fw['min_source_ancestor_ess']:g}; "
        f"largest family={fw['max_source_family_fraction']:g}"
    )
    print(
        f"Plateau margins: RMS={fw['rms_plateau_relative_tolerance']:g}, "
        f"density L2={fw['density_plateau_relative_l2_tolerance']:g}; "
        f"confidence={fw['plateau_equivalence_confidence_level']:g}"
    )
    print(f"Output: {plan['output']}", flush=True)


def _vmc(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Sample a trial wavefunction with RWM or MALA.")
    _case_arguments(parser)
    _guide_arguments(parser)
    parser.add_argument("--sampler", choices=("mala", "rwm"), default="mala")
    parser.add_argument("--seeds", default="7001,7002,7003,7004,7005")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--burn-in-steps", type=int, default=20000)
    parser.add_argument("--production-steps", type=int, default=20000)
    parser.add_argument("--mala-step", type=float, default=0.001)
    parser.add_argument("--rwm-step", type=float, default=0.5)
    parser.add_argument("--drift-limiter", choices=("none", "umrigar"), default="none")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--grid-extent", type=float, default=35)
    parser.add_argument("--bins", type=int, default=840)
    parser.add_argument("--initial-scale", type=float, default=1)
    parser.add_argument("--cutoff-epsilons", default="0.01,0.02,0.04")
    parser.add_argument("--rhat-limit", type=float, default=1.01)
    parser.add_argument("--min-effective-samples", type=float, default=400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    case, seeds = _case(args), _seeds(args.seeds)
    guide = _guide(args, case)
    controls = VariationalRunControls(
        VMCConfig(
            args.walkers,
            args.burn_in_steps,
            args.production_steps,
            args.sampler,
            args.mala_step,
            args.rwm_step,
            args.drift_limiter,
        ),
        args.block_size,
        args.grid_extent,
        args.bins,
        args.initial_scale,
        tuple(float(item) for item in args.cutoff_epsilons.split(",")),
        args.rhat_limit,
        args.min_effective_samples,
    )
    controls.validate()
    _empty_output(args.output)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.verbose_json:
        _show(
            {
                "case": case.case_id,
                "guide": asdict(guide),
                "controls": asdict(controls),
                "seeds": seeds,
                "workers": args.workers,
                "output": str(args.output),
            }
        )
    else:
        print(
            f"{case.case_id} | VMC {args.sampler} | alpha={guide.relative_alpha or 1} "
            f"({guide.validation})"
        )
        print(
            f"Walkers={args.walkers}; seeds={seeds}; workers={args.workers}; "
            f"equilibration/production steps={args.burn_in_steps}/{args.production_steps}"
        )
        print(
            f"RWM step={args.rwm_step:g}; MALA step={args.mala_step:g}; "
            f"drift limiter={args.drift_limiter}; block size={args.block_size}"
        )
        print(
            f"Grid=[{-args.grid_extent:g}, {args.grid_extent:g}]; bins={args.bins}; "
            f"initial scale={args.initial_scale:g}; output={args.output}",
            flush=True,
        )
    if not args.dry_run:
        result = run_variational_workflow(
            case, guide, controls, seeds, workers=args.workers, output=args.output, command=sys.argv
        )
        if args.verbose_json:
            _show(result)
        else:
            for name, estimate in result["estimates"].items():
                print(
                    f"{name}: {estimate['value']:.10g}; seed stderr={estimate['stderr']}; "
                    f"{result['diagnostics'][name]['classification']}"
                )
            print(f"Files written to {args.output}")


def _alpha(argv: list[str]) -> None:
    module = importlib.import_module("vmc.optimize_alpha")
    parser = module.build_parser(require_case=False)
    _case_arguments(parser, existing_case=True)
    args = parser.parse_args(argv)
    args.case = _case(args).case_id
    _empty_output(args.output_dir)
    module.execute(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hard-rod calculations from the repository root."
    )
    parser.add_argument("command", choices=("dmc", "vmc", "optimize-alpha", *ASSESSMENTS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command in ASSESSMENTS:
            module = importlib.import_module(ASSESSMENTS[args.command])
            sys.argv = [sys.argv[0], *args.arguments]
            module.main()
        else:
            {"dmc": _dmc, "vmc": _vmc, "optimize-alpha": _alpha}[args.command](args.arguments)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
