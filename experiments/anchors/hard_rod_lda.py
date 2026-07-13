from __future__ import annotations

import argparse
from pathlib import Path

from hrdmc.io import print_run_summary
from hrdmc.workflows.anchors.hard_rod_lda import (
    HardRodLDADiagnosticConfig,
    run_hard_rod_lda_diagnostic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build analytic trapped hard-rod LDA diagnostics.")
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--rod-lengths", default="0,0.02,0.05,0.1,0.18803,0.4")
    parser.add_argument("--x-extent", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=2401)
    parser.add_argument("--representative-rod-length", type=float, default=0.18803)
    parser.add_argument("--small-a-rod-length", type=float, default=0.02)
    parser.add_argument("--cubic-abs-tolerance", type=float, default=1e-8)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = HardRodLDADiagnosticConfig(
        n_particles=args.n_particles,
        rod_lengths=tuple(_parse_floats(args.rod_lengths)),
        x_extent=args.x_extent,
        n_grid=args.n_grid,
        representative_rod_length=args.representative_rod_length,
        small_a_rod_length=args.small_a_rod_length,
        cubic_abs_tolerance=args.cubic_abs_tolerance,
        plot_formats=tuple(_parse_strings(args.plot_formats)),
        write_plots=not args.skip_plots,
    )
    output_dir = args.output_dir or (
        repo_root / "results" / "trapped_hard_rods" / "theory" / "hard_rod_lda"
    )
    payload = run_hard_rod_lda_diagnostic(
        config,
        output_dir=output_dir,
        command=_command_from_args(args),
        write=not args.no_write,
    )
    print_run_summary(
        run="hard_rod_lda",
        status=str(payload["status"]),
        summary={
            "n_particles": args.n_particles,
            "rod_length_count": len(config.rod_lengths),
        },
        artifacts={
            "summary": None if args.no_write else str(output_dir / "summary.json"),
            "output_dir": None if args.no_write else str(output_dir),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )
    if payload["status"] != "accepted":
        raise SystemExit(1)


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _command_from_args(args: argparse.Namespace) -> list[str]:
    command = ["python", "experiments/anchors/hard_rod_lda.py"]
    for key, value in vars(args).items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key.replace('_', '-')}")
            continue
        if value is None:
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


if __name__ == "__main__":
    main()
