from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrdmc.artifacts import write_json
from hrdmc.io import print_run_summary
from hrdmc.plotting import write_benchmark_packet_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render benchmark-packet comparison figures from summary.json."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.summary.read_text())
    output_dir = args.output_dir or args.summary.parent
    plot_paths = write_benchmark_packet_plots(
        output_dir,
        payload,
        formats=_parse_str_tuple(args.plot_formats),
    )
    manifest = {
        "schema_version": "benchmark_packet_plot_manifest_v1",
        "summary": str(args.summary),
        "plots": plot_paths,
    }
    write_json(output_dir / "plot_manifest.json", manifest)
    manifest_path = output_dir / "plot_manifest.json"
    print_run_summary(
        run="plot_benchmark_packet",
        status="completed",
        summary={"plot_count": len(plot_paths)},
        artifacts={"manifest": str(manifest_path), "output_dir": str(output_dir)},
        verbose_payload=manifest,
        verbose_json=args.verbose_json,
    )


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one format is required")
    return values


if __name__ == "__main__":
    main()
