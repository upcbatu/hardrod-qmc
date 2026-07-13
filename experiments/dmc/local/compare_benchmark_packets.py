from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrdmc.artifacts import write_json
from hrdmc.io import print_run_summary
from hrdmc.plotting import write_benchmark_packet_comparison_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render shared-axis comparison plots from benchmark-packet summaries."
    )
    parser.add_argument(
        "--packet",
        action="append",
        required=True,
        help="Packet in label=/path/to/summary.json form. Repeat for each panel.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = [_load_packet(value) for value in args.packet]
    entries = [(label, summary_path.parent, payload) for label, summary_path, payload in loaded]
    plot_paths = write_benchmark_packet_comparison_plots(
        args.output_dir,
        entries,
        formats=_parse_str_tuple(args.plot_formats),
    )
    manifest = {
        "schema_version": "benchmark_packet_comparison_manifest_v1",
        "packets": [
            {"label": label, "summary": str(summary_path)} for label, summary_path, _ in loaded
        ],
        "plots": plot_paths,
    }
    write_json(args.output_dir / "comparison_manifest.json", manifest)
    manifest_path = args.output_dir / "comparison_manifest.json"
    print_run_summary(
        run="compare_benchmark_packets",
        status="completed",
        summary={"packet_count": len(loaded), "plot_count": len(plot_paths)},
        artifacts={"manifest": str(manifest_path), "output_dir": str(args.output_dir)},
        verbose_payload=manifest,
        verbose_json=args.verbose_json,
    )


def _load_packet(value: str) -> tuple[str, Path, dict]:
    if "=" not in value:
        raise ValueError("--packet must use label=summary.json form")
    label, path_text = value.split("=", maxsplit=1)
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text())
    return label.strip(), path, payload


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one format is required")
    return values


if __name__ == "__main__":
    main()
