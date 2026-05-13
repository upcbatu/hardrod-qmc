from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrdmc.io.artifacts import write_json_atomic
from hrdmc.plotting.figures.finite_a_n2_reference import (
    write_finite_a_n2_reference_plots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render plots for an existing finite-a N=2 reference packet."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--plot-formats", default="png,pdf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    output_dir = args.summary.parent
    plot_paths = write_finite_a_n2_reference_plots(
        output_dir,
        payload,
        formats=_parse_str_tuple(args.plot_formats),
    )
    payload["plots"] = plot_paths
    write_json_atomic(args.summary, payload)
    print("\n".join(plot_paths))


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one plot format")
    return values


if __name__ == "__main__":
    main()
