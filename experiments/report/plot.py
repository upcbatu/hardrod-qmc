"""Plot supplied density CSVs or explicitly selected numeric CSV columns."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrdmc.plotting.csv_curves import plot_csv_curves, read_csv_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="CSV file(s)")
    parser.add_argument("--reference", type=Path, nargs="+", help="One reference CSV per input")
    parser.add_argument("--x", help="Coordinate column; required with --y for custom CSVs")
    parser.add_argument("--y", help="Value column")
    parser.add_argument("--yerr", help="Symmetric error column; blanks in every row omit the band")
    parser.add_argument("--reference-x", help="Coordinate column in custom reference CSVs")
    parser.add_argument("--reference-y", help="Value column in custom reference CSVs")
    parser.add_argument(
        "--panels", action="store_true", help="One panel per input; default overlays"
    )
    parser.add_argument("--labels", nargs="+", help="One quoted label per input")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    parser.add_argument("--title")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--output", type=Path, required=True, help="PNG, PDF or SVG path")
    args = parser.parse_args()
    curves = [read_csv_curve(p, x=args.x, y=args.y, yerr=args.yerr) for p in args.input]
    references = (
        [read_csv_curve(p, x=args.reference_x, y=args.reference_y) for p in args.reference]
        if args.reference
        else None
    )
    output = plot_csv_curves(
        curves,
        references=references,
        panels=args.panels,
        labels=args.labels,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        xlim=tuple(args.xlim) if args.xlim else None,
        output=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
