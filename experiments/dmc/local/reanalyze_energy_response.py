from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hrdmc.artifacts.layout import repo_root_from
from hrdmc.estimators.energy_response import (
    PairedEnergyResponsePoint,
    paired_trap_r2_from_energy_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the thesis Hellmann-Feynman radius from paired seed energies."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--n-particles", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root_from(Path(__file__))
    source = args.input or root / "data" / "energy_response" / "N20_A10_h0025_5seed.csv"
    if not source.is_file():
        raise SystemExit(f"energy-response input is missing: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = tuple(
            PairedEnergyResponsePoint(
                seed=int(row["seed"]),
                relative_lambda_offset=float(row["relative_lambda_offset"]),
                lambda_value=float(row["lambda_value"]),
                energy=float(row["energy"]),
            )
            for row in csv.DictReader(handle)
        )
    result = paired_trap_r2_from_energy_response(rows, n_particles=args.n_particles)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
