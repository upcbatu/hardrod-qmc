from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hrdmc.estimators.energy_response import (
    PairedEnergyResponsePoint,
    paired_seed_trap_r2,
    paired_trap_r2_from_energy_response,
)

SOURCE = Path(__file__).resolve().parents[2] / "data/energy_response/N20_A10_h0025_5seed.csv"


def _points() -> tuple[PairedEnergyResponsePoint, ...]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return tuple(
            PairedEnergyResponsePoint(
                seed=int(row["seed"]),
                relative_lambda_offset=float(row["relative_lambda_offset"]),
                lambda_value=float(row["lambda_value"]),
                energy=float(row["energy"]),
            )
            for row in csv.DictReader(handle)
        )


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (9701, 3463.7724895125334),
        (9702, 3463.300594766042),
        (9703, 3463.403209248912),
        (9704, 3463.552973104282),
        (9705, 3463.3317839827214),
    ],
)
def test_each_seed_reproduces_its_richardson_radius(seed: int, expected: float) -> None:
    seed_points = tuple(point for point in _points() if point.seed == seed)
    assert paired_seed_trap_r2(seed_points, n_particles=20).richardson_r2 == pytest.approx(expected)


def test_energy_response_reproduces_the_manuscript_radius() -> None:
    result = paired_trap_r2_from_energy_response(_points(), n_particles=20)

    assert result.pure_r2 == pytest.approx(3463.472210122898)
    assert result.rms_radius == pytest.approx(58.85127194991539)
    assert result.rms_radius_seed_stderr == pytest.approx(0.0007373005273995926)
