from __future__ import annotations

import pytest

from hrdmc.estimators.energy_response import PairedEnergyResponsePoint, paired_seed_trap_r2
from hrdmc.system.geometry import lambda_from_relative_offset


def _points() -> tuple[PairedEnergyResponsePoint, ...]:
    return tuple(
        PairedEnergyResponsePoint(
            seed=1,
            relative_lambda_offset=offset,
            lambda_value=lambda_from_relative_offset(offset),
            energy=10.0 + 3.0 * offset,
        )
        for offset in (-0.005, -0.0025, 0.0, 0.0025, 0.005)
    )


def test_energy_response_requires_the_complete_five_point_ladder() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        paired_seed_trap_r2(_points()[:-1], n_particles=20)


def test_energy_response_rejects_a_lambda_that_disagrees_with_its_offset() -> None:
    points = list(_points())
    points[0] = PairedEnergyResponsePoint(
        seed=1,
        relative_lambda_offset=-0.005,
        lambda_value=123.0,
        energy=points[0].energy,
    )

    with pytest.raises(ValueError, match="does not match"):
        paired_seed_trap_r2(tuple(points), n_particles=20)
