from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from hrdmc.statistics.population_bound import PopulationEnergyPoint
from hrdmc.statistics.population_fit import analyze_population_ladder
from hrdmc.uncertainty.population import _require_common_identity


def _point(walkers: int) -> PopulationEnergyPoint:
    return PopulationEnergyPoint(
        walkers=walkers,
        energy=5.0,
        conservative_stderr=1.0e-4,
        seed_ids=(1, 2, 3),
        seed_energies=np.asarray([4.9999, 5.0, 5.0001]),
    )


def test_population_ladder_rejects_nonwalker_control_changes() -> None:
    common = {"dt": 0.01, "drift_limiter": "umrigar", "relative_alpha": 1.5}
    first = SimpleNamespace(
        case_id="N10_A1", dt=0.01, controls={**common, "walkers": 256, "production_tau": 480.0}
    )
    second = SimpleNamespace(
        case_id="N10_A1", dt=0.01, controls={**common, "walkers": 512, "production_tau": 481.0}
    )

    with pytest.raises(ValueError, match="vary only walker count"):
        _require_common_identity(cast(Any, (first, second)))


def test_population_ladder_requires_exact_doublings() -> None:
    with pytest.raises(ValueError, match="walker-count doublings"):
        analyze_population_ladder([_point(256), _point(768)], reporting_resolution=0.01)
