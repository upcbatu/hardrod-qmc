from __future__ import annotations

import copy
from typing import Any

import pytest

from hrdmc.production.matrix.sources import validate_r2_supplement


def _packet(*, tree: str = "a" * 64) -> dict[str, Any]:
    return {
        "manifest": {"provenance": {"implementation": {"source_tree_sha256": tree}}},
        "summary": {
            "case_id": "N20_A10",
            "seeds": [7001, 7002],
            "n_particles": 20,
            "rod_length_ho": 10.0,
            "guide_family": "reduced-tg",
            "guide_parameters": {"relative_alpha": 7.011111084682286},
            "controls": {"dt": 0.00025, "walkers": 512},
            "estimates": {
                "energy": {"value": 35688.0, "stderr": 33.0, "status": "accepted"},
                "r2": {"value": 3463.0, "stderr": 0.1, "status": "accepted"},
            },
        },
    }


def test_r2_supplement_rejects_a_different_implementation() -> None:
    with pytest.raises(ValueError, match="different implementation"):
        validate_r2_supplement(_packet(), _packet(tree="b" * 64))


def test_r2_supplement_rejects_changed_physics_controls() -> None:
    primary = _packet()
    supplement = copy.deepcopy(primary)
    supplement["summary"]["controls"]["walkers"] = 256

    with pytest.raises(ValueError, match="disagrees on controls"):
        validate_r2_supplement(primary, supplement)
