from __future__ import annotations

import math

import numpy as np
import pytest

from hrdmc.sampling.dmc.run import run_streaming_seed
from hrdmc.system.settings import DMCRunControls, parse_case


@pytest.mark.parametrize(
    ("case_id", "controls", "expected"),
    [
        (
            "N10_A0",
            DMCRunControls(0.0025, 64, 0.5, 1.0, 10, 12.0, 240),
            (50.0, 6.21019167694895, 59.75, 1.35837890625),
        ),
        (
            "N10_A0.1",
            DMCRunControls(
                0.0025,
                64,
                0.5,
                1.0,
                10,
                20.0,
                240,
                drift_limiter="umrigar",
                relative_alpha=1.0637325870622627,
            ),
            (56.61405251917114, 7.045090933536422, 59.75, 1.3141958361459367),
        ),
    ],
)
def test_fixed_seed_dmc_preserves_thesis_transition_paths(
    case_id: str,
    controls: DMCRunControls,
    expected: tuple[float, float, float, float],
) -> None:
    summary = run_streaming_seed(
        parse_case(case_id), controls, seed=7001, guide_family="reduced-tg"
    )
    observed = (
        float(summary.mixed_energy),
        float(summary.r2_radius),
        float(np.sum(summary.density)),
        float(np.max(summary.density)),
    )

    for value, reference in zip(observed, expected, strict=True):
        assert math.isclose(value, reference, rel_tol=1.0e-9, abs_tol=0.0)
