from __future__ import annotations

import pytest

from hrdmc.production.matrix.method import DEFAULT_GUIDE_VALIDATION_SUMMARY, row_method
from hrdmc.system.guide_registry import load_validated_reduced_tg_guide
from hrdmc.system.settings import parse_case

GUIDES = {
    "N10_A0.1": 1.0637325870622627,
    "N10_A1": 1.6224444406063525,
    "N10_A10": 5.5908651157560385,
    "N20_A0.1": 1.0908094794241916,
    "N20_A1": 1.8669363227063642,
    "N20_A10": 7.011111084682286,
}


@pytest.mark.parametrize(("case_id", "alpha"), GUIDES.items())
def test_registry_binds_the_six_thesis_widths(case_id: str, alpha: float) -> None:
    artifact = load_validated_reduced_tg_guide(
        DEFAULT_GUIDE_VALIDATION_SUMMARY, case=parse_case(case_id)
    )

    assert artifact.relative_alpha == alpha


@pytest.mark.parametrize("case_id", ["N10_A0", "N20_A0"])
def test_zero_diameter_rows_use_the_reduced_tg_default(case_id: str) -> None:
    method = row_method(case_id, guide_validation_root=None)

    assert method.relative_alpha is None
    assert method.drift_limiter == "none"
