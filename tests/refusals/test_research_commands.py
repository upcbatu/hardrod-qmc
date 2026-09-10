from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hrdmc.system.guide_registry import validate_production_reduced_tg_binding
from hrdmc.system.guide_selection import select_guide
from hrdmc.system.settings import parse_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "extra",
    [
        ["--dt", "0.003", "--density-fw-times", "0,2,4,7"],
        ["--seeds", "17,17"],
    ],
)
def test_dmc_rejects_ambiguous_replication_or_time_inputs(tmp_path: Path, extra: list[str]) -> None:
    output = tmp_path / "result"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run.py",
            "dmc",
            "--case",
            "N2_A0",
            "--output",
            str(output),
            "--dry-run",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_candidate_cannot_change_case_or_imply_registry_validation(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "reduced_tg_relative_alpha_optimization_v1",
                "case_id": "N10_A1",
                "recommended_relative_alpha": 1.6,
            }
        )
    )
    with pytest.raises(ValueError):
        select_guide(parse_case("N10_A0.1"), alpha_from=path)
    with pytest.raises(ValueError):
        select_guide(parse_case("N10_A1"), alpha=1.5, alpha_from=path)
    guide = select_guide(parse_case("N10_A1"), alpha_from=path)
    assert guide.validation == "not_independently_validated"
    binding = {
        "case": parse_case("N10_A1"),
        "guide_family": "reduced-tg",
        "relative_alpha": guide.relative_alpha,
        "source": guide.source,
    }
    with pytest.raises(ValueError):
        validate_production_reduced_tg_binding(**binding)
    validate_production_reduced_tg_binding(**binding, allow_unvalidated=True)


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("existing evidence")
    result = subprocess.run(
        [sys.executable, "experiments/run.py", "vmc", "--case", "N2_A0", "--output", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert marker.read_text() == "existing evidence"
    assert list(tmp_path.iterdir()) == [marker]
