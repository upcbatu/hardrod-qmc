from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "rn_results_manifest.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_rn_results_manifest_points_to_compact_tables() -> None:
    payload = json.loads(MANIFEST.read_text())

    assert payload["schema_version"] == "rn_results_manifest_v1"
    assert "full run artifacts are archived separately" in payload["claim_boundary"]

    for table_path in payload["tables"]:
        path = ROOT / table_path
        assert path.is_file(), table_path
        assert path.suffix == ".csv"


def test_rn_validation_summary_is_exact_homogeneous_gate_material() -> None:
    rows = _read_csv(ROOT / "docs" / "tables" / "rn_validation_summary.csv")

    assert rows
    assert {
        "n_particles",
        "packing_fraction",
        "dt",
        "walkers",
        "mean_abs_error",
        "mean_relative_error",
        "valid_snapshot_fraction_min",
        "finite_local_energy_fraction_min",
        "zero_weight_excluded_count_total",
        "invalid_excluded_count_total",
        "max_mixed_energy_blocking_stderr",
    } == set(rows[0])

    for row in rows:
        assert float(row["valid_snapshot_fraction_min"]) == 1.0
        assert float(row["finite_local_energy_fraction_min"]) == 1.0
        assert int(row["zero_weight_excluded_count_total"]) == 0
        assert int(row["invalid_excluded_count_total"]) == 0
        assert float(row["mean_relative_error"]) < 0.01


def test_rn_grid_summary_is_compact_and_honest_about_warnings() -> None:
    rows = _read_csv(ROOT / "docs" / "tables" / "rn_grid_summary.csv")

    assert {row["campaign"] for row in rows} == {"six_case_10seed", "frontier_eta030"}
    assert {row["case_gate"] for row in rows} == {"False", "True"}

    for row in rows:
        assert row["density_accounting_clean"] == "True"
        assert row["valid_finite_clean"] == "True"
        assert row["rn_weight_controlled"] == "True"
        assert float(row["rhat_energy"]) < 1.05
        assert float(row["rhat_rms"]) < 1.05
        assert float(row["neff_energy"]) > 30.0
        assert float(row["neff_rms"]) > 30.0


def test_rn_seed_robustness_summary_preserves_no_go_and_go_rows() -> None:
    rows = _read_csv(ROOT / "docs" / "tables" / "rn_seed_robustness_summary.csv")

    assert {row["case_id"] for row in rows} == {
        "N4_a0.5_omega0.05",
        "N4_a0.5_omega0.1",
    }
    assert {row["case_gate"] for row in rows} == {"False", "True"}
    assert all(row["campaign"] == "n4_40_new_seed_stress" for row in rows)
