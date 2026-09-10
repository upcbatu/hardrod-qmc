from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from hrdmc.production.variational import VariationalRunControls, run_variational_workflow
from hrdmc.sampling.vmc.transitions import VMCConfig
from hrdmc.system.guide_selection import select_guide
from hrdmc.system.settings import parse_case

ROOT = Path(__file__).resolve().parents[2]


def test_physical_forward_times_survive_a_timestep_override(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run.py",
            "dmc",
            "--preset",
            "thesis",
            "--case",
            "N10_A0.1",
            "--dt",
            "0.00125",
            "--density-fw-times",
            "0,2,4,7",
            "--output",
            str(tmp_path / "planned"),
            "--dry-run",
            "--verbose-json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["density_forward_times"] == [0, 2, 4, 7]
    assert plan["density_lag_steps"] == [0, 1600, 3200, 5600]
    assert not (tmp_path / "planned").exists()


def test_vmc_preserves_seed_identity_and_particle_accounting(tmp_path: Path) -> None:
    case = parse_case("N2_A0")
    controls = VariationalRunControls(
        VMCConfig(8, 4, 16, method="mala", dt=0.01),
        block_size=4,
        bins=16,
        grid_extent=1,
    )
    output = tmp_path / "vmc"
    summary = run_variational_workflow(
        case,
        select_guide(case),
        controls,
        [17, 29],
        workers=1,
        output=output,
        command=[],
    )
    assert summary["seeds"] == [17, 29]
    for seed in (17, 29):
        row = json.loads((output / f"seed_{seed}.json").read_text())
        assert row["seed"] == seed
        density = row["density"]
        mass = np.sum(np.asarray(density["value"]) * np.diff(density["bin_edges"]))
        assert np.isclose(mass + density["out_of_grid_mass"], 2.0)
