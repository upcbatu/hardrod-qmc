from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hrdmc.plotting.csv_curves import read_csv_curve


@pytest.mark.parametrize(
    "header",
    ["q_left,q_right,q_center,n_fw,n_fw_sem", "q_left,q_right,q,density,stderr"],
)
def test_density_columns_preserve_values_errors_and_nonuniform_bins(tmp_path, header):
    path = tmp_path / "density.csv"
    path.write_text(header + "\n-2,0,-1,3,0.25\n0,4,2,5,0.5\n")
    curve = read_csv_curve(path)
    np.testing.assert_array_equal(curve.x, [-1, 2])
    np.testing.assert_array_equal(curve.y, [3, 5])
    np.testing.assert_array_equal(curve.error, [0.25, 0.5])
    np.testing.assert_array_equal(curve.edges, [-2, 0, 4])


def test_lda_has_no_invented_errors_or_bins(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("q,n_lda\n-1,2\n2,4\n")
    curve = read_csv_curve(path)
    np.testing.assert_array_equal(curve.x, [-1, 2])
    np.testing.assert_array_equal(curve.y, [2, 4])
    assert curve.error is None
    assert curve.edges is None


def test_explicit_columns_preserve_custom_values(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("position,energy,uncertainty,unused\n0,-3,0.5,99\n2,-1,0.25,88\n")
    curve = read_csv_curve(path, x="position", y="energy", yerr="uncertainty")
    np.testing.assert_array_equal(curve.x, [0, 2])
    np.testing.assert_array_equal(curve.y, [-3, -1])
    np.testing.assert_array_equal(curve.error, [0.5, 0.25])
    assert curve.edges is None


def test_unavailable_density_errors_remain_unavailable(tmp_path):
    path = tmp_path / "density.csv"
    path.write_text("q,density,stderr\n0,3,\n1,5,\n")
    curve = read_csv_curve(path)
    np.testing.assert_array_equal(curve.y, [3, 5])
    assert curve.error is None


@pytest.mark.parametrize(
    "contents,options",
    [
        ("q,density\n0,1\n", {"x": "q", "y": "missing"}),
        ("q,density,stderr\n0,1,-0.1\n", {}),
        ("q,density\n0,nan\n", {}),
        ("q_left,q_right,q,density\n1,-1,0,2\n", {}),
        ("q_left,q_right,q,density\n0,2,1,2\n1,4,3,3\n", {}),
        ("q_left,q_right,q,density\n0,1,0.5,2\n2,3,2.5,3\n", {}),
    ],
)
def test_invalid_numeric_curves_are_rejected(tmp_path, contents, options):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_csv_curve(path, **options)


def test_public_plot_command_writes_png(tmp_path):
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "curve.csv"
    data.write_text("q,n_lda\n-1,0\n0,2\n1,0\n")
    output = tmp_path / "plots" / "curve.png"
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mplconfig")}
    result = subprocess.run(
        [
            sys.executable,
            str(root / "experiments/run.py"),
            "plot",
            "--input",
            str(data),
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
