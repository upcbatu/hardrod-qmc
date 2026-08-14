from __future__ import annotations

import json
from pathlib import Path

from hrdmc.artifacts.manifest import verify_run_manifest

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "results/dmc/final_matrix/thesis_5seed_all_optimized_final_v1"
EXPECTED = {
    "N10_A0": (
        50.0,
        0.0,
        2.2361402373074997,
        0.00018593638199315568,
        0.04842729065529805,
        0.0025,
        256,
        None,
        None,
    ),
    "N10_A0.1": (
        56.64321333257236,
        3.380629882281105e-05,
        2.45164482991746,
        0.00016400178627944836,
        0.07408011942984905,
        0.0025,
        256,
        1.0637325870622627,
        "umrigar",
    ),
    "N10_A1": (
        146.8372648745057,
        0.00023482824821975792,
        4.614753526972409,
        8.778529396098835e-05,
        0.6729133865423078,
        0.00125,
        512,
        1.6224444406063525,
        "umrigar",
    ),
    "N10_A10": (
        4535.876057509662,
        0.0008664468559287866,
        29.656000786478522,
        0.00013381008036837176,
        3.193532199887779,
        0.000125,
        256,
        5.5908651157560385,
        "umrigar",
    ),
    "N20_A0": (
        200.0,
        0.0,
        3.162151144374758,
        0.00017708571291757678,
        0.029681026034966203,
        0.0025,
        256,
        None,
        None,
    ),
    "N20_A0.1": (
        238.80537962459675,
        7.88659084338527e-05,
        3.602055399411624,
        0.0002771469213549209,
        0.05557267351505521,
        0.0025,
        256,
        1.0908094794241916,
        "umrigar",
    ),
    "N20_A1": (
        838.0703893870741,
        0.0003306746369573825,
        8.072367854727412,
        0.00015311435428234834,
        0.8373490697960493,
        0.000625,
        512,
        1.8669363227063642,
        "umrigar",
    ),
    "N20_A10": (
        35333.61359597035,
        0.0010045971494192765,
        58.85079047515964,
        0.00010417890693022466,
        3.7753375208256585,
        0.00025,
        512,
        7.011111084682286,
        "umrigar",
    ),
}


def test_compact_matrix_manifest_and_manuscript_fields_are_frozen() -> None:
    valid, errors = verify_run_manifest(MATRIX / "run_manifest.json")
    rows = json.loads((MATRIX / "final_matrix_summary.json").read_text(encoding="utf-8"))["rows"]

    assert valid, errors
    assert {
        row["case"]: (
            row["energy"],
            row["energy_stderr"],
            row["rms_radius"],
            row["rms_mc_statistical_stderr"],
            row["density_fw_relative_l2_vs_lda"],
            row["dt"],
            row["walkers"],
            row["relative_alpha"],
            row["drift_limiter"],
        )
        for row in rows
    } == EXPECTED
