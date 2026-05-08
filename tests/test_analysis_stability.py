from __future__ import annotations

import numpy as np
import pytest

from hrdmc.analysis import summarize_replicate_metrics


def test_summarize_replicate_metrics_reports_spread() -> None:
    rows = [
        {
            "density_l2_error_vmc_vs_lda": 0.2,
            "relative_density_l2_error_vmc_vs_lda": 0.1,
            "sampled_rms_radius": 1.0,
            "rms_radius_error_vmc_vs_lda": -0.1,
            "acceptance_rate": 0.8,
        },
        {
            "density_l2_error_vmc_vs_lda": 0.4,
            "relative_density_l2_error_vmc_vs_lda": 0.2,
            "sampled_rms_radius": 2.0,
            "rms_radius_error_vmc_vs_lda": 0.0,
            "acceptance_rate": 0.9,
        },
        {
            "density_l2_error_vmc_vs_lda": 0.6,
            "relative_density_l2_error_vmc_vs_lda": 0.3,
            "sampled_rms_radius": 3.0,
            "rms_radius_error_vmc_vs_lda": 0.1,
            "acceptance_rate": 1.0,
        },
    ]
    summary = summarize_replicate_metrics(
        rows,
        [
            "density_l2_error_vmc_vs_lda",
            "relative_density_l2_error_vmc_vs_lda",
            "sampled_rms_radius",
            "rms_radius_error_vmc_vs_lda",
            "acceptance_rate",
        ],
    )

    density_summary = summary["density_l2_error_vmc_vs_lda"]
    assert density_summary["count"] == 3
    np.testing.assert_allclose(density_summary["mean"], 0.4)
    np.testing.assert_allclose(density_summary["sample_std"], 0.2)
    np.testing.assert_allclose(density_summary["spread"], 0.4)
    np.testing.assert_allclose(summary["relative_density_l2_error_vmc_vs_lda"]["mean"], 0.2)
    np.testing.assert_allclose(summary["sampled_rms_radius"]["mean"], 2.0)


def test_summarize_replicate_metrics_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one replicate"):
        summarize_replicate_metrics([], ["acceptance_rate"])
