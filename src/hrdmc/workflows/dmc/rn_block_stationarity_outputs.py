from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.io.artifacts import ensure_dir


def write_case_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "case_gate",
        "old_case_gate",
        "hygiene_gate",
        "classification",
        "final_classification",
        "seed_count",
        "parallel_workers",
        "mixed_energy",
        "mixed_energy_seed_stderr",
        "mixed_energy_blocking_stderr",
        "mixed_energy_conservative_stderr",
        "mixed_energy_uncertainty_status",
        "rms_radius",
        "rms_radius_seed_stderr",
        "rms_radius_blocking_stderr",
        "rms_radius_conservative_stderr",
        "rms_radius_uncertainty_status",
        "density_relative_l2",
        "density_relative_l2_seed_stderr",
        "uncertainty_status",
        "max_spread_blocking_z",
        "blocking_plateau_energy",
        "blocking_plateau_rms",
        "blocked_zscore_max_energy",
        "blocked_zscore_max_rms",
        "robust_zscore_max_energy",
        "robust_zscore_max_rms",
        "lda_total_energy",
        "energy_dmc_minus_lda",
        "lda_rms_radius",
        "rms_dmc_minus_lda",
        "density_integral",
        "density_accounting_clean",
        "valid_finite_clean",
        "rn_weight_controlled",
        "rhat_energy",
        "rhat_rms",
        "rhat_r2",
        "neff_energy",
        "neff_rms",
        "neff_r2",
        "stationarity_energy",
        "stationarity_rms",
        "stationarity_r2",
        "spread_warning_count",
        "ess_fraction_min",
        "log_weight_span_max",
        "rn_weight_status",
        "lost_out_of_grid_sample_count_total",
        "guide_batch_backend",
    ]
    output_path = ensure_dir(output_dir) / "case_table.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plt = load_pyplot(output_dir)
    plot_dir = ensure_dir(output_dir / "plots")
    plot_paths = [
        plot_metric_bars(
            plt,
            rows,
            ["rhat_energy", "rhat_rms", "rhat_r2"],
            "R-hat",
            plot_dir / "rhat_by_case.png",
            reference=1.05,
        ),
        plot_metric_bars(
            plt,
            rows,
            ["neff_energy", "neff_rms", "neff_r2"],
            "minimum effective independent samples",
            plot_dir / "neff_by_case.png",
            reference=30.0,
        ),
        plot_metric_bars(
            plt,
            rows,
            ["density_relative_l2"],
            "relative density L2",
            plot_dir / "density_l2_by_case.png",
        ),
        plot_metric_bars(
            plt,
            rows,
            ["energy_dmc_minus_lda"],
            "DMC energy minus LDA",
            plot_dir / "energy_dmc_minus_lda_by_case.png",
            reference=0.0,
        ),
    ]
    return [str(path.relative_to(output_dir)) for path in plot_paths]


def load_pyplot(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / "mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_bars(
    plt,
    rows: list[dict[str, Any]],
    fields: list[str],
    ylabel: str,
    output_path: Path,
    *,
    reference: float | None = None,
) -> Path:
    labels = [str(row["case_id"]) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = min(0.8 / len(fields), 0.35)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(labels)), 4.6), constrained_layout=True)
    for index, field in enumerate(fields):
        offset = (index - 0.5 * (len(fields) - 1)) * width
        values = [float(row[field]) for row in rows]
        ax.bar(x + offset, values, width=width, label=field)
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if len(fields) > 1:
        ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
