from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.manifest import ensure_dir, write_csv, write_json, write_run_manifest


def write_stationarity_grid_artifacts(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    plots: bool,
    command: list[str] | None,
) -> dict[str, Path]:
    """Persist one stationarity grid and bind its outputs in a run manifest."""
    root = ensure_dir(output_dir)
    summary_path = root / "summary.json"
    write_json(summary_path, payload)
    case_table_path = _write_case_table(root, rows)
    plot_paths: list[str] = []
    if plots:
        plot_paths = write_plots(root, rows)
        payload["plots"] = plot_paths
        write_json(summary_path, payload)
    manifest_artifacts = [summary_path, case_table_path]
    manifest_artifacts.extend(root / path for path in plot_paths)
    manifest_path = write_run_manifest(
        root,
        run_name="dmc_trapped_stationarity_grid",
        config=config,
        artifacts=manifest_artifacts,
        status=str(payload["status"]),
    )
    return {
        "summary": summary_path,
        "case_table": case_table_path,
        "run_manifest": manifest_path,
    }
def _write_case_table(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("case table requires at least one row")
    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        )
    )
    output_path = ensure_dir(output_dir) / "case_table.csv"
    return write_csv(output_path, rows, fieldnames=fields)
def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plt = load_pyplot(output_dir)
    plot_dir = ensure_dir(output_dir / "plots")
    plot_paths = [
        _plot_metric_bars(
            plt,
            rows,
            ["rhat_energy", "rhat_rms", "rhat_r2"],
            "R-hat",
            plot_dir / "rhat_by_case.png",
            reference=1.05,
        ),
        _plot_metric_bars(
            plt,
            rows,
            ["neff_energy", "neff_rms", "neff_r2"],
            "minimum effective independent samples",
            plot_dir / "neff_by_case.png",
            reference=30.0,
        ),
        _plot_metric_bars(
            plt,
            rows,
            ["density_relative_l2"],
            "relative density L2",
            plot_dir / "density_l2_by_case.png",
        ),
        _plot_metric_bars(
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
def _plot_metric_bars(
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
