from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.style import load_pyplot, save_figure


def write_claim_matrix_plot(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Render an observable-by-case claim matrix from assembled case rows."""

    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    if not rows:
        return []
    observables = ("energy", "r2", "rms", "density")
    cases = [str(row.get("case_id", index)) for index, row in enumerate(rows)]
    matrix = np.zeros((len(observables), len(cases)), dtype=float)
    labels: list[list[str]] = []
    for obs_i, observable in enumerate(observables):
        label_row: list[str] = []
        for case_i, row in enumerate(rows):
            status = str(row.get(f"{observable}_status", row.get("status", "")))
            matrix[obs_i, case_i] = _status_score(status)
            label_row.append(_short_status(status))
        labels.append(label_row)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(cases)), 3.8))
    image = ax.imshow(matrix, vmin=0.0, vmax=2.0, cmap=_claim_cmap(plt))
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels(cases, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(observables)))
    ax.set_yticklabels(observables)
    for i in range(len(observables)):
        for j in range(len(cases)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7)
    ax.set_title("Claim matrix")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    paths = save_figure(fig, plot_dir / "claim_matrix", formats)
    plt.close(fig)
    return [str(path.relative_to(output)) for path in paths]


def _status_score(status: str) -> float:
    if "GO" in status and "NO_GO" not in status:
        return 2.0
    if "WARNING" in status or "BIAS" in status:
        return 1.0
    return 0.0


def _short_status(status: str) -> str:
    if "GO" in status and "NO_GO" not in status:
        return "GO"
    if "WARNING" in status:
        return "WARN"
    if "NOT_EVALUATED" in status:
        return "N/A"
    return "NO-GO"


def _claim_cmap(plt: Any) -> Any:  # noqa: ANN401
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "hrdmc_claim",
        [tokens.METHODOLOGY_NO_GO, tokens.PRECISION_WARN, tokens.METHODOLOGY_GO],
    )
