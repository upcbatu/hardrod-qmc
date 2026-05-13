from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_pyplot(output_dir: str | Path | None = None) -> Any:  # noqa: ANN401
    """Load Matplotlib with a non-interactive backend and stable rcParams."""

    if output_dir is not None:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(output_dir) / "mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise RuntimeError(
            "matplotlib is unavailable for plotting in the current environment"
        ) from exc
    apply_thesis_style(plt)
    return plt


def apply_thesis_style(plt: Any) -> None:  # noqa: ANN401
    """Apply a compact static style for thesis and artifact figures."""

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D8D8D8",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.8,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(
    fig: Any,  # noqa: ANN401
    base_path: str | Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Save a figure to every requested format and close it."""

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        path = base.parent / f"{base.name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    return paths
