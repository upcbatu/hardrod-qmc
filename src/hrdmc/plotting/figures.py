from __future__ import annotations

from pathlib import Path

from hrdmc.estimators.pair_distribution import PairDistributionResult
from hrdmc.estimators.structure_factor import StructureFactorResult


def _require_matplotlib():  # noqa: ANN202
    try:
        import matplotlib.pyplot as plt
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise RuntimeError(
            "matplotlib is unavailable for plotting in the current environment. "
            "Install project dependencies with a writable cache directory or run the "
            "estimators without plotting."
        ) from exc
    return plt


def plot_pair_distribution(result: PairDistributionResult, path: str | Path) -> None:
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.r, result.g_r, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Pair distribution function")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_structure_factor(result: StructureFactorResult, path: str | Path) -> None:
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(result.k, result.s_k, yerr=result.stderr, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("k")
    ax.set_ylabel("S(k)")
    ax.set_title("Static structure factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
