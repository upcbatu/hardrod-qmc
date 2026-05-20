from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.numerics import finite_float


def draw_energy_stationarity_panel(
    ax_trace: Any,  # noqa: ANN401
    ax_blocks: Any,  # noqa: ANN401
    payload: dict[str, Any],
) -> None:
    stationarity = payload.get("stationarity", {})
    mean_energy = finite_float(stationarity.get("mixed_energy"))
    trace_drawn = _draw_running_means(ax_trace, stationarity, mean_energy)
    if not trace_drawn:
        ax_trace.text(
            0.5,
            0.5,
            "energy trace unavailable",
            transform=ax_trace.transAxes,
            ha="center",
        )
        ax_trace.set_axis_off()
    _draw_block_residuals(ax_blocks, payload)


def _draw_running_means(
    ax: Any,  # noqa: ANN401
    stationarity: dict[str, Any],
    mean_energy: float,
) -> bool:
    paths = stationarity.get("trace_artifacts", [])
    if not isinstance(paths, list):
        return False
    trace_paths = [
        Path(path)
        for path in paths
        if isinstance(path, str) and path.endswith("_trace.csv")
    ]
    if not trace_paths:
        return False
    drawn = False
    for path in trace_paths:
        if not path.exists():
            continue
        data = np.genfromtxt(path, delimiter=",", names=True)
        names = data.dtype.names
        if data.size == 0 or names is None or "tau" not in names or "mixed_energy" not in names:
            continue
        tau = np.atleast_1d(np.asarray(data["tau"], dtype=float))
        energy = np.atleast_1d(np.asarray(data["mixed_energy"], dtype=float))
        finite = np.isfinite(tau) & np.isfinite(energy)
        tau = tau[finite]
        energy = energy[finite]
        if tau.size == 0:
            continue
        running = np.cumsum(energy) / np.arange(1, energy.size + 1, dtype=float)
        step = max(1, tau.size // 900)
        seed = _seed_from_trace_path(path)
        ax.plot(
            tau[::step],
            running[::step],
            linewidth=1.0,
            alpha=0.34,
            label=f"seed {seed}" if seed else None,
        )
        drawn = True
    if np.isfinite(mean_energy):
        ax.axhline(
            mean_energy,
            color=tokens.INK,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="final mean",
        )
    ax.set_title("Energy running means", loc="left", pad=4)
    ax.set_ylabel(r"$E$")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="best", fontsize=7, ncol=2)
    return drawn


def _draw_block_residuals(
    ax: Any,  # noqa: ANN401
    payload: dict[str, Any],
) -> None:
    stationarity = payload.get("stationarity", {})
    diagnostics = stationarity.get("diagnostics", {}).get("energy", {})
    chain_rows = diagnostics.get("chain_diagnostics", [])
    seeds = stationarity.get("seeds", [])
    if not isinstance(chain_rows, list) or not chain_rows:
        ax.text(
            0.5,
            0.5,
            "energy block diagnostics unavailable",
            transform=ax.transAxes,
            ha="center",
        )
        ax.set_axis_off()
        return
    production_tau = finite_float(payload.get("controls", {}).get("production_tau"))
    for index, row in enumerate(chain_rows):
        if not isinstance(row, dict):
            continue
        block_means = np.asarray(row.get("block_means", []), dtype=float)
        block_means = block_means[np.isfinite(block_means)]
        if block_means.size == 0:
            continue
        # The stationarity spread metric is seed-internal, so center the
        # visual residuals by each seed's own block mean.
        reference = float(np.mean(block_means))
        x = np.arange(1, block_means.size + 1, dtype=float)
        if np.isfinite(production_tau):
            x = (x - 0.5) * production_tau / block_means.size
        seed = seeds[index] if isinstance(seeds, list) and index < len(seeds) else ""
        warning = bool(row.get("spread_warning", False))
        ax.plot(
            x,
            block_means - reference,
            marker="o",
            linewidth=1.7 if warning else 1.0,
            alpha=0.92 if warning else 0.48,
            label=f"seed {seed}" if seed else None,
        )
    ax.axhline(0.0, color=tokens.INK, linestyle=(0, (4, 2)), linewidth=0.9)
    ax.set_title("Energy block residuals", loc="left", pad=4)
    ax.set_xlabel("production time")
    ax.set_ylabel(r"$E_{\mathrm{block}}-\bar E_{\mathrm{seed}}$")
    ax.legend(loc="best", fontsize=7, ncol=3)


def _seed_from_trace_path(path: Path) -> str:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        return ""
    return stem.split(marker, maxsplit=1)[1].split("_", maxsplit=1)[0]
