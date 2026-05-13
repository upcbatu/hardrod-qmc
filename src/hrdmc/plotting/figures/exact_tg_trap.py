from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.style import load_pyplot, save_figure


def write_exact_tg_trap_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Render exact trapped TG validation plots."""

    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    paths.extend(_write_energy_plot(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_density_plot(plt, plot_dir, payload, formats=formats))
    return [str(path.relative_to(output)) for path in paths]


def _write_energy_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    exact = payload.get("exact_solution", {})
    seed_rows = payload.get("seed_summaries", [])
    exact_energy = float(exact.get("energy_total", np.nan))
    seed_values = np.asarray([row.get("mixed_energy", np.nan) for row in seed_rows], dtype=float)
    seed_values = seed_values[np.isfinite(seed_values)]
    mean_energy = float(payload.get("mixed_energy", np.nan))
    stderr = float(payload.get("mixed_energy_seed_stderr", np.nan))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    _draw_exact_header(fig, payload)
    ax.axhline(
        exact_energy,
        color=tokens.LDA_PREDICTION,
        linestyle=(0, (4, 2)),
        linewidth=1.35,
        label="exact TG energy",
    )
    if seed_values.size:
        xs = np.arange(seed_values.size, dtype=float)
        ax.scatter(xs, seed_values, color=tokens.DMC_DIAGNOSTIC, s=28, label="seed energies")
        ax.errorbar(
            [xs.size],
            [mean_energy],
            yerr=[stderr] if np.isfinite(stderr) else None,
            fmt="o",
            color=tokens.DMC_PRIMARY,
            capsize=4,
            label="RN-DMC mean",
        )
        ax.set_xlim(-0.7, xs.size + 0.7)
        ax.set_xticks([*xs, xs.size])
        ax.set_xticklabels([str(i + 1) for i in range(seed_values.size)] + ["mean"])
    ax.set_title("Exact trapped TG energy anchor")
    ax.set_ylabel(r"$E$")
    ax.text(
        0.02,
        0.05,
        f"abs error={payload.get('absolute_energy_error', float('nan')):.3g}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    ax.legend(loc="best", fontsize=8)
    paths = save_figure(fig, plot_dir / "exact_tg_energy_comparison", formats)
    plt.close(fig)
    return paths


def _write_density_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    exact = payload.get("exact_solution", {})
    density = payload.get("density_profile", {})
    exact_x = np.asarray(exact.get("density_x", []), dtype=float)
    exact_n = np.asarray(exact.get("density_n_x", []), dtype=float)
    mixed_x = np.asarray(density.get("x", []), dtype=float)
    mixed_n = np.asarray(density.get("mixed_n_x", []), dtype=float)
    if exact_x.size == 0 or exact_n.size == 0:
        return []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    _draw_exact_header(fig, payload)
    ax.plot(
        exact_x,
        exact_n,
        color=tokens.LDA_PREDICTION,
        linestyle=(0, (4, 2)),
        linewidth=1.45,
        label="exact TG density",
    )
    if mixed_x.size == mixed_n.size and mixed_x.size > 0:
        ax.plot(
            mixed_x,
            mixed_n,
            color=tokens.DMC_PRIMARY,
            linewidth=1.85,
            label="RN-DMC mixed density diagnostic",
        )
    ax.set_title("Exact trapped TG density anchor")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$n(x)$")
    ax.text(
        0.02,
        0.05,
        "a=0 exact TG validation; density is diagnostic unless estimator gates request it",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    ax.legend(loc="best", fontsize=8)
    paths = save_figure(fig, plot_dir / "exact_tg_density_comparison", formats)
    plt.close(fig)
    return paths


def _draw_exact_header(fig: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    exact = payload.get("exact_solution", {})
    controls = payload.get("controls", {})
    text = (
        f"exact TG anchor  N={exact.get('n_particles', '?')}  a=0  "
        f"omega={exact.get('omega', '?')}  dt={controls.get('dt', '?')}  "
        f"M={controls.get('walkers', '?')}  seeds={payload.get('seed_count', '?')}  "
        f"prod_tau={controls.get('production_tau', '?')}"
    )
    fig.suptitle(
        text,
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color=tokens.INK_SOFT,
    )
