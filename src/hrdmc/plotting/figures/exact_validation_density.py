from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.figures.exact_validation_common import (
    disable_layout_engine,
    draw_packet_header,
)
from hrdmc.plotting.style import save_figure
from hrdmc.theory import (
    trapped_tg_density_profile,
    trapped_tg_density_profile_semiclassical,
)


def write_trapped_tg_density_plots(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    paths: list[Path] = []
    for anchor in payload.get("trapped_tg_anchors", []):
        density = anchor.get("density_profile", {})
        x = np.asarray(density.get("x", []), dtype=float)
        exact = np.asarray(density.get("exact_n_x", []), dtype=float)
        exact_bin = np.asarray(density.get("exact_bin_averaged_n_x", []), dtype=float)
        bin_edges = np.asarray(density.get("bin_edges", []), dtype=float)
        mixed = np.asarray(density.get("mixed_n_x", []), dtype=float)
        pure = np.asarray(density.get("pure_fw_n_x", []), dtype=float)
        if x.size == 0 or exact.size != x.size:
            continue
        paths.extend(
            _write_single_density_plot(
                plt,
                plot_dir,
                payload,
                anchor,
                x=x,
                exact=exact,
                exact_bin=exact_bin,
                bin_edges=bin_edges,
                mixed=mixed,
                pure=pure,
                formats=formats,
            )
        )
    return paths


def write_tg_density_limit_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    omegas = _trapped_anchor_omegas(payload)
    if not omegas:
        return []
    n_values = (2, 4, 8, 16)
    fig, axes = plt.subplots(
        1,
        len(omegas),
        figsize=(6.2 * len(omegas), 4.5),
        squeeze=False,
        constrained_layout=False,
    )
    disable_layout_engine(fig)
    draw_packet_header(fig, payload)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.055, right=0.985, wspace=0.25)
    fig.suptitle(
        r"Trapped TG density: finite $N$ exact curves vs $N\to\infty$ LDA limit",
        y=0.965,
        fontsize=12,
        fontweight="semibold",
    )
    for ax, omega in zip(axes[0], omegas, strict=True):
        _draw_tg_density_limit_panel(ax, omega=omega, n_values=n_values)
    paths = save_figure(fig, plot_dir / "exact_tg_density_large_n_limit", formats)
    plt.close(fig)
    return paths


def _write_single_density_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    anchor: dict[str, Any],
    *,
    x: np.ndarray,
    exact: np.ndarray,
    exact_bin: np.ndarray,
    bin_edges: np.ndarray,
    mixed: np.ndarray,
    pure: np.ndarray,
    formats: tuple[str, ...],
) -> list[Path]:
    exact_plot_x, exact_plot_y = _dense_trapped_tg_density_from_anchor(anchor, x, exact)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        sharex=True,
        constrained_layout=False,
    )
    ax = axes[0]
    residual_ax = axes[1]
    disable_layout_engine(fig)
    draw_packet_header(fig, payload)
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.08)
    ax.plot(
        exact_plot_x,
        exact_plot_y,
        color=tokens.LDA_PREDICTION,
        linewidth=1.15,
        alpha=0.55,
        label="exact TG smooth context",
    )
    if exact_bin.size == x.size:
        _plot_binned_density(
            ax,
            x,
            exact_bin,
            bin_edges=bin_edges,
            color=tokens.LDA_PREDICTION,
            linestyle=(0, (5, 2)),
            linewidth=1.7,
            label="exact TG bin avg",
            zorder=2,
        )
    if mixed.size == x.size:
        _plot_binned_density(
            ax,
            x,
            mixed,
            bin_edges=bin_edges,
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            linewidth=1.9,
            label="mixed diagnostic",
            zorder=4,
        )
    if pure.size == x.size:
        _plot_binned_density(
            ax,
            x,
            pure,
            bin_edges=bin_edges,
            color=tokens.DMC_PRIMARY,
            linestyle="-",
            linewidth=2.1,
            label="transported FW",
            zorder=3,
        )
    ax.text(
        0.01,
        0.02,
        "FW/mixed/exact-bin are plotted as histogram-bin values; smooth exact is context only.",
        transform=ax.transAxes,
        fontsize=7,
        color=tokens.INK_SOFT,
    )
    _draw_density_residuals(
        residual_ax,
        x,
        exact_bin=exact_bin,
        bin_edges=bin_edges,
        mixed=mixed,
        pure=pure,
    )
    title = _density_title(anchor)
    ax.set_title(title, pad=10)
    ax.set_ylabel(r"$n(x)$")
    residual_ax.set_xlabel(r"$x$")
    residual_ax.set_ylabel(r"$\Delta n(x)$")
    ax.legend(loc="best", fontsize=8)
    filename = f"exact_density_{anchor.get('anchor_id', 'trapped_tg')}"
    paths = save_figure(fig, plot_dir / filename, formats)
    plt.close(fig)
    return paths


def _draw_density_residuals(
    ax: Any,  # noqa: ANN401
    x: np.ndarray,
    *,
    exact_bin: np.ndarray,
    bin_edges: np.ndarray,
    mixed: np.ndarray,
    pure: np.ndarray,
) -> None:
    if exact_bin.size != x.size:
        return
    if pure.size == x.size:
        _plot_binned_density(
            ax,
            x,
            pure - exact_bin,
            bin_edges=bin_edges,
            color=tokens.DMC_PRIMARY,
            linestyle="-",
            linewidth=1.5,
            label="FW - exact bin avg",
        )
    if mixed.size == x.size:
        _plot_binned_density(
            ax,
            x,
            mixed - exact_bin,
            bin_edges=bin_edges,
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            linewidth=1.5,
            alpha=0.95,
            label="mixed - exact bin avg",
        )
    ax.axhline(0.0, color=tokens.INK_SOFT, linewidth=0.8)
    ax.legend(loc="best", fontsize=7)


def _plot_binned_density(
    ax: Any,  # noqa: ANN401
    x: np.ndarray,
    values: np.ndarray,
    *,
    bin_edges: np.ndarray,
    color: str,
    linestyle: object,
    linewidth: float,
    label: str,
    alpha: float = 1.0,
    zorder: int | None = None,
) -> None:
    if bin_edges.size == values.size + 1:
        ax.stairs(
            values,
            bin_edges,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            alpha=alpha,
            zorder=zorder,
        )
        ax.plot(
            x,
            values,
            marker="o",
            linestyle="None",
            markersize=2.2,
            color=color,
            alpha=0.7 * alpha,
            zorder=zorder,
        )
        return
    ax.step(
        x,
        values,
        where="mid",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        alpha=alpha,
        zorder=zorder,
    )


def _density_title(anchor: dict[str, Any]) -> str:
    comparison = anchor.get("exact_comparison", {})
    density_l2 = comparison.get("pure_density_relative_l2")
    title = f"{anchor.get('anchor_id', 'trapped TG')} density"
    if density_l2 is not None:
        title += f"  |  FW L2={float(density_l2):.3g}"
    return title


def _dense_trapped_tg_density_from_anchor(
    anchor: dict[str, Any],
    x: np.ndarray,
    fallback_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    exact_solution = anchor.get("exact_solution", {})
    try:
        n_particles = int(exact_solution.get("n_particles"))
        omega = float(exact_solution.get("omega"))
    except (TypeError, ValueError):
        return x, fallback_density
    if not np.isfinite(omega) or x.size < 2:
        return x, fallback_density
    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), max(900, 20 * x.size))
    dense_density = trapped_tg_density_profile(
        dense_x,
        n_particles=n_particles,
        omega=omega,
    )
    return dense_x, dense_density


def _draw_tg_density_limit_panel(
    ax: Any,  # noqa: ANN401
    *,
    omega: float,
    n_values: tuple[int, ...],
) -> None:
    max_support = float(np.sqrt(2.0 * max(n_values) * np.sqrt(2.0) / omega))
    x = np.linspace(-1.12 * max_support, 1.12 * max_support, 900)
    colors = ("#0072B2", "#D55E00", "#117733", "#CC79A7")
    for color, n_particles in zip(colors, n_values, strict=True):
        finite_n = trapped_tg_density_profile(x, n_particles=n_particles, omega=omega)
        large_n = trapped_tg_density_profile_semiclassical(
            x,
            n_particles=n_particles,
            omega=omega,
        )
        ax.plot(
            x,
            finite_n,
            color=color,
            linewidth=1.55,
            label=f"N={n_particles} finite",
        )
        ax.plot(
            x,
            large_n,
            color=color,
            linestyle=(0, (4, 2)),
            linewidth=1.05,
            alpha=0.85,
            label=f"N={n_particles} LDA limit",
        )
    ax.set_title(f"omega={omega:g}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$n(x)$")
    ax.legend(loc="upper right", fontsize=7.1, ncol=2)


def _trapped_anchor_omegas(payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for anchor in payload.get("trapped_tg_anchors", []):
        exact = anchor.get("exact_solution", {})
        try:
            omega = float(exact.get("omega"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(omega) and omega not in values:
            values.append(omega)
    return values
