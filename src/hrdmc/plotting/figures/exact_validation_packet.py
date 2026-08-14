from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.plotting.style import load_pyplot, save_figure
from hrdmc.theory.tonks_girardeau import (
    trapped_tg_density_profile,
    trapped_tg_density_profile_semiclassical,
)


def write_exact_validation_packet_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Write the exact-anchor scorecard and density diagnostics."""
    output, plot_dir = Path(output_dir), Path(output_dir) / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    anchors = [row for row in payload.get("trapped_tg_anchors", []) if isinstance(row, dict)]
    if anchors:
        figure = _scorecard(plt, anchors)
        paths.extend(save_figure(figure, plot_dir / "exact_anchor_errors", formats))
        plt.close(figure)
    for anchor in anchors:
        figure = _density(plt, anchor)
        if figure is not None:
            stem = f"exact_density_{anchor.get('anchor_id', 'trapped_tg')}"
            paths.extend(save_figure(figure, plot_dir / stem, formats))
            plt.close(figure)
    if anchors:
        figure = _large_n_limit(plt, anchors)
        paths.extend(save_figure(figure, plot_dir / "exact_tg_density_large_n_limit", formats))
        plt.close(figure)
    return [str(path.relative_to(output)) for path in paths]
def _scorecard(plt: Any, anchors: list[dict[str, Any]]) -> Any:
    labels, energy, radius, density = [], [], [], []
    for anchor in anchors:
        comparison = _mapping(anchor.get("exact_comparison"))
        labels.append(str(anchor.get("anchor_id", "")))
        energy.append(_number(anchor.get("absolute_energy_error")))
        radius.append(_number(comparison.get("pure_r2_relative_error")))
        density.append(_number(comparison.get("pure_density_relative_l2")))
    figure, axis = plt.subplots(figsize=(max(6.4, 1.4 * len(labels)), 3.8))
    x = np.arange(len(labels))
    axis.semilogy(x, energy, "o-", label="energy absolute error")
    axis.semilogy(x, radius, "s-", label=r"$R^2$ relative error")
    axis.semilogy(x, density, "^-", label="density relative L2")
    axis.set(xticks=x, xticklabels=labels, title="Exact trapped-TG validation")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)
    return figure
def _density(plt: Any, anchor: dict[str, Any]) -> Any | None:
    row = _mapping(anchor.get("density_profile"))
    x, exact = _vector(row.get("x")), _vector(row.get("exact_bin_averaged_n_x"))
    if not x.size or exact.size != x.size:
        return None
    pure, mixed = _vector(row.get("pure_fw_n_x")), _vector(row.get("mixed_n_x"))
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    axes[0].plot(x, exact, color="black", label="exact bin average")
    for values, label, style in ((pure, "forward walking", "-"), (mixed, "mixed", ":")):
        if values.size == x.size:
            axes[0].plot(x, values, style, label=label)
            axes[1].plot(x, values - exact, style, label=f"{label} - exact")
    axes[0].set(ylabel=r"$n(x)$", title=str(anchor.get("anchor_id", "")))
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel=r"$x/a_{ho}$", ylabel="residual")
    return figure
def _large_n_limit(plt: Any, anchors: list[dict[str, Any]]) -> Any:
    omegas = sorted({_number(row.get("omega", 1.0)) for row in anchors}) or [1.0]
    figure, axes = plt.subplots(1, len(omegas), figsize=(5.5 * len(omegas), 4.0), squeeze=False)
    for axis, omega in zip(axes[0], omegas, strict=True):
        for n_particles in (2, 4, 8, 16):
            extent = 1.35 * np.sqrt(2.0 * n_particles / omega)
            x = np.linspace(-extent, extent, 600)
            axis.plot(
                x,
                trapped_tg_density_profile(x, n_particles=n_particles, omega=omega),
                label=f"N={n_particles}",
            )
        axis.plot(
            x,
            trapped_tg_density_profile_semiclassical(x, n_particles=16, omega=omega),
            "--",
            color="black",
            label="large-N LDA",
        )
        axis.set(title=rf"$\omega={omega:g}$", xlabel="x", ylabel="n(x)")
        axis.legend(fontsize=7)
    return figure
def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
def _vector(value: object) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=float)
    return array if array.ndim == 1 else np.asarray([], dtype=float)
def _number(value: object) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")
