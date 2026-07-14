from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.plotting.components import (
    draw_case_header,
    draw_chain_panel,
    draw_density_panel,
    draw_energy_stationarity_panel,
    draw_fw_lag_panel,
    draw_scalar_panel,
)
from hrdmc.plotting.numerics import array_or_none
from hrdmc.plotting.style import load_pyplot, save_figure


def write_benchmark_packet_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Render benchmark packet plots from an assembled workflow payload."""

    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    paths.extend(_write_scalar_comparison(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_density_comparison(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_numerical_diagnostics(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_energy_stationarity_diagnostics(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_fw_lag_diagnostics(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_optional_vector_observables(plt, plot_dir, payload, formats=formats))
    paths.extend(_write_one_page_packet(plt, plot_dir, payload, formats=formats))
    return [str(path.relative_to(output)) for path in paths]


def _write_scalar_comparison(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    estimates = payload.get("estimates", {})
    rows = [
        ("Energy", r"$E$", "energy", estimates.get("energy", {})),
        (r"$R^2$", r"$R^2$", "r2", estimates.get("r2", {})),
        (
            r"$R_\mathrm{rms}$",
            r"$R_\mathrm{rms}$",
            "rms",
            estimates.get("rms", {}),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    draw_case_header(fig, payload)
    precision_status = _precision_status(payload)
    for ax, (title, ylabel, observable, data) in zip(axes, rows, strict=True):
        draw_scalar_panel(
            ax,
            title=title,
            ylabel=ylabel,
            data=data,
            observable=observable,
            precision_status=precision_status,
        )
    paths = save_figure(fig, plot_dir / "scalar_comparison", formats)
    plt.close(fig)
    return paths


def _write_density_comparison(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.7, 6.3),
        sharex=True,
        gridspec_kw={"height_ratios": (2.35, 1.0)},
        constrained_layout=False,
    )
    draw_density_panel(axes[0], payload, residual_ax=axes[1])
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.985, hspace=0.12)
    paths = save_figure(fig, plot_dir / "density_comparison", formats)
    plt.close(fig)
    return paths


def _write_numerical_diagnostics(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    draw_case_header(fig, payload)
    draw_chain_panel(axes[0], axes[1], payload)
    paths = save_figure(fig, plot_dir / "numerical_diagnostics", formats)
    plt.close(fig)
    return paths


def _write_energy_stationarity_diagnostics(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.8),
        sharex=False,
        gridspec_kw={"height_ratios": (1.15, 1.0)},
    )
    draw_case_header(fig, payload)
    draw_energy_stationarity_panel(axes[0], axes[1], payload)
    paths = save_figure(fig, plot_dir / "energy_stationarity_diagnostics", formats)
    plt.close(fig)
    return paths


def _write_fw_lag_diagnostics(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=False)
    draw_case_header(fig, payload)
    draw_fw_lag_panel(ax, payload)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.11, right=0.985)
    paths = save_figure(fig, plot_dir / "fw_lag_diagnostics", formats)
    plt.close(fig)
    return paths


def _write_optional_vector_observables(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    estimates = payload.get("estimates", {})
    paths: list[Path] = []
    paths.extend(
        _write_xy_figure(
            plt,
            plot_dir,
            payload,
            estimates.get("pair_distance_density", {}),
            x_key="x",
            x_label="pair distance",
            y_label="pair-distance density",
            title="Pair-distance density",
            filename="pair_distance_density",
            formats=formats,
        )
    )
    paths.extend(
        _write_xy_figure(
            plt,
            plot_dir,
            payload,
            estimates.get("structure_factor", {}),
            x_key="k_values",
            x_label=r"$k$",
            y_label=r"$S(k)$",
            title="Finite-cloud structure factor",
            filename="structure_factor",
            formats=formats,
        )
    )
    return paths


def _write_xy_figure(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    observable: dict[str, Any],
    *,
    x_key: str,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    formats: tuple[str, ...],
) -> list[Path]:
    x = array_or_none(observable.get(x_key))
    y = array_or_none(observable.get("value"))
    if x is None or y is None:
        return []
    stderr = array_or_none(observable.get("stderr"))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    draw_case_header(fig, payload)
    ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.8)
    if stderr is not None and stderr.shape == y.shape:
        ax.fill_between(x, y - stderr, y + stderr, alpha=0.22, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    status = str(observable.get("status", ""))
    if status:
        ax.text(0.02, 0.96, status, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    paths = save_figure(fig, plot_dir / filename, formats)
    plt.close(fig)
    return paths


def _write_one_page_packet(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    estimates = payload.get("estimates", {})
    fig = plt.figure(figsize=(8.27, 11.69))
    draw_case_header(fig, payload)
    spec = fig.add_gridspec(
        4,
        3,
        height_ratios=(0.95, 1.5, 1.1, 1.25),
        hspace=0.45,
        wspace=0.34,
    )
    scalar_axes = [fig.add_subplot(spec[0, i]) for i in range(3)]
    draw_scalar_panel(
        scalar_axes[0],
        title="Energy",
        ylabel=r"$E$",
        data=estimates.get("energy", {}),
        observable="energy",
        precision_status=_precision_status(payload),
    )
    draw_scalar_panel(
        scalar_axes[1],
        title=r"$R^2$",
        ylabel=r"$R^2$",
        data=estimates.get("r2", {}),
        observable="r2",
        precision_status=_precision_status(payload),
    )
    draw_scalar_panel(
        scalar_axes[2],
        title=r"$R_\mathrm{rms}$",
        ylabel=r"$R_\mathrm{rms}$",
        data=estimates.get("rms", {}),
        observable="rms",
        precision_status=_precision_status(payload),
    )
    density_spec = spec[1, :].subgridspec(
        2,
        1,
        height_ratios=(2.2, 0.8),
        hspace=0.14,
    )
    draw_density_panel(
        fig.add_subplot(density_spec[0, 0]),
        payload,
        residual_ax=fig.add_subplot(density_spec[1, 0]),
    )
    draw_chain_panel(fig.add_subplot(spec[2, :2]), fig.add_subplot(spec[2, 2]), payload)
    draw_fw_lag_panel(fig.add_subplot(spec[3, :]), payload)
    paths = save_figure(fig, plot_dir / "benchmark_packet_one_page", formats)
    plt.close(fig)
    return paths


def _precision_status(payload: dict[str, Any]) -> str:
    stationarity = payload.get("stationarity", {})
    return str(stationarity.get("precision_status", payload.get("status", "")))
