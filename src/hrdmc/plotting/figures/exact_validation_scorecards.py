from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.plotting import tokens
from hrdmc.plotting.figures.exact_validation_common import (
    anchor_label,
    compact_fw_gate,
    disable_layout_engine,
    draw_empty_table,
    draw_metric_chip,
    draw_multiline_metric,
    draw_packet_header,
    draw_row_rule,
    draw_status_chip,
    draw_table_header,
    finite_value,
    format_float,
    format_percent,
    fw_gate_status,
    value_pair,
)
from hrdmc.plotting.style import save_figure


def write_anchor_error_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    trapped = payload.get("trapped_tg_anchors", [])
    if not trapped:
        return []

    trapped_height = max(3.4, 0.58 * len(trapped) + 1.85)
    fig = plt.figure(
        figsize=(13.8, trapped_height + 1.15),
        constrained_layout=False,
    )
    disable_layout_engine(fig)
    trapped_ax = fig.add_subplot(1, 1, 1)
    draw_packet_header(fig, payload)
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.035, right=0.985)
    _draw_trapped_scorecard(trapped_ax, trapped)
    fig.suptitle(
        "Canonical trapped exact validation scorecard",
        y=0.985,
        fontsize=13,
        fontweight="semibold",
    )
    paths = save_figure(fig, plot_dir / "exact_anchor_errors", formats)
    plt.close(fig)
    return paths


def write_homogeneous_sanity_plot(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    anchors = payload.get("homogeneous_ring_anchors", [])
    if not anchors:
        return []

    height = max(2.6, 0.46 * len(anchors) + 1.55)
    fig, ax = plt.subplots(figsize=(12.2, height), constrained_layout=False)
    disable_layout_engine(fig)
    draw_packet_header(fig, payload)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.04, right=0.985)
    fig.suptitle(
        "Homogeneous hard-rod analytic sanity check",
        y=0.965,
        fontsize=12,
        fontweight="semibold",
    )
    _draw_homogeneous_sanity_scorecard(ax, anchors)
    paths = save_figure(fig, plot_dir / "homogeneous_ring_sanity", formats)
    plt.close(fig)
    return paths


def _draw_trapped_scorecard(
    ax: Any,  # noqa: ANN401
    anchors: list[dict[str, Any]],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        1.02,
        "Trapped Tonks-Girardeau anchors: full RN-DMC engine against analytic "
        "harmonic TG reference",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="semibold",
        color=tokens.INK,
    )
    ax.text(
        0.0,
        0.965,
        "Exact = analytic TG value. Energy is total E. Coordinate observables use "
        "transported FW; mixed coordinate values are diagnostics only.",
        transform=ax.transAxes,
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    ax.text(
        0.0,
        0.925,
        "Color key: green = metric within tolerance; red = metric outside "
        "tolerance; yellow FW gate = plateau/sample gate, not exact-reference error.",
        transform=ax.transAxes,
        fontsize=7.8,
        color=tokens.INK_SOFT,
    )

    columns = (
        (0.01, "anchor"),
        (0.19, "energy: DMC / exact"),
        (0.37, "R2: FW / exact"),
        (0.54, "RMS: FW / exact"),
        (0.70, "density L2: FW / mixed"),
        (0.82, "FW gate"),
        (0.94, "overall"),
    )
    draw_table_header(ax, columns, y=0.865)
    if not anchors:
        draw_empty_table(ax, "No trapped TG anchors in payload.")
        return

    row_step = min(0.145, 0.74 / max(len(anchors), 1))
    y = 0.785
    for anchor in anchors:
        comparison = anchor.get("exact_comparison", {})
        exact = anchor.get("exact_solution", {})
        draw_row_rule(ax, y + row_step * 0.42)
        ax.text(0.01, y, anchor_label(anchor), transform=ax.transAxes, fontsize=8.5)

        energy_error = finite_value(anchor.get("absolute_energy_error"))
        energy_tol = finite_value(anchor.get("energy_tolerance"))
        energy_text = value_pair(
            anchor.get("mixed_energy"),
            exact.get("energy_total"),
            error=energy_error,
            tolerance=energy_tol,
            error_label="abs",
        )
        draw_multiline_metric(ax, 0.19, y, energy_text, energy_error, energy_tol)

        r2_error = finite_value(comparison.get("pure_r2_relative_error"))
        r2_tol = finite_value(comparison.get("pure_r2_relative_tolerance"))
        r2_text = value_pair(
            comparison.get("pure_r2_radius"),
            comparison.get("exact_r2_radius"),
            error=r2_error,
            tolerance=r2_tol,
            error_label="rel",
        )
        draw_multiline_metric(ax, 0.37, y, r2_text, r2_error, r2_tol)

        rms_error = finite_value(comparison.get("pure_rms_relative_error"))
        rms_tol = finite_value(comparison.get("pure_rms_relative_tolerance"))
        rms_text = value_pair(
            comparison.get("pure_rms_radius"),
            comparison.get("exact_rms_radius"),
            error=rms_error,
            tolerance=rms_tol,
            error_label="rel",
        )
        draw_multiline_metric(ax, 0.54, y, rms_text, rms_error, rms_tol)

        density_error = finite_value(comparison.get("pure_density_relative_l2"))
        density_tol = finite_value(comparison.get("pure_density_l2_tolerance"))
        density_text = (
            f"FW={format_percent(density_error)}",
            "mixed="
            + format_percent(
                finite_value(comparison.get("mixed_density_relative_l2_diagnostic"))
            ),
        )
        draw_multiline_metric(ax, 0.70, y, density_text, density_error, density_tol)

        fw_gate = str(comparison.get("transported_fw_gate", "?"))
        draw_status_chip(ax, 0.82, y, compact_fw_gate(fw_gate), fw_gate_status(fw_gate))
        draw_status_chip(
            ax,
            0.94,
            y,
            str(anchor.get("status", "?")).upper(),
            anchor.get("status"),
        )
        y -= row_step


def _draw_homogeneous_sanity_scorecard(
    ax: Any,  # noqa: ANN401
    anchors: list[dict[str, Any]],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        1.03,
        "Local-energy formula check only; this does not run the canonical "
        "trapped RN-DMC benchmark.",
        transform=ax.transAxes,
        fontsize=8.3,
        color=tokens.INK,
    )
    ax.text(
        0.0,
        0.945,
        "Energy is E/N for a periodic homogeneous ring. Useful as a math/sanity "
        "anchor; not a trapped density, R2/RMS, or LDA-failure validation.",
        transform=ax.transAxes,
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    columns = (
        (0.01, "anchor"),
        (0.31, "DMC E/N"),
        (0.49, "exact E/N"),
        (0.66, "abs error"),
        (0.80, "tolerance"),
        (0.93, "status"),
    )
    draw_table_header(ax, columns, y=0.84)
    row_step = min(0.13, 0.68 / max(len(anchors), 1))
    y = 0.745
    for anchor in anchors:
        draw_row_rule(ax, y + row_step * 0.42)
        error = finite_value(anchor.get("max_energy_per_particle_abs_error"))
        tolerance = finite_value(anchor.get("tolerance_energy_per_particle_abs"))
        ax.text(0.01, y, anchor_label(anchor), transform=ax.transAxes, fontsize=8.5)
        ax.text(
            0.31,
            y,
            format_float(anchor.get("energy_per_particle_mean")),
            transform=ax.transAxes,
            fontsize=8.4,
            family="monospace",
        )
        ax.text(
            0.49,
            y,
            format_float(anchor.get("energy_per_particle_exact_finite_N")),
            transform=ax.transAxes,
            fontsize=8.4,
            family="monospace",
        )
        draw_metric_chip(ax, 0.66, y, format_float(error, precision=2), error, tolerance)
        ax.text(
            0.80,
            y,
            format_float(tolerance, precision=1),
            transform=ax.transAxes,
            fontsize=8.4,
            family="monospace",
            color=tokens.INK_SOFT,
        )
        draw_status_chip(
            ax,
            0.93,
            y,
            str(anchor.get("status", "?")).upper(),
            anchor.get("status"),
        )
        y -= row_step
