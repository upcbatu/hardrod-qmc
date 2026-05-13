from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens


def draw_table_header(
    ax: Any,  # noqa: ANN401
    columns: tuple[tuple[float, str], ...],
    *,
    y: float,
) -> None:
    for x, label in columns:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=7.7,
            color=tokens.INK_SOFT,
            fontweight="semibold",
        )
    ax.axhline(y - 0.035, color=tokens.INK_FAINT, linewidth=0.8)


def draw_empty_table(ax: Any, text: str) -> None:  # noqa: ANN401
    ax.text(
        0.01,
        0.72,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=tokens.INK_SOFT,
    )


def draw_row_rule(ax: Any, y: float) -> None:  # noqa: ANN401
    ax.axhline(y, color=tokens.INK_FAINT, linewidth=0.55)


def draw_multiline_metric(
    ax: Any,  # noqa: ANN401
    x: float,
    y: float,
    lines: tuple[str, str],
    error: float,
    tolerance: float,
) -> None:
    status = metric_status(error, tolerance)
    color = status_color(status)
    ax.text(
        x,
        y + 0.020,
        lines[0],
        transform=ax.transAxes,
        fontsize=8.3,
        family="monospace",
        color=tokens.INK,
    )
    ax.text(
        x,
        y - 0.026,
        lines[1],
        transform=ax.transAxes,
        fontsize=7.3,
        family="monospace",
        color=color,
    )


def draw_metric_chip(
    ax: Any,  # noqa: ANN401
    x: float,
    y: float,
    label: str,
    value: float,
    tolerance: float,
) -> None:
    draw_status_chip(ax, x, y, label, metric_status(value, tolerance))


def draw_status_chip(
    ax: Any,  # noqa: ANN401
    x: float,
    y: float,
    label: str,
    status: object,
) -> None:
    normalized = status_name(status)
    color = status_color(normalized)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=7.4,
        family="monospace",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": color,
            "alpha": 0.10,
            "edgecolor": color,
            "linewidth": 0.8,
        },
    )


def value_pair(
    estimate: object,
    exact: object,
    *,
    error: float,
    tolerance: float,
    error_label: str,
) -> tuple[str, str]:
    return (
        f"{format_float(estimate)} / {format_float(exact)}",
        f"{error_label}={format_error(error)}  tol={format_error(tolerance)}",
    )


def format_error(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2g}%"


def format_float(value: object, *, precision: int = 6) -> str:
    parsed = finite_value(value)
    if not np.isfinite(parsed):
        return "n/a"
    return f"{parsed:.{precision}g}"


def metric_status(value: float, tolerance: float) -> str:
    if not np.isfinite(value) or not np.isfinite(tolerance):
        return "na"
    return "go" if value <= tolerance else "no_go"


def status_name(status: object) -> str:
    raw = str(status).lower()
    if raw in {"passed", "pass", "go"}:
        return "go"
    if raw in {"warning", "warn", "precision_warning"}:
        return "warn"
    if raw in {"failed", "fail", "no_go"}:
        return "no_go"
    if "warning" in raw:
        return "warn"
    if "no_go" in raw or "failed" in raw or "fail" in raw:
        return "no_go"
    if "go" in raw or "passed" in raw:
        return "go"
    return "na"


def status_color(status: object) -> str:
    normalized = status_name(status)
    return tokens.STATUS_COLORS.get(normalized, tokens.REGIME_NA)


def fw_gate_status(gate: str) -> str:
    if gate == "PURE_WALKING_GO":
        return "go"
    if gate in {"PURE_WALKING_PLATEAU_NO_GO", "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO"}:
        return "warn"
    if gate == "?":
        return "na"
    return "no_go"


def compact_fw_gate(gate: str) -> str:
    if gate == "PURE_WALKING_GO":
        return "FW GO"
    if gate == "PURE_WALKING_PLATEAU_NO_GO":
        return "PLATEAU NO-GO"
    if gate == "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO":
        return "SAMPLE NO-GO"
    return gate.replace("PURE_WALKING_", "")[:15]


def anchor_label(anchor: dict[str, Any]) -> str:
    label = str(anchor.get("anchor_id", "?"))
    for prefix in ("trapped_tg_", "hom_ring_"):
        label = label.removeprefix(prefix)
    return label.replace("_omega", "  omega=").replace("_eta", "  eta=")


def finite_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def disable_layout_engine(fig: Any) -> None:  # noqa: ANN401
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("none")


def draw_packet_header(fig: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    text = (
        "canonical exact validation packet  "
        f"status={payload.get('status', '?')}  "
        f"anchors={len(payload.get('anchor_table', []))}"
    )
    fig.text(
        0.01,
        0.02,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        color=tokens.INK_SOFT,
    )
