from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.numerics import array_or_none, density_integral_status, finite_float


def draw_density_panel(
    ax: Any,  # noqa: ANN401
    payload: dict[str, Any],
    *,
    residual_ax: Any | None = None,  # noqa: ANN401
) -> None:
    density = payload.get("paper_values", {}).get("density", {})
    x = array_or_none(density.get("x"))
    value = array_or_none(density.get("value"))
    if x is None or value is None:
        ax.text(0.5, 0.5, "density unavailable", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    stderr = array_or_none(density.get("stderr"))
    lda_x = array_or_none(density.get("lda_x"))
    lda_value = array_or_none(density.get("lda_value"))
    mixed_x = array_or_none(density.get("mixed_diagnostic_x"))
    mixed_value = array_or_none(density.get("mixed_diagnostic_value"))
    bin_edges = array_or_none(density.get("bin_edges"))
    lda_on_x = _reference_on_grid(x, lda_x, lda_value)

    if lda_x is not None and lda_value is not None:
        ax.plot(
            lda_x,
            lda_value,
            color=tokens.LDA_PREDICTION,
            linestyle=(0, (4, 2)),
            linewidth=1.25,
            label="LDA envelope (smooth)",
            zorder=1,
        )
    ax.step(
        x,
        value,
        where="mid",
        color=tokens.DMC_PRIMARY,
        linewidth=2.3,
        label="FW density (bin estimator)",
        zorder=3,
    )
    if mixed_x is not None and mixed_value is not None:
        ax.step(
            mixed_x,
            mixed_value,
            where="mid",
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            linewidth=2.0,
            alpha=1.0,
            marker="x",
            markevery=max(1, int(mixed_x.size // 16)),
            markersize=4.0,
            label="mixed diagnostic (bin)",
            zorder=4,
        )
    if _should_draw_band(stderr, value, payload):
        ax.fill_between(
            x,
            value - stderr,
            value + stderr,
            color=tokens.SEED_BAND,
            alpha=0.35,
            linewidth=0,
            step="mid",
            label="seed stderr",
        )
    integral = finite_float(density.get("integral"))
    n_particles = int(payload.get("n_particles", 0) or 0)
    ax.set_title(_density_title(payload), loc="left", pad=6)
    if residual_ax is None:
        ax.set_xlabel(r"$x$")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$n(x)$")
    ax.legend(loc="upper right", fontsize=7.4, frameon=False)
    _draw_density_metrics(ax, x, value, lda_on_x, mixed_x, mixed_value, integral, n_particles)
    if (
        mixed_x is not None
        and mixed_value is not None
        and _curves_overlap(x, value, mixed_x, mixed_value)
    ):
        ax.text(
            0.98,
            0.06,
            "mixed diagnostic overlaps FW",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=tokens.DMC_DIAGNOSTIC,
        )
    if residual_ax is not None:
        _draw_density_residuals(
            residual_ax,
            x=x,
            value=value,
            lda_on_x=lda_on_x,
            mixed_x=mixed_x,
            mixed_value=mixed_value,
        )
        _set_density_xlim(ax, residual_ax, x, value, lda_x, lda_value, mixed_x, mixed_value)
    elif bin_edges is not None:
        _set_density_xlim(ax, None, x, value, lda_x, lda_value, mixed_x, mixed_value)


def _should_draw_band(
    stderr: np.ndarray | None,
    value: np.ndarray,
    payload: dict[str, Any],
) -> bool:
    seed_count = int(payload.get("seed_count", 0) or 0)
    return stderr is not None and stderr.shape == value.shape and seed_count >= 5


def _density_title(payload: dict[str, Any]) -> str:
    n_particles = payload.get("n_particles")
    rod_length = payload.get("rod_length")
    omega = payload.get("omega")
    if n_particles is not None and rod_length is not None and omega is not None:
        return (
            f"Density comparison: N={int(n_particles)}, "
            f"a={float(rod_length):g}, omega={float(omega):g}"
        )
    case_id = payload.get("case_id")
    if isinstance(case_id, str) and case_id:
        return f"Density comparison: {case_id}"
    return r"Density $n(x)$"


def _reference_on_grid(
    x: np.ndarray,
    lda_x: np.ndarray | None,
    lda_value: np.ndarray | None,
) -> np.ndarray | None:
    if lda_x is None or lda_value is None or lda_x.size < 2:
        return None
    return np.interp(x, lda_x, lda_value)


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(value - reference) / denominator)


def _draw_density_metrics(
    ax: Any,  # noqa: ANN401
    x: np.ndarray,
    value: np.ndarray,
    lda_on_x: np.ndarray | None,
    mixed_x: np.ndarray | None,
    mixed_value: np.ndarray | None,
    integral: float,
    n_particles: int,
) -> None:
    lines = [density_integral_status(integral, n_particles)]
    if lda_on_x is None:
        return
    lines.append(f"FW-LDA rel L2: {_relative_l2(value, lda_on_x):.3f}")
    if mixed_x is not None and mixed_value is not None:
        mixed_on_x = np.interp(x, mixed_x, mixed_value)
        mixed_l2 = _relative_l2(mixed_on_x, lda_on_x)
        fw_l2 = _relative_l2(value, lda_on_x)
        lines.append(f"mixed-LDA rel L2: {mixed_l2:.3f}")
        lines.append(f"FW-mixed rel L2: {_relative_l2(value, mixed_on_x):.3f}")
        if np.isfinite(fw_l2) and np.isfinite(mixed_l2) and fw_l2 > mixed_l2:
            lines.append("FW farther from LDA than mixed")
    ax.text(
        0.015,
        0.955,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=tokens.INK_SOFT,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": tokens.INK_FAINT,
            "alpha": 0.88,
        },
        zorder=8,
    )


def _draw_density_residuals(
    ax: Any,  # noqa: ANN401
    *,
    x: np.ndarray,
    value: np.ndarray,
    lda_on_x: np.ndarray | None,
    mixed_x: np.ndarray | None,
    mixed_value: np.ndarray | None,
) -> None:
    if lda_on_x is None:
        ax.text(0.5, 0.5, "LDA residual unavailable", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    fw_residual = value - lda_on_x
    ax.axhline(0.0, color=tokens.INK_SOFT, linewidth=0.8)
    ax.step(
        x,
        fw_residual,
        where="mid",
        color=tokens.DMC_PRIMARY,
        linewidth=1.65,
        label="FW - LDA",
    )
    residuals = [fw_residual]
    if mixed_x is not None and mixed_value is not None:
        mixed_on_x = np.interp(x, mixed_x, mixed_value)
        mixed_residual = mixed_on_x - lda_on_x
        residuals.append(mixed_residual)
        ax.step(
            x,
            mixed_residual,
            where="mid",
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            linewidth=1.5,
            marker="x",
            markevery=max(1, int(x.size // 18)),
            markersize=3.4,
            label="mixed - LDA",
        )
    max_abs = max(float(np.nanmax(np.abs(residual))) for residual in residuals)
    if max_abs > 0.0 and np.isfinite(max_abs):
        ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Delta n$")
    ax.set_title("Residual to LDA envelope", loc="left", pad=3, fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7.2)


def _set_density_xlim(
    ax: Any,  # noqa: ANN401
    residual_ax: Any | None,  # noqa: ANN401
    x: np.ndarray,
    value: np.ndarray,
    lda_x: np.ndarray | None,
    lda_value: np.ndarray | None,
    mixed_x: np.ndarray | None,
    mixed_value: np.ndarray | None,
) -> None:
    xs = [x]
    ys = [value]
    if lda_x is not None and lda_value is not None:
        xs.append(lda_x)
        ys.append(lda_value)
    if mixed_x is not None and mixed_value is not None:
        xs.append(mixed_x)
        ys.append(mixed_value)
    max_y = max(float(np.nanmax(np.abs(y))) for y in ys if y.size)
    if not np.isfinite(max_y) or max_y <= 0.0:
        return
    threshold = 0.006 * max_y
    active_edges: list[float] = []
    for x_values, y_values in zip(xs, ys, strict=True):
        active = np.flatnonzero(np.abs(y_values) > threshold)
        if active.size:
            active_edges.extend([float(x_values[active[0]]), float(x_values[active[-1]])])
    if not active_edges:
        return
    left = min(active_edges)
    right = max(active_edges)
    margin = max(0.08 * (right - left), float(np.median(np.diff(np.sort(x)))) * 2.0)
    xlim = (left - margin, right + margin)
    ax.set_xlim(*xlim)
    if residual_ax is not None:
        residual_ax.set_xlim(*xlim)


def _curves_overlap(
    x: np.ndarray,
    value: np.ndarray,
    mixed_x: np.ndarray,
    mixed_value: np.ndarray,
) -> bool:
    if x.size < 2 or mixed_x.size < 2:
        return False
    interpolated = np.interp(x, mixed_x, mixed_value)
    scale = max(float(np.nanmax(np.abs(value))), 1.0)
    return bool(np.nanmax(np.abs(interpolated - value)) < 0.01 * scale)
