from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.components.tier_badge import (
    draw_tier_badge,
    methodology_label,
    precision_label,
)
from hrdmc.plotting.numerics import array_or_none, density_integral_status, finite_float


def draw_density_panel(ax: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
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

    if lda_x is not None and lda_value is not None:
        ax.plot(
            lda_x,
            lda_value,
            color=tokens.LDA_PREDICTION,
            linestyle=(0, (4, 2)),
            linewidth=1.25,
            label="trapped LDA envelope",
        )
    ax.plot(
        x,
        value,
        color=tokens.DMC_PRIMARY,
        linewidth=2.15,
        label="transported FW pure estimator",
        zorder=3,
    )
    if mixed_x is not None and mixed_value is not None:
        ax.plot(
            mixed_x,
            mixed_value,
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            linewidth=1.65,
            alpha=0.95,
            marker="x",
            markevery=max(1, int(mixed_x.size // 16)),
            markersize=3.2,
            label="mixed coordinate diagnostic",
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
            label="seed stderr",
        )
    integral = finite_float(density.get("integral"))
    n_particles = int(payload.get("n_particles", 0) or 0)
    ax.set_title(
        f"Density n(x), {density_integral_status(integral, n_particles)}\n"
        f"{_density_message(x=x, value=value, lda_x=lda_x, lda_value=lda_value)}"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$n(x)$")
    ax.legend(loc="upper right", fontsize=8)
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
    draw_tier_badge(
        ax,
        methodology=methodology_label(str(density.get("status", "")), observable="density"),
        precision=precision_label(str(payload.get("pure_fw_claim_status", ""))),
    )


def _should_draw_band(
    stderr: np.ndarray | None,
    value: np.ndarray,
    payload: dict[str, Any],
) -> bool:
    seed_count = int(payload.get("seed_count", 0) or 0)
    return stderr is not None and stderr.shape == value.shape and seed_count >= 5


def _density_message(
    *,
    x: np.ndarray,
    value: np.ndarray,
    lda_x: np.ndarray | None,
    lda_value: np.ndarray | None,
) -> str:
    if lda_x is not None and lda_value is not None:
        if _has_two_peak_signal(x, value) and not _has_two_peak_signal(lda_x, lda_value):
            return "finite-N two-peak FW density vs smooth trapped LDA envelope"
    return "transported FW density vs trapped LDA envelope"


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


def _has_two_peak_signal(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size < 5 or y.size < 5:
        return False
    center_idx = int(np.argmin(np.abs(x)))
    center_value = y[center_idx]
    left = np.max(y[:center_idx]) if center_idx > 0 else center_value
    right = np.max(y[center_idx + 1 :]) if center_idx + 1 < y.size else center_value
    return bool(left > 1.12 * center_value and right > 1.12 * center_value)
