from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.components.tier_badge import (
    draw_tier_badge,
    methodology_label,
    precision_label,
)
from hrdmc.plotting.numerics import finite_float
from hrdmc.plotting.primitives import errorbar_with_floor


def draw_scalar_panel(
    ax: Any,  # noqa: ANN401
    *,
    title: str,
    ylabel: str,
    data: dict[str, Any],
    observable: str,
    precision_status: str,
) -> None:
    value = finite_float(data.get("value"))
    stderr = finite_float(data.get("stderr"))
    lda = finite_float(data.get("lda_value"))
    status = str(data.get("status", ""))
    if not np.isfinite(value) or not np.isfinite(lda):
        _draw_unavailable(ax, title, status)
        return

    ax.axhline(
        lda,
        color=tokens.LDA_PREDICTION,
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        label="LDA reference",
    )
    sigma_text = errorbar_with_floor(
        ax,
        value=value,
        stderr=stderr,
        reference=lda,
        label="DMC/FW estimator",
    )
    ax.scatter([0.0], [lda], marker="_", s=240, color=tokens.LDA_PREDICTION, zorder=3)
    delta = value - lda
    ax.set_xlim(-0.65, 0.65)
    ax.set_xticks([])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.text(
        0.03,
        0.04,
        f"Delta={delta:.4g}\n{sigma_text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    draw_tier_badge(
        ax,
        methodology=methodology_label(status, observable=observable),
        precision=precision_label(precision_status),
    )


def _draw_unavailable(ax: Any, title: str, status: str) -> None:  # noqa: ANN401
    ax.set_title(title)
    ax.set_facecolor("#F3F3F3")
    ax.text(
        0.5,
        0.5,
        status or "REGIME_NOT_SUPPORTED",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    ax.set_xticks([])
    ax.set_yticks([])
