from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.components.status_badge import (
    draw_status_badge,
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
    diagnostic_lines = [f"Delta={delta:.4g}", sigma_text]
    lag_bound = finite_float(data.get("fw_lag_systematic_upper_bound"))
    relative_lag_bound = finite_float(data.get("fw_lag_systematic_relative_upper_bound"))
    lag_confidence = finite_float(data.get("fw_lag_equivalence_confidence_level"))
    if observable == "rms" and np.isfinite(lag_bound) and np.isfinite(relative_lag_bound):
        confidence_text = f"{100.0 * lag_confidence:g}%" if np.isfinite(lag_confidence) else "FW"
        diagnostic_lines.append(
            f"{confidence_text} lag bound <= {lag_bound:.3g} ({100.0 * relative_lag_bound:.3g}%)"
        )
    ax.set_xlim(-0.65, 0.65)
    ax.set_xticks([])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.text(
        0.03,
        0.04,
        "\n".join(diagnostic_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    draw_status_badge(
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
        (status or "not evaluated").replace("_", " "),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=tokens.INK_SOFT,
    )
    ax.set_xticks([])
    ax.set_yticks([])
