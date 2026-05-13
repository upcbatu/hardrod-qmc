from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.numerics import display_sigma, smart_ylim


def errorbar_with_floor(
    ax: Any,  # noqa: ANN401
    *,
    value: float,
    stderr: float,
    reference: float,
    label: str,
) -> str:
    delta = value - reference
    sigma = display_sigma(delta, stderr)
    yerr = None if sigma.status == "stderr_at_floor" or not np.isfinite(stderr) else [stderr]
    ax.errorbar(
        [0.0],
        [value],
        yerr=yerr,
        fmt=tokens.DMC_MARKER,
        color=tokens.DMC_PRIMARY,
        ecolor=tokens.DMC_PRIMARY,
        capsize=4,
        markersize=5,
        label=label,
    )
    ax.set_ylim(*smart_ylim(value, reference, stderr))
    return sigma.text


def threshold_lollipop(
    ax: Any,  # noqa: ANN401
    *,
    values: np.ndarray,
    labels: list[str],
    threshold: float,
    pass_below: bool,
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> None:
    x = np.arange(values.size)
    colors = [
        tokens.METHODOLOGY_GO if ((value <= threshold) if pass_below else (value >= threshold))
        else tokens.METHODOLOGY_NO_GO
        for value in values
    ]
    ax.axhline(threshold, color=tokens.INK, linestyle=(0, (4, 2)), linewidth=1.1)
    for xi, value, color in zip(x, values, colors, strict=True):
        bottom = threshold if not log_scale else min(value, threshold)
        top = value if not log_scale else max(value, threshold)
        ax.vlines(xi, bottom, top, color=color, linewidth=2.0, alpha=0.75)
        ax.scatter([xi], [value], color=color, s=42, zorder=3)
        ax.text(xi, value, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
        lo = max(min(float(np.min(values)), threshold) * 0.65, 1.0e-3)
        hi = max(float(np.max(values)), threshold) * 1.55
        ax.set_ylim(lo, hi)
    else:
        lo = min(float(np.min(values)), threshold)
        hi = max(float(np.max(values)), threshold)
        pad = max(0.08 * (hi - lo), 0.01 * max(abs(threshold), 1.0))
        ax.set_ylim(lo - pad, hi + pad)
