from __future__ import annotations

from typing import Any

from hrdmc.plotting import tokens


def draw_status_badge(
    ax: Any,  # noqa: ANN401
    *,
    methodology: str,
    precision: str,
    loc: tuple[float, float] = (0.985, 0.98),
) -> None:
    color = _combined_color(methodology, precision)
    x, y = loc
    label = methodology if precision == "N/A" else f"{methodology} / {precision}"
    _badge(ax, x, y, label, color)


def methodology_label(status: str, *, observable: str) -> str:
    normalized = status.lower()
    labels = tokens.ESTIMATOR_STATUS_LABELS
    if "not_evaluated" in normalized:
        return labels["NOT_EVALUATED"]
    if normalized == "accepted" and observable == "energy":
        return labels["ENERGY_ESTIMATOR_ACCEPTED"]
    if normalized == "accepted":
        return labels["PURE_ESTIMATOR_ACCEPTED"]
    if "bias" in normalized or "plateau_unresolved" in normalized:
        return labels["ESTIMATOR_BIAS_BRACKETED"]
    return labels["ESTIMATOR_REJECTED"]


def precision_label(status: str) -> str:
    normalized = status.lower()
    if "two_of_three_error_estimates_agree" in normalized or normalized == "accepted":
        return "CORRELATED ERROR"
    if any(token in normalized for token in ("warning", "disagree", "unresolved")):
        return "WARNING"
    if "not_evaluated" in normalized or "unavailable" in normalized:
        return "N/A"
    if normalized:
        return "REJECTED"
    return "N/A"


def _badge(ax: Any, x: float, y: float, label: str, color: str) -> None:  # noqa: ANN401
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="right" if x > 0.5 else "left",
        va="top",
        fontsize=6.5,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.06",
            "facecolor": color,
            "edgecolor": "none",
            "alpha": 0.92,
        },
    )


def _status_color(label: str) -> str:
    if "REJECTED" in label:
        return tokens.REJECTED
    if "BIAS" in label:
        return tokens.ESTIMATOR_BIAS_BRACKETED
    if "N/A" in label:
        return tokens.REGIME_NA
    return tokens.ACCEPTED


def _combined_color(methodology: str, precision: str) -> str:
    if "REJECTED" in methodology or "REJECTED" in precision:
        return tokens.REJECTED
    if "WARNING" in precision:
        return tokens.PRECISION_WARN
    return _status_color(methodology)
