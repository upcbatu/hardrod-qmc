from __future__ import annotations

from typing import Any

from hrdmc.plotting import tokens


def draw_tier_badge(
    ax: Any,  # noqa: ANN401
    *,
    methodology: str,
    precision: str,
    loc: tuple[float, float] = (0.985, 0.98),
) -> None:
    method_color = _combined_color(methodology, precision)
    x, y = loc
    label = methodology if precision == "N/A" else f"{methodology} / {precision}"
    _badge(ax, x, y, label, method_color)


def methodology_label(status: str, *, observable: str) -> str:
    if "NO_GO" in status:
        return tokens.TIER_GLYPHS["METHODOLOGY_NO_GO"]
    if "NOT_EVALUATED" in status:
        return tokens.TIER_GLYPHS["REGIME_NOT_SUPPORTED"]
    if observable == "energy" and status in {"ENERGY_CLAIM_GO", "GO", "PASS_CANDIDATE"}:
        return tokens.TIER_GLYPHS["METHODOLOGY_GO_VIA_MIXED_COMMUTES"]
    if "GO" in status:
        return tokens.TIER_GLYPHS["METHODOLOGY_GO_VIA_PURE_PLATEAU"]
    if "BIAS" in status or "PLATEAU" in status:
        return tokens.TIER_GLYPHS["METHODOLOGY_BIAS_BRACKETED"]
    return tokens.TIER_GLYPHS["METHODOLOGY_NO_GO"]


def precision_label(status: str) -> str:
    if "TRIANGULATED_2_OF_3" in status or status == "GO":
        return "TRIANGULATED"
    if "WARNING" in status or "DISAGREE" in status:
        return "WARN"
    if "NO_GO" in status:
        return "NO-GO"
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


def _tier_color(label: str) -> str:
    if "NO-GO" in label:
        return tokens.METHODOLOGY_NO_GO
    if "BIAS" in label:
        return tokens.METHODOLOGY_BIAS_BRACKETED
    if "N/A" in label:
        return tokens.REGIME_NA
    return tokens.METHODOLOGY_GO


def _combined_color(methodology: str, precision: str) -> str:
    if "NO-GO" in methodology or "NO-GO" in precision:
        return tokens.METHODOLOGY_NO_GO
    if "WARN" in precision:
        return tokens.PRECISION_WARN
    return _tier_color(methodology)
