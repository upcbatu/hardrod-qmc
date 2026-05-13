from __future__ import annotations

from typing import Any

from hrdmc.plotting import tokens
from hrdmc.plotting.numerics import metric_array
from hrdmc.plotting.primitives import threshold_lollipop


def draw_chain_panel(ax_rhat: Any, ax_neff: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    stationarity = payload.get("stationarity", {})
    labels = ["energy", "mixed RMS", r"mixed $R^2$"]
    rhat = metric_array(stationarity, ("rhat_energy", "rhat_rms", "rhat_r2"))
    neff = metric_array(stationarity, ("neff_energy", "neff_rms", "neff_r2"))
    if rhat is not None:
        threshold_lollipop(
            ax_rhat,
            values=rhat,
            labels=labels,
            threshold=1.05,
            pass_below=True,
            ylabel=r"$\hat{R}$",
            title="Chain agreement",
        )
    else:
        ax_rhat.text(0.5, 0.5, "R-hat unavailable", transform=ax_rhat.transAxes, ha="center")
        ax_rhat.set_axis_off()
    if neff is not None:
        threshold_lollipop(
            ax_neff,
            values=neff,
            labels=labels,
            threshold=30.0,
            pass_below=False,
            ylabel=r"$N_\mathrm{eff}$",
            title="Effective sample size",
            log_scale=True,
        )
        ax_neff.text(
            0.02,
            0.08,
            "log scale; energy often has larger effective sample size",
            transform=ax_neff.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=tokens.INK_SOFT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    else:
        ax_neff.text(0.5, 0.5, "N_eff unavailable", transform=ax_neff.transAxes, ha="center")
        ax_neff.set_axis_off()
