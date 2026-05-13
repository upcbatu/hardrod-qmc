from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.numerics import finite_float


def draw_fw_lag_panel(ax: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    r2 = _r2_lag_payload(payload)
    if not r2:
        ax.text(0.5, 0.5, "FW R2 lag ladder unavailable", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    lag_values = _lag_dict_to_arrays(r2.get("values_by_lag", {}))
    stderr_values = _lag_dict_to_arrays(r2.get("stderr_by_lag", {}))
    if lag_values is None:
        ax.text(0.5, 0.5, "FW lag values unavailable", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    lags, values = lag_values
    stderr = stderr_values[1] if stderr_values is not None else None
    ax.plot(lags, values, marker="o", color=tokens.DMC_PRIMARY, label=r"pure FW $R^2$")
    if stderr is not None and stderr.shape == values.shape:
        ax.fill_between(lags, values - stderr, values + stderr, color=tokens.SEED_BAND, alpha=0.3)
    plateau = finite_float(r2.get("plateau_value"))
    if np.isfinite(plateau):
        ax.axhline(
            plateau,
            color=tokens.METHODOLOGY_GO,
            linestyle=(0, (4, 2)),
            linewidth=1.2,
            label="plateau",
        )
    ax.set_title("Transported FW R2 lag ladder")
    ax.set_xlabel("lag steps")
    ax.set_ylabel(r"$R^2$")
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.02,
        0.04,
        f"status={r2.get('plateau_status', 'unknown')}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )


def _r2_lag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    seeds = payload.get("seed_results", [])
    if isinstance(seeds, list) and seeds:
        first = seeds[0].get("pure_walking", {}).get("observable_results", {})
        if isinstance(first, dict) and "r2" in first:
            return first["r2"]
    pure = payload.get("pure_walking", {}).get("observables", {})
    return pure.get("r2", {}) if isinstance(pure, dict) else {}


def _lag_dict_to_arrays(value: object) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(value, dict) or not value:
        return None
    pairs: list[tuple[int, float]] = []
    for key, item in value.items():
        try:
            lag = int(key)
            scalar = float(np.asarray(item, dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(scalar):
            pairs.append((lag, scalar))
    if not pairs:
        return None
    pairs.sort(key=lambda pair: pair[0])
    return (
        np.asarray([pair[0] for pair in pairs], dtype=float),
        np.asarray([pair[1] for pair in pairs], dtype=float),
    )
