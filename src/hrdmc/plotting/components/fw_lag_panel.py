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
    _draw_seed_lag_traces(ax, payload)
    selected_lags = _integer_set(r2.get("selected_window_lags"))
    if selected_lags:
        _draw_aggregate_support(
            ax,
            lags=lags,
            values=values,
            stderr=stderr,
            selected_lags=selected_lags,
        )
    else:
        ax.plot(
            lags,
            values,
            marker="o",
            color=tokens.DMC_PRIMARY,
            linewidth=2.0,
            label=r"aggregate FW $R^2$",
        )
        if stderr is not None and stderr.shape == values.shape:
            ax.fill_between(
                lags,
                values - stderr,
                values + stderr,
                color=tokens.SEED_BAND,
                alpha=0.3,
            )
    plateau = finite_float(r2.get("plateau_value"))
    if np.isfinite(plateau):
        selected = sorted(selected_lags)
        if selected:
            ax.hlines(
                plateau,
                selected[0],
                selected[-1],
                color=tokens.ACCEPTED,
                linestyle=(0, (4, 2)),
                linewidth=1.4,
                label="selected plateau",
            )
        else:
            ax.axhline(
                plateau,
                color=tokens.ACCEPTED,
                linestyle=(0, (4, 2)),
                linewidth=1.2,
                label="plateau",
            )
    ax.set_title("Transported FW R2 lag ladder")
    ax.set_xlabel("lag steps")
    ax.set_ylabel(r"$R^2$")
    ax.legend(loc="best", fontsize=8)
    status_text = f"status={r2.get('plateau_status', 'unknown')}"
    if selected_lags:
        status_text += "; selected=" + ",".join(str(lag) for lag in sorted(selected_lags))
    ax.text(
        0.02,
        0.04,
        status_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=tokens.INK_SOFT,
    )


def _r2_lag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = _aggregate_r2_lag_payload(payload)
    if aggregate:
        return aggregate
    seeds = payload.get("seed_results", [])
    if isinstance(seeds, list) and seeds:
        first = seeds[0].get("pure_walking", {}).get("observable_results", {})
        if isinstance(first, dict) and "r2" in first:
            return first["r2"]
    pure = payload.get("pure_walking", {}).get("observables", {})
    return pure.get("r2", {}) if isinstance(pure, dict) else {}


def _aggregate_r2_lag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pure = payload.get("pure_walking", {})
    if not isinstance(pure, dict):
        return {}
    diagnostics = pure.get("r2_aggregate_plateau_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return {}
    values = diagnostics.get("values_by_lag")
    stderr = diagnostics.get("stderr_by_lag")
    if not isinstance(values, dict) or not values:
        return {}
    return {
        "values_by_lag": values,
        "stderr_by_lag": stderr,
        "plateau_value": pure.get("r2_aggregate_plateau_value"),
        "plateau_status": pure.get("r2_aggregate_plateau_status", ""),
        "selected_window_lags": diagnostics.get("selected_window_lags", []),
        "excluded_unsupported_lags": diagnostics.get("excluded_unsupported_lags", []),
    }


def _draw_aggregate_support(
    ax: Any,  # noqa: ANN401
    *,
    lags: np.ndarray,
    values: np.ndarray,
    stderr: np.ndarray | None,
    selected_lags: set[int],
) -> None:
    selected_mask = np.asarray([int(lag) in selected_lags for lag in lags], dtype=bool)
    diagnostic_mask = ~selected_mask
    if np.any(diagnostic_mask):
        ax.plot(
            lags[diagnostic_mask],
            values[diagnostic_mask],
            linestyle="none",
            marker="x",
            markersize=5.0,
            color=tokens.DMC_DIAGNOSTIC,
            label="non-plateau lag diagnostics",
            zorder=2,
        )
    ax.plot(
        lags[selected_mask],
        values[selected_mask],
        marker="o",
        color=tokens.DMC_PRIMARY,
        linewidth=2.2,
        label=r"supported aggregate FW $R^2$",
        zorder=4,
    )
    if stderr is not None and stderr.shape == values.shape:
        selected_stderr = stderr[selected_mask]
        selected_values = values[selected_mask]
        selected_x = lags[selected_mask]
        ax.fill_between(
            selected_x,
            selected_values - selected_stderr,
            selected_values + selected_stderr,
            color=tokens.SEED_BAND,
            alpha=0.35,
            zorder=3,
        )


def _integer_set(value: object) -> set[int]:
    if not isinstance(value, (list, tuple)):
        return set()
    try:
        return {int(item) for item in value}
    except (TypeError, ValueError):
        return set()


def _draw_seed_lag_traces(ax: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    seeds = payload.get("seed_results", [])
    if not isinstance(seeds, list):
        return
    label_used = False
    for seed_payload in seeds:
        r2 = seed_payload.get("pure_walking", {}).get("observable_results", {}).get("r2", {})
        seed_lag_values = _lag_dict_to_arrays(r2.get("values_by_lag", {}))
        if seed_lag_values is None:
            continue
        seed_lags, seed_values = seed_lag_values
        ax.plot(
            seed_lags,
            seed_values,
            marker=".",
            linewidth=0.9,
            markersize=3.0,
            alpha=0.22,
            color=tokens.INK_SOFT,
            label="seed traces" if not label_used else None,
            zorder=1,
        )
        label_used = True


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
