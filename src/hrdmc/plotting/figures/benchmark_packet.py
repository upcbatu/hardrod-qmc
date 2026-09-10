from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.plotting.style import load_pyplot, save_figure


def write_benchmark_packet_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Write the six compact diagnostics consumed by a benchmark packet."""
    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    writers: tuple[tuple[str, Callable[[Any, dict[str, Any]], Any]], ...] = (
        ("scalar_comparison", _scalar_figure),
        ("density_comparison", _density_figure),
        ("numerical_diagnostics", _chain_figure),
        ("energy_stationarity_diagnostics", _energy_trace_figure),
        ("fw_lag_diagnostics", _fw_figure),
        ("benchmark_packet_one_page", _packet_figure),
    )
    paths: list[Path] = []
    for stem, writer in writers:
        figure = writer(plt, payload)
        paths.extend(save_figure(figure, plot_dir / stem, formats))
        plt.close(figure)
    return [str(path.relative_to(output)) for path in paths]


def _scalar_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    estimates = _mapping(payload.get("estimates"))
    for axis, (name, label) in zip(
        axes,
        (("energy", "Energy"), ("r2", r"$R^2$"), ("rms", r"$R_{rms}$")),
        strict=True,
    ):
        row = _mapping(estimates.get(name))
        value, stderr, reference = (
            _number(row.get("value")),
            _number(row.get("stderr")),
            _number(row.get("lda", row.get("reference"))),
        )
        axis.errorbar([0], [value], yerr=[stderr], fmt="o", label="DMC")
        if np.isfinite(reference):
            axis.axhline(reference, color="black", linestyle="--", label="LDA/reference")
        axis.set(title=label, xticks=[])
        axis.legend(fontsize=8)
    _title(fig, payload)
    return fig


def _density_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True)
    density = _mapping(_mapping(payload.get("estimates")).get("density"))
    x, value = _vector(density.get("x")), _vector(density.get("value"))
    lda_x, lda = _vector(density.get("lda_x")), _vector(density.get("lda_value"))
    has_density = x.size > 0 and x.shape == value.shape and np.all(np.isfinite(value))
    has_lda = lda_x.size > 0 and lda_x.shape == lda.shape and np.all(np.isfinite(lda))
    if has_density:
        axes[0].plot(x, value, label="forward-walking DMC")
    else:
        axes[0].text(
            0.5,
            0.5,
            "Aggregate FW density unavailable; see seed results",
            transform=axes[0].transAxes,
            ha="center",
        )
    if has_lda:
        axes[0].plot(lda_x, lda, "--", color="black", label="LDA")
    axes[0].set_ylabel(r"$n(x)$")
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    if has_density and has_lda:
        axes[1].plot(x, value - np.interp(x, lda_x, lda))
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel=r"$x/a_{ho}$", ylabel="DMC - LDA")
    _title(fig, payload)
    return fig


def _chain_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    stationarity = _mapping(payload.get("stationarity"))
    diagnostics = _mapping(stationarity.get("diagnostics"))
    names = list(diagnostics)
    rows = [_mapping(diagnostics[name]) for name in names]
    axes[0].plot([_number(row.get("rhat")) for row in rows], "o")
    axes[0].set(title="Split R-hat", xticks=range(len(names)), xticklabels=names)
    axes[1].plot([_number(row.get("min_effective_independent_samples")) for row in rows], "o")
    axes[1].set(
        title="Minimum effective observations per seed", xticks=range(len(names)), xticklabels=names
    )
    _title(fig, payload)
    return fig


def _energy_trace_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    for seed in payload.get("seed_results", []):
        row = _mapping(seed)
        trace = _vector(row.get("block_energies", row.get("energy_trace")))
        if trace.size:
            axes[0].plot(trace, alpha=0.75, label=str(row.get("seed", "")))
            axes[1].plot(np.cumsum(trace) / np.arange(1, trace.size + 1), alpha=0.75)
    axes[0].set(ylabel="block energy", title="Energy stationarity")
    axes[1].set(xlabel="block", ylabel="cumulative mean")
    if axes[0].lines:
        axes[0].legend(fontsize=7, ncol=5)
    else:
        axes[0].text(
            0.5,
            0.5,
            "Full traces are in the trace artifacts; see summary.json",
            transform=axes[0].transAxes,
            ha="center",
        )
    _title(fig, payload)
    return fig


def _fw_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    dt = _number(_mapping(payload.get("controls")).get("dt"))
    scale = dt if np.isfinite(dt) and dt > 0 else 1.0
    xlabel = "Forward time (1/Omega)" if scale == dt else "Forward lag (steps)"
    for seed in payload.get("seed_results", []):
        results = _mapping(_mapping(seed.get("pure_walking")).get("observable_results"))
        rms = _mapping(_mapping(results.get("r2")).get("rms_radius_by_lag"))
        ancestry = _mapping(
            _mapping(results.get("density")).get("block_source_ancestor_ess_min_by_lag")
        )
        for axis, values in zip(axes, (rms, ancestry), strict=True):
            lags = sorted(values, key=int)
            if lags:
                axis.plot(
                    [int(lag) * scale for lag in lags],
                    [_number(values[lag]) for lag in lags],
                    "o-",
                    label=str(seed["seed"]),
                )
    axes[0].set(xlabel=xlabel, ylabel="RMS radius / a_ho", title="Measured radii by seed")
    axes[1].set(
        xlabel=xlabel,
        ylabel="Minimum block ancestor ESS",
        title="Density transport support by seed",
    )
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "No lag estimates available", transform=axis.transAxes, ha="center")
    _title(fig, payload)
    return fig


def _packet_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig = plt.figure(figsize=(8.27, 6.0))
    axis = fig.add_subplot(111)
    axis.axis("off")
    estimates = _mapping(payload.get("estimates"))
    lines = [f"{payload.get('case_id', '')} — {payload.get('status', '')}"]
    for key in ("energy", "r2", "rms"):
        row = _mapping(estimates.get(key))
        lines.append(f"{key}: {_number(row.get('value')):.8g} ± {_number(row.get('stderr')):.3g}")
    lines.append(f"seeds: {payload.get('seeds', [])}")
    lines.append(f"energy validation: {payload.get('energy_validation_status', '')}")
    lines.append(f"FW validation: {payload.get('pure_fw_validation_status', '')}")
    axis.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace")
    return fig


def _title(fig: Any, payload: dict[str, Any]) -> None:
    fig.suptitle(f"{payload.get('case_id', '')}  |  {payload.get('status', '')}")


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _vector(value: object) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=float)
    return array if array.ndim == 1 else np.asarray([], dtype=float)


def _number(value: object) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")
