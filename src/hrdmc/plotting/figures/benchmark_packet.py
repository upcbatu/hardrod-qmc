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
    axes[0].plot(x, value, label="forward-walking DMC")
    axes[0].plot(lda_x, lda, "--", color="black", label="LDA")
    axes[0].set_ylabel(r"$n(x)$")
    axes[0].legend(fontsize=8)
    if x.size and lda_x.size:
        axes[1].plot(x, value - np.interp(x, lda_x, lda))
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel=r"$x/a_{ho}$", ylabel="DMC - LDA")
    _title(fig, payload)
    return fig
def _chain_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    stationarity = _mapping(payload.get("stationarity"))
    metrics = stationarity.get("metrics", stationarity.get("observables", {}))
    rows = list(metrics.values()) if isinstance(metrics, dict) else []
    rhat = [_number(_mapping(row).get("split_rhat")) for row in rows]
    ess = [_number(_mapping(row).get("effective_sample_size")) for row in rows]
    axes[0].plot(rhat, "o-")
    axes[0].axhline(1.01, color="black", linestyle="--")
    axes[0].set(title="Split R-hat", xlabel="observable")
    axes[1].plot(ess, "o-")
    axes[1].set(title="Effective sample count", xlabel="observable")
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
    _title(fig, payload)
    return fig
def _fw_figure(plt: Any, payload: dict[str, Any]) -> Any:
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    pure = _mapping(payload.get("pure_walking"))
    for name, marker in (("r2", "o"), ("density", "s")):
        row = _mapping(_mapping(pure.get("observables")).get(name))
        lags = _vector(row.get("lags", row.get("lag_steps")))
        values = _vector(row.get("lag_values", row.get("values")))
        if lags.size and values.size == lags.size and values.ndim == 1:
            axis.plot(lags, values, marker=marker, label=name)
    axis.set(xlabel="forward-walking lag (steps)", ylabel="estimate", title="Lag dependence")
    if axis.lines:
        axis.legend()
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
