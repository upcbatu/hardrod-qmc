from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.style import load_pyplot, save_figure

PacketEntry = tuple[str, Path, dict[str, Any]]


def write_benchmark_packet_comparison_plots(
    output_dir: str | Path,
    packet_entries: list[PacketEntry],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    paths.extend(_write_energy_running_mean(plt, plot_dir, packet_entries, formats=formats))
    paths.extend(_write_energy_block_residuals(plt, plot_dir, packet_entries, formats=formats))
    paths.extend(_write_fw_r2_lag_ladder(plt, plot_dir, packet_entries, formats=formats))
    paths.extend(_write_density_comparison(plt, plot_dir, packet_entries, formats=formats))
    return [str(path.relative_to(output)) for path in paths]


def _write_energy_running_mean(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    packet_entries: list[PacketEntry],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    traces_by_packet: list[list[tuple[str, np.ndarray, np.ndarray]]] = []
    all_running: list[float] = []
    max_tau = 0.0
    for _, root, payload in packet_entries:
        traces: list[tuple[str, np.ndarray, np.ndarray]] = []
        for path in _trace_paths(root, payload):
            tau, running = _running_energy_trace(path)
            if tau.size == 0:
                continue
            traces.append((_seed_from_trace_path(path), tau, running))
            all_running.extend(running.tolist())
            max_tau = max(max_tau, float(np.max(tau)))
        traces_by_packet.append(traces)
    if not all_running:
        return []
    ymin, ymax = _padded_limits(all_running, min_pad=0.005)
    fig, axes = plt.subplots(
        1,
        len(packet_entries),
        figsize=(5.6 * len(packet_entries), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, _, payload), traces in zip(
        axes,
        packet_entries,
        traces_by_packet,
        strict=True,
    ):
        final_mean = _finite_float(payload.get("stationarity", {}).get("mixed_energy"))
        for seed, tau, running in traces:
            stride = max(1, tau.size // 900)
            ax.plot(
                tau[::stride],
                running[::stride],
                linewidth=1.0,
                alpha=0.42,
                label=f"seed {seed}",
            )
        if np.isfinite(final_mean):
            ax.axhline(
                final_mean,
                color=tokens.INK,
                linestyle=(0, (4, 2)),
                linewidth=1.0,
                label="final mean",
            )
        ax.set_title(label)
        ax.set_xlabel("production time")
        ax.set_xlim(0.0, max_tau)
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("running mean energy")
    axes[-1].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle("Energy running means, shared y-axis")
    paths = save_figure(fig, plot_dir / "energy_running_mean_shared_y", formats)
    plt.close(fig)
    return paths


def _write_energy_block_residuals(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    packet_entries: list[PacketEntry],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    rows_by_packet: list[list[tuple[str, np.ndarray, np.ndarray, bool]]] = []
    all_residuals: list[float] = []
    max_time = 0.0
    for _, _, payload in packet_entries:
        stationarity = payload.get("stationarity", {})
        final_mean = _finite_float(stationarity.get("mixed_energy"))
        production_tau = _finite_float(payload.get("controls", {}).get("production_tau"))
        seeds = stationarity.get("seeds", [])
        chain_rows = (
            stationarity.get("diagnostics", {}).get("energy", {}).get("chain_diagnostics", [])
        )
        case_rows: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
        for index, row in enumerate(chain_rows if isinstance(chain_rows, list) else []):
            if not isinstance(row, dict):
                continue
            blocks = np.asarray(row.get("block_means", []), dtype=float)
            blocks = blocks[np.isfinite(blocks)]
            if blocks.size == 0:
                continue
            reference = final_mean if np.isfinite(final_mean) else float(np.mean(blocks))
            x = np.arange(1, blocks.size + 1, dtype=float)
            if np.isfinite(production_tau):
                x = (x - 0.5) * production_tau / blocks.size
                max_time = max(max_time, float(production_tau))
            seed = str(seeds[index]) if isinstance(seeds, list) and index < len(seeds) else ""
            residuals = blocks - reference
            all_residuals.extend(residuals.tolist())
            case_rows.append((seed, x, residuals, bool(row.get("spread_warning", False))))
        rows_by_packet.append(case_rows)
    if not all_residuals:
        return []
    limit = max(abs(float(np.min(all_residuals))), abs(float(np.max(all_residuals))), 0.001) * 1.12
    fig, axes = plt.subplots(
        1,
        len(packet_entries),
        figsize=(5.6 * len(packet_entries), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, _, _), rows in zip(axes, packet_entries, rows_by_packet, strict=True):
        for seed, x, residuals, warning in rows:
            ax.plot(
                x,
                residuals,
                marker="o",
                linewidth=1.8 if warning else 1.0,
                alpha=0.95 if warning else 0.48,
                label=f"seed {seed}",
            )
        ax.axhline(0.0, color=tokens.INK, linestyle=(0, (4, 2)), linewidth=0.9)
        ax.set_title(label)
        ax.set_xlabel("production time")
        if max_time > 0.0:
            ax.set_xlim(0.0, max_time)
        ax.set_ylim(-limit, limit)
    axes[0].set_ylabel(r"block mean energy $-\bar E$")
    axes[-1].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle("Energy block residuals, shared y-axis")
    paths = save_figure(fig, plot_dir / "energy_block_residuals_shared_y", formats)
    plt.close(fig)
    return paths


def _write_fw_r2_lag_ladder(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    packet_entries: list[PacketEntry],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    data_by_packet = []
    all_values: list[float] = []
    for _, _, payload in packet_entries:
        pure = payload.get("pure_walking", {})
        diagnostics = pure.get("r2_aggregate_plateau_diagnostics", {})
        lags, values = _lag_dict_arrays(diagnostics.get("values_by_lag", {}))
        _, stderr = _lag_dict_arrays(diagnostics.get("stderr_by_lag", {}))
        seed_curves = []
        all_values.extend(values.tolist())
        for seed_payload in payload.get("seed_results", []):
            r2 = seed_payload.get("pure_walking", {}).get("observable_results", {}).get("r2", {})
            seed_lags, seed_values = _lag_dict_arrays(r2.get("values_by_lag", {}))
            if seed_values.size:
                seed_curves.append((seed_lags, seed_values))
                all_values.extend(seed_values.tolist())
        data_by_packet.append(
            (
                lags,
                values,
                stderr,
                pure.get("r2_aggregate_plateau_value"),
                pure.get("r2_aggregate_plateau_status"),
                seed_curves,
            )
        )
    if not all_values:
        return []
    ymin, ymax = _padded_limits(all_values, min_pad=0.02)
    fig, axes = plt.subplots(
        1,
        len(packet_entries),
        figsize=(5.6 * len(packet_entries), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, _, _), (lags, values, stderr, plateau, status, seed_curves) in zip(
        axes,
        packet_entries,
        data_by_packet,
        strict=True,
    ):
        for seed_lags, seed_values in seed_curves:
            ax.plot(
                seed_lags,
                seed_values,
                color=tokens.INK_SOFT,
                alpha=0.22,
                linewidth=0.9,
                marker=".",
                markersize=3,
            )
        ax.plot(
            lags,
            values,
            color=tokens.DMC_PRIMARY,
            linewidth=2.0,
            marker="o",
            label="aggregate",
        )
        if stderr.size == values.size:
            ax.fill_between(
                lags,
                values - stderr,
                values + stderr,
                color=tokens.SEED_BAND,
                alpha=0.28,
                linewidth=0,
            )
        plateau_value = _finite_float(plateau)
        if np.isfinite(plateau_value):
            ax.axhline(
                plateau_value,
                color=tokens.ACCEPTED,
                linestyle=(0, (4, 2)),
                linewidth=1.1,
                label="plateau",
            )
        ax.text(
            0.02,
            0.05,
            str(status),
            transform=ax.transAxes,
            fontsize=8,
            color=tokens.INK_SOFT,
        )
        ax.set_title(label)
        ax.set_xlabel("lag steps")
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel(r"FW $R^2$")
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle(r"FW $R^2$ lag ladders, shared y-axis")
    paths = save_figure(fig, plot_dir / "fw_r2_lag_shared_y", formats)
    plt.close(fig)
    return paths


def _write_density_comparison(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    packet_entries: list[PacketEntry],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    data_by_packet = []
    all_x: list[float] = []
    all_y: list[float] = []
    for _, _, payload in packet_entries:
        density = payload.get("estimates", {}).get("density", {})
        x = np.asarray(density.get("x", []), dtype=float)
        value = np.asarray(density.get("value", []), dtype=float)
        mixed_x = np.asarray(density.get("mixed_diagnostic_x", []), dtype=float)
        mixed_value = np.asarray(density.get("mixed_diagnostic_value", []), dtype=float)
        lda_x = np.asarray(density.get("lda_x", []), dtype=float)
        lda_value = np.asarray(density.get("lda_value", []), dtype=float)
        data_by_packet.append((x, value, mixed_x, mixed_value, lda_x, lda_value))
        for array in (x, mixed_x, lda_x):
            all_x.extend(array[np.isfinite(array)].tolist())
        for array in (value, mixed_value, lda_value):
            all_y.extend(array[np.isfinite(array)].tolist())
    if not all_x or not all_y:
        return []
    ymax = float(np.max(all_y))
    fig, axes = plt.subplots(
        1,
        len(packet_entries),
        figsize=(5.9 * len(packet_entries), 4.4),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, _, _), (x, value, mixed_x, mixed_value, lda_x, lda_value) in zip(
        axes,
        packet_entries,
        data_by_packet,
        strict=True,
    ):
        if lda_x.size and lda_value.size:
            ax.plot(
                lda_x,
                lda_value,
                color=tokens.LDA_PREDICTION,
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                label="LDA",
            )
        if x.size and value.size:
            ax.step(x, value, where="mid", color=tokens.DMC_PRIMARY, linewidth=1.8, label="FW")
        if mixed_x.size and mixed_value.size:
            ax.step(
                mixed_x,
                mixed_value,
                where="mid",
                color=tokens.DMC_DIAGNOSTIC,
                linestyle=(0, (1, 2)),
                linewidth=1.45,
                alpha=0.9,
                label="mixed",
            )
        ax.set_title(label)
        ax.set_xlabel(r"$q$")
        ax.set_xlim(float(np.min(all_x)), float(np.max(all_x)))
        ax.set_ylim(0.0, ymax * 1.06)
    axes[0].set_ylabel(r"$n(q)$")
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle("Density comparison, shared axes")
    paths = save_figure(fig, plot_dir / "density_shared_axes", formats)
    plt.close(fig)
    return paths


def _trace_paths(root: Path, payload: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for item in payload.get("stationarity", {}).get("trace_artifacts", []):
        if not isinstance(item, str) or not item.endswith("_trace.csv"):
            continue
        path = Path(item)
        if not path.exists():
            path = root / item
        if path.exists():
            out.append(path)
    return out


def _running_energy_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0 or data.dtype.names is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if "tau" not in data.dtype.names or "mixed_energy" not in data.dtype.names:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    tau = np.atleast_1d(np.asarray(data["tau"], dtype=float))
    energy = np.atleast_1d(np.asarray(data["mixed_energy"], dtype=float))
    finite = np.isfinite(tau) & np.isfinite(energy)
    tau = tau[finite]
    energy = energy[finite]
    if tau.size == 0:
        return tau, energy
    running = np.cumsum(energy) / np.arange(1, energy.size + 1, dtype=float)
    return tau, running


def _lag_dict_arrays(values_by_lag: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(values_by_lag, dict):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    pairs: list[tuple[int, float]] = []
    for key, value in values_by_lag.items():
        try:
            lag = int(key)
            scalar = float(np.asarray(value, dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(scalar):
            pairs.append((lag, scalar))
    pairs.sort()
    return (
        np.asarray([lag for lag, _ in pairs], dtype=float),
        np.asarray([value for _, value in pairs], dtype=float),
    )


def _padded_limits(values: list[float], *, min_pad: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    pad = max(min_pad, 0.06 * (ymax - ymin))
    return ymin - pad, ymax + pad


def _finite_float(value: object) -> float:
    try:
        out = float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _seed_from_trace_path(path: Path) -> str:
    marker = "_seed"
    if marker not in path.stem:
        return ""
    return path.stem.split(marker, maxsplit=1)[1].split("_", maxsplit=1)[0]
