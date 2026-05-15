from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.plotting import tokens
from hrdmc.plotting.style import load_pyplot


def write_finite_a_n2_reference_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    paths.extend(_write_scorecard(plt, plot_dir, payload, formats=formats))
    for case in payload.get("case_results", []):
        paths.extend(_write_density_case(plt, plot_dir, payload, case, formats=formats))
        paths.extend(_write_scalar_case(plt, plot_dir, payload, case, formats=formats))
    return [str(path.relative_to(output)) for path in paths]


def _write_scorecard(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    cases = payload.get("case_results", [])
    if not cases:
        return []
    labels = [str(case.get("case_id", "")) for case in cases]
    metrics = [
        ("energy", "energy_abs_error", "energy_abs"),
        (r"$R^2$", "pure_r2_relative_error", "pure_r2_relative"),
        (r"$R_\mathrm{rms}$", "pure_rms_relative_error", "pure_rms_relative"),
        ("density", "pure_density_relative_l2", "pure_density_l2"),
    ]
    ratios = np.empty((len(metrics), len(cases)), dtype=float)
    for row, (_title, value_key, tolerance_key) in enumerate(metrics):
        tolerance = _payload_tolerance(payload, tolerance_key)
        for col, case in enumerate(cases):
            value = _case_table_float(case, value_key)
            ratios[row, col] = value / tolerance if tolerance > 0.0 else np.nan

    fig, ax = plt.subplots(
        figsize=(max(6.2, 1.5 * len(labels)), 3.9),
        constrained_layout=False,
    )
    _disable_auto_layout(fig)
    _draw_header(fig, payload, "Finite-a N=2 exact-reference scorecard")
    fig.subplots_adjust(top=0.76, bottom=0.22)
    clipped = np.clip(ratios, 0.0, 1.5)
    image = ax.imshow(clipped, aspect="auto", vmin=0.0, vmax=1.5, cmap="RdYlGn_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([metric[0] for metric in metrics])
    ax.set_title("error / tolerance; green means below declared gate")
    for row in range(ratios.shape[0]):
        for col in range(ratios.shape[1]):
            ratio = ratios[row, col]
            label = "n/a" if not np.isfinite(ratio) else f"{ratio:.2g}x"
            ax.text(col, row, label, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("error / tolerance")
    _disable_auto_layout(fig)
    paths = _save_figure(fig, plot_dir / "finite_a_n2_reference_scorecard", formats)
    plt.close(fig)
    return paths


def _write_scalar_case(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    case: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    comparison = case.get("comparison", {})
    entries = [
        ("Energy", comparison.get("energy", {}), "dmc", "reference", "abs_error"),
        (r"$R^2$", comparison.get("r2", {}), "pure_fw", "reference", "pure_relative_error"),
        (
            r"$R_\mathrm{rms}$",
            comparison.get("rms", {}),
            "pure_fw",
            "reference",
            "pure_relative_error",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), constrained_layout=False)
    _disable_auto_layout(fig)
    _draw_header(fig, payload, str(case.get("case_id", "")))
    fig.subplots_adjust(top=0.73, bottom=0.16, wspace=0.32)
    for ax, (title, data, value_key, reference_key, error_key) in zip(
        axes,
        entries,
        strict=True,
    ):
        value = _float_or_nan(data.get(value_key))
        reference = _float_or_nan(data.get(reference_key))
        error = _float_or_nan(data.get(error_key))
        ax.axhline(
            reference,
            color=tokens.INK,
            linestyle=(0, (5, 2)),
            linewidth=1.3,
            label="N=2 reference",
        )
        ax.scatter([0], [value], color=tokens.DMC_PRIMARY, s=42, zorder=3, label="packet")
        if title != "Energy":
            mixed = _float_or_nan(data.get("mixed_diagnostic"))
            if np.isfinite(mixed):
                ax.scatter(
                    [0.18],
                    [mixed],
                    color=tokens.DMC_DIAGNOSTIC,
                    marker="x",
                    s=44,
                    linewidths=1.5,
                    zorder=4,
                    label="mixed diagnostic",
                )
        ax.set_xlim(-0.45, 0.55)
        _set_local_ylim(ax, value, reference)
        ax.set_xticks([])
        ax.set_title(title)
        ax.text(
            0.02,
            0.04,
            _error_label(error, absolute=(title == "Energy")),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=tokens.INK_SOFT,
        )
        ax.legend(loc="best", fontsize=7)
    filename = f"finite_a_n2_scalars_{case.get('case_id', 'case')}"
    _disable_auto_layout(fig)
    paths = _save_figure(fig, plot_dir / filename, formats)
    plt.close(fig)
    return paths


def _write_density_case(
    plt: Any,  # noqa: ANN401
    plot_dir: Path,
    payload: dict[str, Any],
    case: dict[str, Any],
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    density = case.get("comparison", {}).get("density", {})
    x = _array_or_none(density.get("x"))
    edges = _array_or_none(density.get("bin_edges"))
    reference = _array_or_none(density.get("reference_bin_averaged_n_x"))
    pure = _array_or_none(density.get("pure_fw_n_x"))
    mixed = _array_or_none(density.get("mixed_diagnostic_n_x"))
    if x is None or edges is None or reference is None:
        return []
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.2),
        gridspec_kw={"height_ratios": [2.75, 1.35]},
        sharex=True,
        constrained_layout=False,
    )
    _disable_auto_layout(fig)
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.12)
    ax = axes[0]
    residual_ax = axes[1]
    _plot_density_curve(
        ax,
        x,
        reference,
        bin_edges=edges,
        color=tokens.INK,
        linestyle="-",
        marker="s",
        label="N=2 reference bin average (gate)",
        linewidth=2.25,
        zorder=5,
    )
    if pure is not None:
        _plot_density_curve(
            ax,
            x,
            pure,
            bin_edges=edges,
            color=tokens.DMC_PRIMARY,
            linestyle="-",
            marker="o",
            label="transported FW (bin)",
            linewidth=1.95,
            zorder=4,
        )
    if mixed is not None:
        _plot_density_curve(
            ax,
            x,
            mixed,
            bin_edges=edges,
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            marker="x",
            label="mixed diagnostic (bin)",
            linewidth=1.7,
            zorder=3,
        )
        _plot_density_curve(
            residual_ax,
            x,
            mixed - reference,
            bin_edges=edges,
            color=tokens.DMC_DIAGNOSTIC,
            linestyle=(0, (1, 2)),
            marker="x",
            label="mixed - reference bin average",
            linewidth=1.6,
        )
    if pure is not None:
        _plot_density_curve(
            residual_ax,
            x,
            pure - reference,
            bin_edges=edges,
            color=tokens.DMC_PRIMARY,
            linestyle="-",
            marker="o",
            label="FW - reference bin average",
            linewidth=1.7,
        )
    _set_density_xlim(ax, residual_ax, x, reference, pure, mixed)
    _set_residual_ylim(residual_ax, reference, pure, mixed)
    ax.set_ylabel(r"$n(x)$")
    ax.set_title(
        "Density comparison: finite-a N=2 exact reference",
        loc="left",
        pad=6,
    )
    ax.legend(loc="upper right", fontsize=8)
    _draw_density_metrics_box(residual_ax, case)
    residual_ax.axhline(0.0, color=tokens.INK_SOFT, linewidth=0.8)
    residual_ax.set_xlabel(r"$x$")
    residual_ax.set_ylabel(r"$n_\mathrm{est}-n_\mathrm{ref}$")
    residual_ax.set_title("Residual to exact bin average", loc="left", pad=3, fontsize=8.5)
    residual_ax.legend(loc="upper right", fontsize=7)
    filename = f"finite_a_n2_density_{case.get('case_id', 'case')}"
    _disable_auto_layout(fig)
    paths = _save_figure(fig, plot_dir / filename, formats)
    plt.close(fig)
    return paths


def _plot_density_curve(
    ax: Any,  # noqa: ANN401
    x: np.ndarray,
    values: np.ndarray,
    *,
    bin_edges: np.ndarray | None,
    color: str,
    linestyle: object,
    marker: str | None,
    label: str,
    linewidth: float = 1.9,
    zorder: int = 2,
) -> None:
    if x.shape != values.shape:
        return
    marker_every = max(1, x.size // 28)
    if bin_edges is not None and bin_edges.size == values.size + 1:
        ax.stairs(
            values,
            bin_edges,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )
        if marker is not None:
            ax.plot(
                x,
                values,
                color=color,
                linestyle="None",
                marker=marker,
                markevery=marker_every,
                markersize=3.4,
                alpha=0.82,
                zorder=zorder,
            )
        return
    ax.plot(
        x,
        values,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=marker_every if marker is not None else None,
        markersize=3.7 if marker is not None else 0.0,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )


def _draw_header(fig: Any, payload: dict[str, Any], title: str) -> None:  # noqa: ANN401
    controls = payload.get("controls", {})
    subtitle = (
        f"{payload.get('status')} | "
        f"dt={controls.get('dt')} M={controls.get('walkers')} "
        f"prod_tau={controls.get('production_tau')} "
        f"seeds={','.join(str(seed) for seed in payload.get('seeds', []))}"
    )
    fig.suptitle(title, fontsize=12, fontweight="semibold", y=0.965)
    fig.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
        color=tokens.INK_SOFT,
    )


def _disable_auto_layout(fig: Any) -> None:  # noqa: ANN401
    """Allow explicit report-style margins despite global constrained layout."""

    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("none")
    elif hasattr(fig, "set_constrained_layout"):
        fig.set_constrained_layout(False)


def _save_figure(fig: Any, base_path: str | Path, formats: tuple[str, ...]) -> list[Path]:  # noqa: ANN401
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        _disable_auto_layout(fig)
        suffix = fmt.lower().lstrip(".")
        path = base.parent / f"{base.name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    return paths


def _draw_density_metrics_box(ax: Any, case: dict[str, Any]) -> None:  # noqa: ANN401
    ax.text(
        0.012,
        0.95,
        _density_status_text(case),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=tokens.INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": tokens.INK_FAINT,
            "alpha": 0.92,
        },
    )


def _density_status_text(case: dict[str, Any]) -> str:
    density = case.get("comparison", {}).get("density", {})
    l2 = _float_or_nan(density.get("pure_relative_l2"))
    mixed_l2 = _float_or_nan(density.get("mixed_relative_l2_diagnostic"))
    accounting = _float_or_nan(density.get("density_accounting_abs_error"))
    fw_closer = bool(density.get("fw_closer_than_mixed_diagnostic"))
    return (
        f"FW rel-L2={_number(l2)}   mixed rel-L2={_number(mixed_l2)}\n"
        f"FW closer on density: {'yes' if fw_closer else 'no'}   "
        f"density accounting abs={_number(accounting)}\n"
        "Gate comparison uses histogram-bin averages."
    )


def _set_density_xlim(
    ax: Any,  # noqa: ANN401
    residual_ax: Any,  # noqa: ANN401
    x: np.ndarray,
    reference: np.ndarray,
    pure: np.ndarray | None,
    mixed: np.ndarray | None,
) -> None:
    series = [reference]
    if pure is not None:
        series.append(pure)
    if mixed is not None:
        series.append(mixed)
    envelope = np.maximum.reduce([np.abs(values) for values in series])
    threshold = 0.003 * float(np.max(envelope)) if envelope.size else 0.0
    active = np.flatnonzero(envelope > threshold)
    if active.size < 2:
        return
    left = max(0, int(active[0]) - 6)
    right = min(x.size - 1, int(active[-1]) + 6)
    ax.set_xlim(float(x[left]), float(x[right]))
    residual_ax.set_xlim(float(x[left]), float(x[right]))


def _set_residual_ylim(
    ax: Any,  # noqa: ANN401
    reference: np.ndarray,
    pure: np.ndarray | None,
    mixed: np.ndarray | None,
) -> None:
    residuals: list[np.ndarray] = []
    if pure is not None and pure.shape == reference.shape:
        residuals.append(pure - reference)
    if mixed is not None and mixed.shape == reference.shape:
        residuals.append(mixed - reference)
    if not residuals:
        return
    max_abs = max(float(np.max(np.abs(values))) for values in residuals)
    if max_abs <= 0.0:
        return
    ax.set_ylim(-1.25 * max_abs, 1.25 * max_abs)


def _case_table_float(case: dict[str, Any], key: str) -> float:
    for row in case.get("benchmark_packet", {}).get("case_table", []):
        value = row.get(key)
        parsed = _float_or_nan(value)
        if np.isfinite(parsed):
            return parsed
    table_row = _case_result_table_row(case)
    return _float_or_nan(table_row.get(key))


def _case_result_table_row(case: dict[str, Any]) -> dict[str, Any]:
    comparison = case.get("comparison", {})
    return {
        "energy_abs_error": comparison.get("energy", {}).get("abs_error"),
        "pure_r2_relative_error": comparison.get("r2", {}).get("pure_relative_error"),
        "pure_rms_relative_error": comparison.get("rms", {}).get("pure_relative_error"),
        "pure_density_relative_l2": comparison.get("density", {}).get("pure_relative_l2"),
    }


def _payload_tolerance(payload: dict[str, Any], key: str) -> float:
    return _float_or_nan(payload.get("tolerances", {}).get(key))


def _array_or_none(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def _float_or_nan(value: object) -> float:
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _set_local_ylim(ax: Any, value: float, reference: float) -> None:  # noqa: ANN401
    finite = [item for item in (value, reference) if np.isfinite(item)]
    if not finite:
        return
    low = min(finite)
    high = max(finite)
    span = max(high - low, 0.02 * max(abs(high), abs(low), 1.0))
    ax.set_ylim(low - 1.6 * span, high + 1.6 * span)


def _error_label(error: float, *, absolute: bool) -> str:
    if not np.isfinite(error):
        return "error unavailable"
    return f"abs error={error:.3g}" if absolute else f"relative error={error:.3%}"


def _number(value: float) -> str:
    return f"{value:.4g}" if np.isfinite(value) else "unavailable"
