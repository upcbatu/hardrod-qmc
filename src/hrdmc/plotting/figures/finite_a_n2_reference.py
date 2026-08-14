from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.plotting.style import load_pyplot, save_figure


def write_finite_a_n2_reference_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Write compact N=2 scalar and density reference diagnostics."""
    output, plot_dir = Path(output_dir), Path(output_dir) / "plots"
    plt = load_pyplot(plot_dir)
    cases = [row for row in payload.get("case_results", []) if isinstance(row, dict)]
    paths: list[Path] = []
    if cases:
        figure = _scorecard(plt, payload, cases)
        paths.extend(save_figure(figure, plot_dir / "finite_a_n2_reference_scorecard", formats))
        plt.close(figure)
    for case in cases:
        case_id = str(case.get("case_id", "case"))
        for stem, figure in (
            (f"finite_a_n2_scalars_{case_id}", _scalars(plt, case)),
            (f"finite_a_n2_density_{case_id}", _density(plt, case)),
        ):
            if figure is not None:
                paths.extend(save_figure(figure, plot_dir / stem, formats))
                plt.close(figure)
    return [str(path.relative_to(output)) for path in paths]
def _scorecard(plt: Any, payload: dict[str, Any], cases: list[dict[str, Any]]) -> Any:
    metrics = (
        ("energy_abs_error", "energy_abs"),
        ("pure_r2_relative_error", "pure_r2_relative"),
        ("pure_rms_relative_error", "pure_rms_relative"),
        ("pure_density_relative_l2", "pure_density_l2"),
    )
    values = np.asarray(
        [
            [
                _table_value(case, key)
                / max(_number(_mapping(payload.get("tolerances")).get(tol)), 1e-300)
                for case in cases
            ]
            for key, tol in metrics
        ]
    )
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(cases)), 3.6))
    image = axis.imshow(
        np.clip(values, 0.0, 1.5), aspect="auto", vmin=0.0, vmax=1.5, cmap="RdYlGn_r"
    )
    axis.set(
        xticks=np.arange(len(cases)),
        xticklabels=[str(case.get("case_id", "")) for case in cases],
        yticks=np.arange(4),
        yticklabels=("energy", r"$R^2$", "RMS", "density"),
        title="error / declared tolerance",
    )
    figure.colorbar(image, ax=axis)
    return figure
def _scalars(plt: Any, case: dict[str, Any]) -> Any:
    comparison = _mapping(case.get("comparison"))
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for axis, (name, estimate_key) in zip(
        axes,
        (("energy", "dmc"), ("r2", "pure_fw"), ("rms", "pure_fw")),
        strict=True,
    ):
        row = _mapping(comparison.get(name))
        value, reference = _number(row.get(estimate_key)), _number(row.get("reference"))
        axis.scatter([0], [value], label="DMC")
        axis.axhline(reference, color="black", linestyle="--", label="exact")
        axis.set(title=name, xticks=[])
        axis.legend(fontsize=7)
    figure.suptitle(str(case.get("case_id", "")))
    return figure
def _density(plt: Any, case: dict[str, Any]) -> Any | None:
    density = _mapping(_mapping(case.get("comparison")).get("density"))
    x, reference = _vector(density.get("x")), _vector(density.get("reference_bin_averaged_n_x"))
    if not x.size or reference.size != x.size:
        return None
    pure, mixed = _vector(density.get("pure_fw_n_x")), _vector(density.get("mixed_diagnostic_n_x"))
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    axes[0].plot(x, reference, color="black", label="exact bin average")
    for values, label, style in ((pure, "forward walking", "-"), (mixed, "mixed", ":")):
        if values.size == x.size:
            axes[0].plot(x, values, style, label=label)
            axes[1].plot(x, values - reference, style, label=f"{label} - exact")
    axes[0].set(ylabel="n(x)", title=str(case.get("case_id", "")))
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel="x", ylabel="residual")
    return figure
def _table_value(case: dict[str, Any], key: str) -> float:
    row = _mapping(case.get("benchmark_packet")).get("case_table", [])
    if isinstance(row, list):
        for item in row:
            value = _number(_mapping(item).get(key))
            if np.isfinite(value):
                return value
    comparison = _mapping(case.get("comparison"))
    fallback = {
        "energy_abs_error": _mapping(comparison.get("energy")).get("abs_error"),
        "pure_r2_relative_error": _mapping(comparison.get("r2")).get("pure_relative_error"),
        "pure_rms_relative_error": _mapping(comparison.get("rms")).get("pure_relative_error"),
        "pure_density_relative_l2": _mapping(comparison.get("density")).get("pure_relative_l2"),
    }
    return _number(fallback.get(key))
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
