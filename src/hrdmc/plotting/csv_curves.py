"""Read numeric CSV curves and plot them without changing the supplied values."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.plotting.style import load_pyplot


@dataclass(frozen=True)
class CsvCurve:
    path: Path
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    error: NDArray[np.float64] | None
    edges: NDArray[np.float64] | None
    x_column: str
    y_column: str
    density: bool

    @property
    def label(self) -> str:
        return self.path.parent.name if self.path.stem == "density" else self.path.stem


def _columns(headers: list[str], x: str | None, y: str | None, error: str | None):
    if (x is None) != (y is None):
        raise ValueError("provide both --x and --y")
    if x is not None and y is not None:
        return x, y, error, False
    for cx, cy, ce in (
        ("q_center", "n_fw", "n_fw_sem"),
        ("q", "density", "stderr"),
        ("q", "n_lda", None),
    ):
        if cx in headers and cy in headers:
            return cx, cy, error or (ce if ce in headers else None), True
    raise ValueError("unrecognized CSV columns; specify --x and --y, and optionally --yerr")


def _numeric(rows: list[dict], column: str) -> NDArray[np.float64]:
    try:
        values = np.asarray([row[column] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must contain a number in every row") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} contains non-finite values")
    return values


def _bin_edges(rows: list[dict], headers: list[str], x: NDArray[np.float64]):
    if "q_left" not in headers and "q_right" not in headers:
        return None
    left, right = (_numeric(rows, name) for name in ("q_left", "q_right"))
    if np.any(right <= left) or not np.allclose(right[:-1], left[1:], rtol=1e-12, atol=1e-12):
        raise ValueError("q_left/q_right must describe positive-width, contiguous ordered bins")
    if np.any(x < left) or np.any(x > right):
        raise ValueError("each coordinate must lie inside its bin")
    return np.r_[left, right[-1]]


def read_csv_curve(
    path: Path, *, x: str | None = None, y: str | None = None, yerr: str | None = None
) -> CsvCurve:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows or len(headers) != len(set(headers)):
        raise ValueError(f"{path}: expected a header with unique names and at least one data row")
    cx, cy, ce, density = _columns(headers, x, y, yerr)
    values_x, values_y = _numeric(rows, cx), _numeric(rows, cy)
    if np.any(np.diff(values_x) <= 0):
        raise ValueError(f"{path}: x values must be strictly increasing")
    error = None
    if ce is not None:
        if ce not in headers:
            raise ValueError(f"{path}: missing error column {ce!r}")
        if not all(row[ce] == "" for row in rows):
            error = _numeric(rows, ce)
            if np.any(error < 0):
                raise ValueError(f"{path}: error values must be nonnegative")
    edges = _bin_edges(rows, headers, values_x) if cx in {"q", "q_center"} else None
    return CsvCurve(path, values_x, values_y, error, edges, cx, cy, density)


def _draw(axis: Any, curve: CsvCurve, *, label: str, reference: bool = False) -> None:
    style = "--" if reference else "-"
    if curve.edges is None:
        (artist,) = axis.plot(curve.x, curve.y, linestyle=style, label=label)
        color = artist.get_color()
    else:
        artist = axis.stairs(curve.y, curve.edges, linestyle=style, label=label)
        color = artist.get_edgecolor()
    if curve.error is None:
        return
    lower, upper = curve.y - curve.error, curve.y + curve.error
    if curve.edges is None:
        axis.fill_between(curve.x, lower, upper, color=color, alpha=0.2)
    else:
        axis.stairs(upper, curve.edges, baseline=lower, fill=True, color=color, alpha=0.2)


def plot_csv_curves(
    curves: list[CsvCurve],
    *,
    output: Path,
    references: list[CsvCurve] | None = None,
    panels: bool = False,
    labels: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> Path:
    if not curves:
        raise ValueError("provide at least one input CSV")
    if references is not None and len(references) != len(curves):
        raise ValueError("provide one --reference CSV per --input CSV, in the same order")
    if labels is not None and len(labels) != len(curves):
        raise ValueError("provide one --labels entry per --input CSV")
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("--output must end in .png, .pdf or .svg")
    if xlim is not None and (not all(map(math.isfinite, xlim)) or xlim[0] >= xlim[1]):
        raise ValueError("--xlim requires finite increasing bounds")
    all_curves = curves + (references or [])
    if output.resolve() in {curve.path.resolve() for curve in all_curves}:
        raise ValueError("output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = load_pyplot()
    count = len(curves) if panels else 1
    columns = min(3, count)
    fig, axes = plt.subplots(
        math.ceil(count / columns),
        columns,
        squeeze=False,
        figsize=(5 * columns, 3.8 * math.ceil(count / columns)),
    )
    try:
        _draw_panels(axes.flat, curves, references, panels, labels, xlabel, ylabel, xlim)
        for axis in list(axes.flat)[count:]:
            axis.set_visible(False)
        if title:
            fig.suptitle(title)
        fig.savefig(output)
    finally:
        plt.close(fig)
    return output.resolve()


def _draw_panels(axes, curves, references, panels, labels, xlabel, ylabel, xlim) -> None:
    for i, curve in enumerate(curves):
        axis = axes[i if panels else 0]
        label = labels[i] if labels else curve.label
        _draw(axis, curve, label=label)
        if references:
            reference = references[i]
            _draw(axis, reference, label=reference.label, reference=True)
        if panels:
            axis.set_title(label)
        axis.set_xlabel(xlabel or (r"$q=x/a_{\mathrm{ho}}$" if curve.density else curve.x_column))
        axis.set_ylabel(ylabel or (r"$a_{\mathrm{ho}}\,n(x)$" if curve.density else curve.y_column))
        if xlim:
            axis.set_xlim(*xlim)
        axis.legend()
