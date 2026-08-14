#!/usr/bin/env python3
# ruff: noqa: I001
"""Generate the supervisor-report figures from the assembled DMC matrix."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

os.environ["MPLBACKEND"] = os.environ.get("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from hrdmc.statistics.density import unit_mass_shell_average

ASSEMBLY_SCHEMA = "dmc_final_matrix_assembly_v1"
SOURCE_SCHEMA = "dmc_benchmark_packet_v3"
CASE_ORDER = (
    "N10_A0",
    "N10_A0.1",
    "N10_A1",
    "N10_A10",
    "N20_A0",
    "N20_A0.1",
    "N20_A1",
    "N20_A10",
)
ROD_LENGTH_LABELS = ("0", "0.1", "1", "10")
COLORS = {10: "#0072B2", 20: "#D55E00"}  # Okabe--Ito blue and vermillion
MARKERS = {10: "o", 20: "s"}
LINESTYLES = {10: "-", 20: "--"}
RMS_FW_EQUIVALENCE_MARGIN = 1.0e-3
DENSITY_FW_EQUIVALENCE_MARGIN = 3.0e-2


@dataclass(frozen=True)
class DensityProfile:
    case: str
    n_particles: int
    rod_length_ho: float
    bin_edges: np.ndarray
    x: np.ndarray
    value: np.ndarray
    stderr: np.ndarray
    seed_values: np.ndarray
    lda_x: np.ndarray
    lda_value: np.ndarray


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_values(case: str) -> tuple[int, float]:
    _expect(case.startswith("N") and "_A" in case, f"invalid case id: {case}")
    n_text, rod_text = case[1:].split("_A", maxsplit=1)
    return int(n_text), float(rod_text)


def _as_finite_vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _expect(array.ndim == 1, f"{name} must be one-dimensional")
    if length is not None:
        _expect(array.size == length, f"{name} has length {array.size}; expected {length}")
    _expect(array.size > 1, f"{name} must contain at least two values")
    _expect(bool(np.all(np.isfinite(array))), f"{name} contains non-finite values")
    return array


def _load_sources(
    assembly_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, DensityProfile]]:
    assembly = _read_json(assembly_path)
    _expect(assembly.get("status") == "accepted", "assembled matrix is not accepted")
    _expect(tuple(assembly.get("case_order", ())) == CASE_ORDER, "unexpected case order")
    rows = assembly.get("rows")
    _expect(isinstance(rows, list) and len(rows) == len(CASE_ORDER), "expected eight rows")
    rows = cast(list[dict[str, Any]], rows)
    row_by_case = {str(row.get("case")): row for row in rows}
    _expect(tuple(row_by_case) == CASE_ORDER, "row order does not match case order")
    density_by_case: dict[str, DensityProfile] = {}
    for case in CASE_ORDER:
        row = row_by_case[case]
        _expect(row.get("status") == "accepted", f"{case}: assembled status is not accepted")
        source_path = _source_path(assembly_path, row, "primary_source", case)
        source = _read_json(source_path)
        _expect(source.get("case_id") == case, f"{case}: source case mismatch")
        n_particles, rod_length_ho = _case_values(case)
        _expect(source.get("n_particles") == n_particles, f"{case}: particle-count mismatch")
        _expect(
            math.isclose(
                float(cast(Any, source.get("rod_length_ho"))), rod_length_ho, abs_tol=1e-12
            ),
            f"{case}: rod-length mismatch",
        )
        estimates = _mapping(source.get("estimates"), f"{case}.estimates")
        density = _mapping(estimates.get("density"), f"{case}.density")
        _expect(density.get("status") == "accepted", f"{case}: density status is not accepted")
        x = _as_finite_vector(density.get("x"), name=f"{case}.density.x")
        value = _as_finite_vector(density.get("value"), name=f"{case}.density.value", length=x.size)
        stderr = _as_finite_vector(
            density.get("stderr"), name=f"{case}.density.stderr", length=x.size
        )
        lda_x = _as_finite_vector(density.get("lda_x"), name=f"{case}.density.lda_x")
        lda_value = _as_finite_vector(
            density.get("lda_value"), name=f"{case}.density.lda_value", length=lda_x.size
        )
        bin_edges = _as_finite_vector(
            density.get("bin_edges"), name=f"{case}.density.bin_edges", length=x.size + 1
        )
        widths = np.diff(bin_edges)
        _expect(bool(np.all(widths > 0.0)), f"{case}: density bin edges are not increasing")
        _expect(
            bool(np.allclose(widths, widths[0], rtol=5e-11, atol=5e-13)),
            f"{case}: density bins are not uniform",
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        _expect(
            bool(np.allclose(x, centers, rtol=5e-11, atol=5e-11)),
            f"{case}: x is not bin-centred",
        )
        _expect(bool(np.all(value >= 0.0)), f"{case}: negative FW density")
        _expect(bool(np.all(stderr >= 0.0)), f"{case}: negative density standard error")
        _expect(bool(np.all(lda_value >= 0.0)), f"{case}: negative LDA density")
        seed_results = source.get("seed_results")
        _expect(
            isinstance(seed_results, list)
            and len(seed_results) == int(cast(Any, source.get("seed_count"))),
            f"{case}: missing seed-level density results",
        )
        seed_results = cast(list[dict[str, Any]], seed_results)
        seed_values = _seed_density_values(case, seed_results, x.size)
        _expect(bool(np.all(seed_values >= 0.0)), f"{case}: negative seed-level FW density")
        _expect(
            bool(np.allclose(seed_values.mean(axis=0), value, rtol=0.0, atol=5e-13)),
            f"{case}: aggregate FW density is not the seed mean",
        )
        integral = float(np.sum(value * widths))
        recorded_integral = float(cast(Any, density.get("integral")))
        _expect(
            math.isclose(integral, recorded_integral, rel_tol=2e-11, abs_tol=2e-11),
            f"{case}: plotted density integral differs from source integral",
        )
        _expect(
            math.isclose(integral, n_particles, rel_tol=0.0, abs_tol=5e-3),
            f"{case}: density integral {integral:.8g} differs from N={n_particles}",
        )
        density_by_case[case] = DensityProfile(
            case=case,
            n_particles=n_particles,
            rod_length_ho=rod_length_ho,
            bin_edges=bin_edges,
            x=x,
            value=value,
            stderr=stderr,
            seed_values=seed_values,
            lda_x=lda_x,
            lda_value=lda_value,
        )
    return assembly, row_by_case, density_by_case


def _seed_density_values(
    case: str, seed_results: list[dict[str, Any]], bin_count: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, seed_result in enumerate(seed_results):
        pure = _mapping(seed_result.get("pure_walking"), f"{case}.seed[{index}].pure")
        observables = _mapping(pure.get("observable_results"), "observable_results")
        density = _mapping(observables.get("density"), "density")
        _expect(
            density.get("plateau_status") == "plateau_resolved",
            f"{case}: seed {index} has no resolved density plateau",
        )
        rows.append(
            _as_finite_vector(
                density.get("plateau_value"),
                name=f"{case}.seed_density[{index}]",
                length=bin_count,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _source_path(assembly_path: Path, row: dict[str, Any], key: str, case: str) -> Path:
    locator = _mapping(row.get(key), f"{case}.{key}")
    relative = locator.get("summary_path")
    _expect(isinstance(relative, str), f"{case}: missing {key} summary path")
    path = (assembly_path.parent / cast(str, relative)).resolve()
    _expect(path.is_file(), f"{case}: source summary is missing: {path}")
    _expect(_sha256(path) == locator.get("summary_sha256"), f"{case}: source hash mismatch")
    return path


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_pair(fig: Figure, output_dir: Path, stem: str, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": title, "Creator": "hardrod-qmc report figure generator"}
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


def _plot_lda_comparison(rows: dict[str, dict[str, Any]], output_dir: Path) -> None:
    x = np.arange(len(ROD_LENGTH_LABELS), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.15), sharex=True)
    absolute_axis = axes[0]
    absolute_handles: list[Line2D] = []
    for n_particles in (10, 20):
        cases = [f"N{n_particles}_A{label}" for label in ROD_LENGTH_LABELS]
        energies = np.asarray([rows[case]["energy"] for case in cases], dtype=float)
        errors = np.asarray([rows[case]["energy_stderr"] for case in cases], dtype=float)
        references = np.asarray([rows[case]["energy_lda"] for case in cases], dtype=float)
        references[0] = n_particles**2 / 2.0
        absolute_axis.plot(
            x,
            references,
            color=COLORS[n_particles],
            linestyle=LINESTYLES[n_particles],
            linewidth=1.25,
            alpha=0.82,
            zorder=2,
        )
        absolute_axis.errorbar(
            x,
            energies,
            yerr=errors,
            color=COLORS[n_particles],
            marker=MARKERS[n_particles],
            linestyle="none",
            capsize=2.0,
            elinewidth=0.8,
            markeredgewidth=0.8,
            markerfacecolor="white" if n_particles == 20 else COLORS[n_particles],
            zorder=3,
        )
        absolute_handles.extend(
            (
                Line2D(
                    [],
                    [],
                    color=COLORS[n_particles],
                    marker=MARKERS[n_particles],
                    linestyle="none",
                    markerfacecolor="white" if n_particles == 20 else COLORS[n_particles],
                    label=rf"DMC, $N={n_particles}$",
                ),
                Line2D(
                    [],
                    [],
                    color=COLORS[n_particles],
                    linestyle=LINESTYLES[n_particles],
                    label=rf"LDA, $N={n_particles}$",
                ),
            )
        )
    absolute_axis.set_yscale("log")
    absolute_axis.set_ylabel(r"Energy  $E/(\hbar\omega)$")
    absolute_axis.set_title("(a) Full energy", loc="left", fontweight="semibold", pad=7)
    absolute_axis.legend(
        handles=absolute_handles,
        frameon=False,
        fontsize=6.9,
        ncol=1,
        loc="upper left",
        borderaxespad=0.2,
        handlelength=1.7,
        labelspacing=0.25,
    )
    fields = (
        (
            "energy",
            "energy_relative_delta_vs_lda",
            "energy_stderr",
            "energy_lda",
            "(b) Energy deviation",
        ),
        (
            "rms_radius",
            "rms_relative_delta_vs_lda",
            "rms_mc_statistical_stderr",
            "rms_lda",
            "(c) RMS-radius deviation",
        ),
    )
    for axis, (estimate_key, value_key, error_key, reference_key, title) in zip(
        axes[1:], fields, strict=True
    ):
        for n_particles in (10, 20):
            cases = [f"N{n_particles}_A{label}" for label in ROD_LENGTH_LABELS]
            values = 100.0 * np.asarray([rows[case][value_key] for case in cases], dtype=float)
            references = np.asarray([rows[case][reference_key] for case in cases], dtype=float)
            # At zero rod length the Tonks--Girardeau anchors are analytic.
            # Use those exact reference values instead of retaining the tiny
            # LDA grid/quadrature residual present in the serialized artifact.
            references[0] = (
                n_particles**2 / 2.0 if estimate_key == "energy" else math.sqrt(n_particles / 2.0)
            )
            values[0] = 100.0 * (rows[cases[0]][estimate_key] / references[0] - 1.0)
            if estimate_key == "energy":
                _expect(abs(values[0]) < 1e-12, f"{cases[0]}: exact TG energy anchor mismatch")
            errors = (
                100.0
                * np.asarray([rows[case][error_key] for case in cases], dtype=float)
                / references
            )
            axis.errorbar(
                x,
                values,
                yerr=errors,
                color=COLORS[n_particles],
                marker=MARKERS[n_particles],
                linestyle=LINESTYLES[n_particles],
                capsize=2.2,
                elinewidth=0.9,
                markeredgewidth=0.8,
                markerfacecolor="white" if n_particles == 20 else COLORS[n_particles],
                label=rf"$N={n_particles}$",
                zorder=3,
            )
        axis.axhline(0.0, color="#666666", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
        axis.set_title(title, loc="left", fontweight="semibold", pad=7)
    axes[1].set_ylabel(r"Relative deviation from LDA  [\%]")
    axes[1].legend(frameon=False, loc="lower left", fontsize=7.4)
    for axis in axes:
        axis.set_xlabel(r"Rod length  $a/a_{\rm ho}$")
        axis.set_xticks(x, ROD_LENGTH_LABELS)
        axis.set_xlim(-0.25, 3.25)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.19, top=0.90, wspace=0.42)
    _save_pair(
        fig,
        output_dir,
        "final_matrix_lda_comparison",
        "DMC and hard-rod LDA energy plus relative energy and RMS-radius differences",
    )


def _density_support_limit(profile: DensityProfile) -> float:
    amplitude = max(float(profile.value.max()), float(profile.lda_value.max()), 1.0)
    support_mask = (
        (profile.value > amplitude * 1e-4)
        | (profile.lda_value > amplitude * 1e-4)
        | (profile.stderr > max(float(profile.stderr.max()) * 1e-4, 1e-12))
    )
    support = float(np.max(np.abs(profile.x[support_mask]))) * 1.05
    if support <= 10.0:
        step = 1.0
    elif support <= 25.0:
        step = 2.0
    elif support <= 60.0:
        step = 5.0
    else:
        step = 10.0
    return step * math.ceil(support / step)


def _plot_density_profiles(densities: dict[str, DensityProfile], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.35, 5.15), sharey=True)
    panel_letters = "abcdefgh"
    for row_index, n_particles in enumerate((10, 20)):
        for column_index, rod_label in enumerate(ROD_LENGTH_LABELS):
            axis = axes[row_index, column_index]
            profile = densities[f"N{n_particles}_A{rod_label}"]
            lower = np.maximum(profile.value - profile.stderr, 0.0)
            upper = profile.value + profile.stderr
            axis.fill_between(
                profile.x,
                lower,
                upper,
                step="mid",
                color=COLORS[10],
                alpha=0.24,
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                profile.lda_x,
                profile.lda_value,
                color="#333333",
                linestyle=(0, (4, 2.3)),
                linewidth=1.2,
                zorder=2,
            )
            axis.stairs(
                profile.value,
                profile.bin_edges,
                color=COLORS[10],
                linewidth=1.05,
                zorder=3,
            )
            if profile.case == "N20_A1":
                shell_average = unit_mass_shell_average(
                    profile.bin_edges,
                    profile.value,
                    particle_count=profile.n_particles,
                    replicate_densities=profile.seed_values,
                )
                _expect(
                    shell_average.stderr is not None,
                    "N20_A1: shell-period standard errors are unavailable",
                )
                axis.errorbar(
                    shell_average.centers,
                    shell_average.values,
                    yerr=shell_average.stderr,
                    color="#009E73",
                    marker="o",
                    markersize=2.8,
                    markeredgewidth=0.5,
                    linewidth=1.45,
                    capsize=1.8,
                    elinewidth=0.8,
                    zorder=4,
                )
            x_limit = _density_support_limit(profile)
            axis.set_xlim(-x_limit, x_limit)
            axis.set_ylim(0.0, 2.3)
            axis.set_xticks((-x_limit, 0.0, x_limit))
            axis.set_yticks((0.0, 0.5, 1.0, 1.5, 2.0))
            axis.grid(axis="y", color="#E0E0E0", linewidth=0.45, alpha=0.75)
            axis.set_axisbelow(True)
            axis.text(
                0.04,
                0.93,
                f"({panel_letters[row_index * 4 + column_index]})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color="#333333",
            )
            if row_index == 0:
                axis.set_title(rf"$a/a_{{\rm ho}}={rod_label}$", pad=7)
            if row_index == 1:
                axis.set_xlabel(r"$x/a_{\rm ho}$")
            if column_index == 0:
                axis.text(
                    0.96,
                    0.93,
                    rf"$N={n_particles}$",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9.0,
                    fontweight="semibold",
                    color="#333333",
                )
    fig.text(0.016, 0.51, r"Density  $n(x)\,a_{\rm ho}$", rotation=90, va="center", ha="center")
    handles = (
        Line2D([], [], color=COLORS[10], linewidth=1.25, label="FW density"),
        Patch(
            facecolor=COLORS[10],
            alpha=0.24,
            edgecolor="none",
            label=r"$1\sigma$ MC uncertainty",
        ),
        Line2D([], [], color="#333333", linewidth=1.2, linestyle=(0, (4, 2.3)), label="LDA"),
        Line2D(
            [],
            [],
            color="#009E73",
            marker="o",
            markersize=3.0,
            linewidth=1.45,
            label="One-shell average, (g)",
        ),
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
        fontsize=7.8,
    )
    fig.subplots_adjust(left=0.082, right=0.992, bottom=0.105, top=0.875, wspace=0.18, hspace=0.22)
    _save_pair(
        fig,
        output_dir,
        "final_matrix_density_profiles",
        "Forward-walking densities, a one-shell-period average, and hard-rod LDA",
    )


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly",
        type=Path,
        default=repo_root
        / "results/dmc/final_matrix/thesis_5seed_all_optimized_final_v1/final_matrix_summary.json",
        help="assembled final-matrix summary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "operator/overleaf/overleaf-shared/figures",
        help="directory receiving PDF and PNG figures",
    )
    return parser.parse_args()


def write_final_report_figures(assembly_path: Path, output_dir: Path) -> tuple[Path, ...]:
    assembly_path, output_dir = assembly_path.resolve(), output_dir.resolve()
    _expect(assembly_path.is_file(), f"assembly summary is missing: {assembly_path}")
    _, rows, densities = _load_sources(assembly_path)
    _configure_matplotlib()
    _plot_lda_comparison(rows, output_dir)
    _plot_density_profiles(densities, output_dir)
    paths: list[Path] = []
    for stem in ("final_matrix_lda_comparison", "final_matrix_density_profiles"):
        for suffix in (".pdf", ".png"):
            path = output_dir / f"{stem}{suffix}"
            _expect(path.is_file() and path.stat().st_size > 1000, f"failed to generate {path}")
            paths.append(path)
    return tuple(paths)


def main() -> None:
    args = _parse_args()
    paths = write_final_report_figures(args.assembly, args.output_dir)
    print(f"Generated {len(paths)} figure files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
