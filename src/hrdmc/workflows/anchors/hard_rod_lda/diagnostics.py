from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.io.artifacts import (
    build_run_provenance,
    ensure_dir,
    write_json_atomic,
    write_run_manifest,
)
from hrdmc.theory import (
    hard_rod_lda_density_from_local_mu_cubic,
    hard_rod_lda_density_small_a_expansion,
    lda_density_profile,
    lda_mean_square_radius,
    lda_rms_radius,
    lda_support_edges,
    lda_total_energy,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HardRodLDADiagnosticConfig:
    n_particles: int = 8
    rod_lengths: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.18803, 0.4)
    x_extent: float = 8.0
    n_grid: int = 2401
    representative_rod_length: float = 0.18803
    small_a_rod_length: float = 0.02
    cubic_abs_tolerance: float = 1e-8
    plot_formats: tuple[str, ...] = ("png", "pdf")
    write_plots: bool = True


def run_hard_rod_lda_diagnostic(
    config: HardRodLDADiagnosticConfig,
    output_dir: Path | None = None,
    *,
    command: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build an analytic LDA diagnostic packet for trapped hard rods.

    The packet checks that the production LDA bisection inversion is equivalent
    to the explicit cubic form of the hard-rod LDA equation, and it writes the
    small-rod expansion that explains the TG semicircle plus first correction.
    """

    _validate_config(config)
    x = np.linspace(-config.x_extent, config.x_extent, config.n_grid, dtype=float)
    potential = 0.5 * x * x
    cases = [
        _case_payload(x, potential, config.n_particles, rod_length)
        for rod_length in config.rod_lengths
    ]
    max_cubic_abs_error = max(float(case["cubic_max_abs_error"]) for case in cases)
    status = "passed" if max_cubic_abs_error <= config.cubic_abs_tolerance else "failed"
    payload: dict[str, Any] = {
        "schema_version": "hard_rod_lda_diagnostic_v1",
        "status": status,
        "claim_boundary": (
            "analytic trapped hard-rod LDA diagnostic; no DMC samples and no "
            "finite-N exact benchmark claim"
        ),
        "units": {
            "length": "harmonic oscillator length a_ho",
            "energy": "hbar*Omega",
            "trap_potential": "V(x)=x^2/2",
        },
        "formulae": {
            "local_equation": (
                "mu_loc = pi^2*n^2*(3-A*n)/(6*(1-A*n)^3)"
            ),
            "cubic_variable": "y = A*n/(1-A*n)",
            "cubic_equation": "2*y^3 + 3*y^2 = 6*A^2*mu_loc/pi^2",
            "density_recovery": "n = y/(A*(1+y))",
            "small_A_expansion": (
                "n(mu_loc,A)=sqrt(2*mu_loc)/pi - 8*A*mu_loc/(3*pi^2) + O(A^2)"
            ),
            "tg_semicircle": "n_TG(x)=sqrt(max(2*(mu_0-x^2/2),0))/pi",
        },
        "config": _config_payload(config),
        "max_cubic_abs_error": max_cubic_abs_error,
        "case_table": cases,
    }
    if write:
        if output_dir is None:
            raise ValueError("output_dir is required when write=True")
        artifacts = _write_artifacts(
            output_dir,
            payload,
            x=x,
            potential=potential,
            config=config,
        )
        write_run_manifest(
            output_dir,
            run_name="hard_rod_lda_diagnostic",
            config=_config_payload(config),
            artifacts=artifacts,
            schema_version=str(payload["schema_version"]),
            provenance=build_run_provenance(command),
            status=status,
        )
    return payload


def _validate_config(config: HardRodLDADiagnosticConfig) -> None:
    if config.n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if config.x_extent <= 0:
        raise ValueError("x_extent must be positive")
    if config.n_grid < 11:
        raise ValueError("n_grid must be at least 11")
    if config.n_grid % 2 == 0:
        raise ValueError("n_grid must be odd so x=0 is on the grid")
    if not config.rod_lengths:
        raise ValueError("at least one rod length is required")
    if any(rod_length < 0.0 for rod_length in config.rod_lengths):
        raise ValueError("rod lengths must be non-negative")
    if config.representative_rod_length not in config.rod_lengths:
        raise ValueError("representative_rod_length must be in rod_lengths")
    if config.small_a_rod_length not in config.rod_lengths:
        raise ValueError("small_a_rod_length must be in rod_lengths")
    if config.cubic_abs_tolerance <= 0:
        raise ValueError("cubic_abs_tolerance must be positive")


def _case_payload(
    x: FloatArray,
    potential: FloatArray,
    n_particles: int,
    rod_length: float,
) -> dict[str, Any]:
    profile = lda_density_profile(
        x,
        potential,
        n_particles=float(n_particles),
        rod_length=rod_length,
    )
    local_mu = np.maximum(profile.chemical_potential - potential, 0.0)
    cubic = np.asarray(
        hard_rod_lda_density_from_local_mu_cubic(local_mu, rod_length),
        dtype=float,
    )
    small_a = np.asarray(
        hard_rod_lda_density_small_a_expansion(local_mu, rod_length),
        dtype=float,
    )
    tg = np.sqrt(2.0 * local_mu) / np.pi
    left, right = lda_support_edges(profile)
    energy_repo = lda_total_energy(profile, rod_length)
    return {
        "rod_length_A": float(rod_length),
        "chemical_potential_hbar_omega": float(profile.chemical_potential),
        "integrated_particles": float(profile.integrated_particles),
        "particle_count_error": float(profile.integrated_particles - n_particles),
        "total_energy_hbar_omega": energy_repo,
        "r2": lda_mean_square_radius(profile),
        "rms_radius": lda_rms_radius(profile),
        "support_left": left,
        "support_right": right,
        "peak_density": float(np.max(profile.n_x)),
        "cubic_max_abs_error": _max_abs(profile.n_x, cubic),
        "cubic_relative_l2": _relative_l2(x, profile.n_x, cubic),
        "small_a_max_abs_error": _max_abs(profile.n_x, small_a),
        "small_a_relative_l2": _relative_l2(x, profile.n_x, small_a),
        "semicircle_relative_l2": _relative_l2(x, profile.n_x, tg),
    }


def _write_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
) -> list[Path]:
    root = ensure_dir(output_dir)
    artifacts = [
        write_json_atomic(root / "summary.json", payload),
        _write_case_table(root / "case_table.csv", payload["case_table"]),
        _write_profile_table(root / "profile_table.csv", x, potential, config),
    ]
    if config.write_plots:
        plot_paths = _write_plots(root, x, potential, config)
        payload["plots"] = [path.relative_to(root).as_posix() for path in plot_paths]
        write_json_atomic(root / "summary.json", payload)
        artifacts[0] = root / "summary.json"
        artifacts.extend(plot_paths)
    return artifacts


def _write_case_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = [
        "rod_length_A",
        "chemical_potential_hbar_omega",
        "integrated_particles",
        "particle_count_error",
        "total_energy_hbar_omega",
        "r2",
        "rms_radius",
        "support_left",
        "support_right",
        "peak_density",
        "cubic_max_abs_error",
        "cubic_relative_l2",
        "small_a_relative_l2",
        "semicircle_relative_l2",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _write_profile_table(
    path: Path,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
) -> Path:
    fields = ["x", "potential"]
    columns: dict[str, FloatArray] = {
        "x": x,
        "potential": potential,
    }
    for rod_length in config.rod_lengths:
        profile = lda_density_profile(
            x,
            potential,
            n_particles=float(config.n_particles),
            rod_length=rod_length,
        )
        local_mu = np.maximum(profile.chemical_potential - potential, 0.0)
        label = _rod_label(rod_length)
        fields.extend(
            [
                f"lda_{label}",
                f"cubic_{label}",
                f"small_a_{label}",
                f"semicircle_same_mu_{label}",
            ]
        )
        columns[f"lda_{label}"] = profile.n_x
        columns[f"cubic_{label}"] = np.asarray(
            hard_rod_lda_density_from_local_mu_cubic(local_mu, rod_length),
            dtype=float,
        )
        columns[f"small_a_{label}"] = np.asarray(
            hard_rod_lda_density_small_a_expansion(local_mu, rod_length),
            dtype=float,
        )
        columns[f"semicircle_same_mu_{label}"] = np.sqrt(2.0 * local_mu) / np.pi
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(x.size):
            writer.writerow({field: float(columns[field][index]) for field in fields})
    return path


def _write_plots(
    output_dir: Path,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
) -> list[Path]:
    plt = _load_pyplot(output_dir)
    plot_dir = ensure_dir(output_dir / "plots")
    paths: list[Path] = []
    for extension in config.plot_formats:
        paths.append(
            _plot_profile_family(
                plt,
                x,
                potential,
                config,
                plot_dir / f"lda_profile_family.{extension}",
            )
        )
        paths.append(
            _plot_cubic_equivalence(
                plt,
                x,
                potential,
                config,
                plot_dir / f"lda_cubic_equivalence.{extension}",
            )
        )
        paths.append(
            _plot_small_a_expansion(
                plt,
                x,
                potential,
                config,
                plot_dir / f"lda_small_a_expansion.{extension}",
            )
        )
    return paths


def _load_pyplot(output_dir: Path):
    scratch_dir = ensure_dir(output_dir / "mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(scratch_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_profile_family(
    plt,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for rod_length in config.rod_lengths:
        profile = lda_density_profile(
            x,
            potential,
            n_particles=float(config.n_particles),
            rod_length=rod_length,
        )
        ax.plot(x, profile.n_x, linewidth=1.6, label=f"A={rod_length:g}")
    ax.set_xlabel(r"$x/a_{\rm ho}$")
    ax.set_ylabel(r"$n(x)a_{\rm ho}$")
    ax.set_title(f"Hard-rod LDA profiles, N={config.n_particles}")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.set_xlim(-config.x_extent, config.x_extent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _plot_cubic_equivalence(
    plt,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
    output_path: Path,
) -> Path:
    rod_length = config.representative_rod_length
    profile = lda_density_profile(
        x,
        potential,
        n_particles=float(config.n_particles),
        rod_length=rod_length,
    )
    local_mu = np.maximum(profile.chemical_potential - potential, 0.0)
    cubic = np.asarray(
        hard_rod_lda_density_from_local_mu_cubic(local_mu, rod_length),
        dtype=float,
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(x, profile.n_x, linewidth=1.8, label="bisection profile")
    ax_top.plot(x, cubic, linestyle="--", linewidth=1.3, label="explicit cubic")
    ax_top.set_ylabel(r"$n(x)a_{\rm ho}$")
    ax_top.set_title(f"Equivalent hard-rod LDA inversions, N={config.n_particles}, A={rod_length:g}")
    ax_top.legend(frameon=False, fontsize=8)
    ax_bottom.plot(x, cubic - profile.n_x, linewidth=1.1, color="black")
    ax_bottom.axhline(0.0, color="0.5", linewidth=0.8)
    ax_bottom.set_xlabel(r"$x/a_{\rm ho}$")
    ax_bottom.set_ylabel("diff.")
    ax_bottom.set_xlim(-config.x_extent, config.x_extent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _plot_small_a_expansion(
    plt,
    x: FloatArray,
    potential: FloatArray,
    config: HardRodLDADiagnosticConfig,
    output_path: Path,
) -> Path:
    rod_length = config.small_a_rod_length
    profile = lda_density_profile(
        x,
        potential,
        n_particles=float(config.n_particles),
        rod_length=rod_length,
    )
    local_mu = np.maximum(profile.chemical_potential - potential, 0.0)
    semicircle = np.sqrt(2.0 * local_mu) / np.pi
    expansion = np.asarray(
        hard_rod_lda_density_small_a_expansion(local_mu, rod_length),
        dtype=float,
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(x, profile.n_x, linewidth=1.8, label="exact local inversion")
    ax_top.plot(x, semicircle, linestyle=":", linewidth=1.5, label="TG semicircle at same mu")
    ax_top.plot(x, expansion, linestyle="--", linewidth=1.3, label="small-A expansion")
    ax_top.set_ylabel(r"$n(x)a_{\rm ho}$")
    ax_top.set_title(f"Small-A LDA expansion, N={config.n_particles}, A={rod_length:g}")
    ax_top.legend(frameon=False, fontsize=8)
    ax_bottom.plot(x, expansion - profile.n_x, linewidth=1.1, color="black")
    ax_bottom.axhline(0.0, color="0.5", linewidth=0.8)
    ax_bottom.set_xlabel(r"$x/a_{\rm ho}$")
    ax_bottom.set_ylabel("exp.-exact")
    ax_bottom.set_xlim(-config.x_extent, config.x_extent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _config_payload(config: HardRodLDADiagnosticConfig) -> dict[str, Any]:
    return {
        "n_particles": config.n_particles,
        "rod_lengths_A": list(config.rod_lengths),
        "x_extent": config.x_extent,
        "n_grid": config.n_grid,
        "representative_rod_length_A": config.representative_rod_length,
        "small_a_rod_length_A": config.small_a_rod_length,
        "cubic_abs_tolerance": config.cubic_abs_tolerance,
        "plot_formats": list(config.plot_formats),
        "write_plots": config.write_plots,
    }


def _relative_l2(x: FloatArray, reference: FloatArray, estimate: FloatArray) -> float:
    numerator = np.trapezoid((estimate - reference) ** 2, x)
    denominator = np.trapezoid(reference**2, x)
    if denominator <= 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(np.sqrt(numerator / denominator))


def _max_abs(reference: FloatArray, estimate: FloatArray) -> float:
    return float(np.max(np.abs(estimate - reference)))


def _rod_label(rod_length: float) -> str:
    return f"A{rod_length:.6g}".replace(".", "p").replace("-", "m")
