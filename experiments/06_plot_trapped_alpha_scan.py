from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PLOT_SPECS = (
    (
        "relative_density_l2_error_vmc_vs_lda",
        "Relative density L2",
        "alpha_vs_relative_density_l2.png",
    ),
    (
        "rms_radius_error_vmc_vs_lda",
        "RMS radius error (VMC - LDA)",
        "alpha_vs_rms_radius_error.png",
    ),
    ("acceptance_rate", "Acceptance rate", "alpha_vs_acceptance.png"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot trapped VMC alpha-scan diagnostics.")
    default_results = REPO_ROOT / "results" / "trapped_vmc_alpha_scan"
    parser.add_argument("--summary", type=Path, default=default_results / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=default_results)
    return parser


def metric(scan_case: dict, name: str, field: str) -> float:
    return float(scan_case["metric_summary"][name][field])


def fmt_mean_stderr(scan_case: dict, name: str) -> str:
    mean = metric(scan_case, name, "mean")
    stderr = metric(scan_case, name, "stderr")
    return f"{mean:.6g} +/- {stderr:.2g}"


def max_abs_replicate(summary: dict, name: str) -> float:
    return max(abs(float(replicate[name])) for replicate in summary["replicates"])


def min_replicate(summary: dict, name: str) -> float:
    return min(float(replicate[name]) for replicate in summary["replicates"])


def max_metric_spread(scan: list[dict], name: str) -> float:
    return max(metric(case, name, "spread") for case in scan)


def load_pyplot():
    import os

    scratch_dir = REPO_ROOT / "results" / "trapped_vmc_alpha_scan" / ".tmp"
    mpl_config_dir = scratch_dir / "matplotlib"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch_dir)
    os.environ["TEMP"] = str(scratch_dir)
    os.environ["TMP"] = str(scratch_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric(
    plt,
    scan: list[dict],
    metric_name: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = [float(case["alpha_multiplier"]) for case in scan]
    y = [metric(case, metric_name, "mean") for case in scan]
    yerr = [metric(case, metric_name, "stderr") for case in scan]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=4)
    ax.set_xlabel("alpha multiplier")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_summary(summary: dict, output_path: Path) -> None:
    controls = summary["controls"]
    scan = summary["scan"]
    completed_count = sum(
        1 for replicate in summary["replicates"] if replicate["status"] == "completed"
    )
    replicate_count = len(summary["replicates"])

    lines = [
        "# Trapped VMC Alpha-Scan Diagnostic",
        "",
        "## Run Controls",
        "",
        f"- N: {controls['n_particles']}",
        f"- omega: {controls['omega']}",
        f"- rod_length: {controls['rod_length']}",
        f"- seeds: {', '.join(str(seed) for seed in controls['seeds'])}",
        f"- steps: {controls['steps']}",
        f"- burn_in: {controls['burn_in']}",
        f"- thinning: {controls['thinning']}",
        f"- grid_extent: {controls['grid_extent']}",
        f"- n_bins: {controls['n_bins']}",
        "",
        "## Diagnostic Checks",
        "",
        f"- summary status: {summary['status']}",
        f"- completed replicates: {completed_count}/{replicate_count}",
        "- minimum valid snapshot fraction: "
        f"{min_replicate(summary, 'valid_snapshot_fraction'):.6g}",
        "- max |sampled density integral error|: "
        f"{max_abs_replicate(summary, 'sampled_density_integral_error'):.3g}",
        "- max |LDA integrated particles error|: "
        f"{max_abs_replicate(summary, 'lda_integrated_particles_error'):.3g}",
        "- max acceptance seed spread by alpha: "
        f"{max_metric_spread(scan, 'acceptance_rate'):.3g}",
        "- max relative-density-L2 seed spread by alpha: "
        f"{max_metric_spread(scan, 'relative_density_l2_error_vmc_vs_lda'):.3g}",
        "",
        "## Replicate Summary",
        "",
        "| alpha | acceptance | relative density L2 | sampled RMS radius | "
        "RMS radius error | sampled potential energy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for case in scan:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{float(case['alpha_multiplier']):.2f}",
                    fmt_mean_stderr(case, "acceptance_rate"),
                    fmt_mean_stderr(case, "relative_density_l2_error_vmc_vs_lda"),
                    fmt_mean_stderr(case, "sampled_rms_radius"),
                    fmt_mean_stderr(case, "rms_radius_error_vmc_vs_lda"),
                    fmt_mean_stderr(case, "sampled_potential_energy_mean"),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a VMC diagnostic alpha scan.",
            "The scan can guide density/radius diagnostics.",
            "It does not select a production variational optimum because trapped local "
            "energy is not implemented.",
            "It does not validate LDA accuracy or DMC readiness.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    with args.summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = sorted(summary["scan"], key=lambda case: float(case["alpha_multiplier"]))
    plt = load_pyplot()

    for metric_name, ylabel, filename in PLOT_SPECS:
        plot_metric(plt, scan, metric_name, ylabel, output_dir / filename)
    write_markdown_summary(summary, output_dir / "alpha_scan_summary.md")


if __name__ == "__main__":
    main()
