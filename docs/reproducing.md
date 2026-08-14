# Reproducing the thesis artifacts

These commands were run from the repository root on 14 August 2026. They do
not launch new production simulations. They reassemble the recorded seed
packets, recompute the numerical summaries, render the two figures, and
reanalyse the tracked Hellmann--Feynman response data. All output goes to a
new temporary directory.

## Prerequisites

The simulation bundles under `results/` are not tracked. Before running the
command, copy these directories from the thesis evidence archive to the paths
shown below:

- `results/dmc/final_matrix/thesis_5seed_all_optimized_v1/`
- `results/dmc/final_matrix/thesis_5seed_all_optimized_v1_supplement/N20_A10_r2_tau15/`
- `results/dmc/systematics/timestep_extrapolation_v4/`
- `results/dmc/systematics/population_systematics_v7_simultaneous/`
- `results/dmc/systematics/population_systematics_v8_selected_treatment/`
- `results/dmc/systematics/fw_sensitivity_v1/`

The Hellmann--Feynman input
`data/energy_response/N20_A10_h0025_5seed.csv` is tracked and needs no external
bundle.

## Command

```bash
bash <<'BASH'
set -euo pipefail

REPRO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/hardrod-qmc-reproduction.XXXXXX")"
export REPRO_ROOT
printf 'REPRO_ROOT=%s\n' "$REPRO_ROOT"

PYTHONPATH=src python3 experiments/dmc/local/assemble_final_matrix.py \
  --source-root results/dmc/final_matrix/thesis_5seed_all_optimized_v1 \
  --output-root "$REPRO_ROOT/final_matrix" \
  --r2-supplement \
    N20_A10=results/dmc/final_matrix/thesis_5seed_all_optimized_v1_supplement/N20_A10_r2_tau15 \
  --retrospective-energy-cases N10_A0.1

cases=(N10_A0.1 N10_A1 N10_A10 N20_A0.1 N20_A1 N20_A10)
systematics_args=()
for case in "${cases[@]}"; do
  systematics_args+=(
    --timestep "$case=results/dmc/systematics/timestep_extrapolation_v4/$case/run_manifest.json"
    --fw-sensitivity "$case=results/dmc/systematics/fw_sensitivity_v1/$case/run_manifest.json"
  )
  if [[ "$case" == N10_A10 ]]; then
    population_root=results/dmc/systematics/population_systematics_v8_selected_treatment
  else
    population_root=results/dmc/systematics/population_systematics_v7_simultaneous
  fi
  systematics_args+=(--population "$case=$population_root/$case/run_manifest.json")
done
qualifier='The declared leading-linear and leading-quadratic zero-step estimates are retained with separate model-order and fit-window allowances below 0.01 hbar*Omega.'
PYTHONPATH=src python3 experiments/dmc/local/assemble_numerical_systematics.py \
  --final-matrix-manifest "$REPRO_ROOT/final_matrix/run_manifest.json" \
  "${systematics_args[@]}" \
  --bounded-qualifier "N10_A10:timestep=$qualifier" \
  --bounded-qualifier "N20_A0.1:timestep=$qualifier" \
  --bounded-qualifier "N20_A1:timestep=$qualifier" \
  --output-dir "$REPRO_ROOT/numerical_systematics"

python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["REPRO_ROOT"])
cases = ("N10_A0.1", "N10_A1", "N10_A10", "N20_A0.1", "N20_A1", "N20_A10")
with (root / "final_matrix" / "final_matrix_table.csv").open(newline="") as handle:
    matrix = {row["case"]: row for row in csv.DictReader(handle)}

with (root / "energy_fit_inputs.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("case", "walkers", "dt_values"))
    writer.writeheader()
    for case in cases:
        point_table = Path("results/dmc/systematics/timestep_extrapolation_v4") / case / "point_table.csv"
        with point_table.open(newline="") as points:
            dt_values = [row["dt"] for row in csv.DictReader(points)]
        writer.writerow({"case": case, "walkers": matrix[case]["walkers"], "dt_values": ",".join(dt_values)})

with (root / "proposal_efficiency.csv").open("w", newline="") as handle:
    fields = (
        "case", "anchor_acceptance", "candidate_dt", "candidate_walkers",
        "candidate_acceptance", "configuration_esjd_ratio",
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for case in cases:
        summary_path = Path("results/dmc/systematics/fw_sensitivity_v1") / case / "summary.json"
        summary = json.loads(summary_path.read_text())
        anchor = summary["treatments"]["anchor_density"]
        candidate = summary["treatments"]["candidate"]
        anchor_telemetry = anchor["proposal_telemetry"]
        candidate_telemetry = candidate["proposal_telemetry"]
        writer.writerow({
            "case": case,
            "anchor_acceptance": anchor_telemetry["local_acceptance_fraction_mean"],
            "candidate_dt": candidate["dt"],
            "candidate_walkers": candidate["walkers"],
            "candidate_acceptance": candidate_telemetry["local_acceptance_fraction_mean"],
            "configuration_esjd_ratio": (
                candidate_telemetry["configuration_esjd_mean"]
                / anchor_telemetry["configuration_esjd_mean"]
            ),
        })

print(f"Generated {root / 'energy_fit_inputs.csv'}")
print(f"Generated {root / 'proposal_efficiency.csv'}")
PY

MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONPATH=src \
  python3 experiments/report/final_figures.py \
  --assembly "$REPRO_ROOT/final_matrix/final_matrix_summary.json" \
  --output-dir "$REPRO_ROOT/figures"

PYTHONPATH=src python3 experiments/dmc/local/reanalyze_energy_response.py \
  | tee "$REPRO_ROOT/hf_radius.json"
BASH
```

The matrix assembly reports `accepted_count: 8`. The numerical-systematics
assembly reports seven publication-ready cases, with only `N20_A10`
unresolved. That status applies to its zero-time-step energy: the manuscript
retains the raw finite-step `N20_A10` energy but does not claim a qualified
zero-time-step value for that case. The figure command writes PDF and PNG
versions of both figures.

## Manuscript artifact map

The table maps each manuscript label or reported value to an output of the
command. “Matrix bundle” and “systematics bundles” refer to the untracked
prerequisites listed above.

| Manuscript label or value | Generated source | Required untracked input |
|---|---|---|
| `tab:settings` | `$REPRO_ROOT/final_matrix/final_matrix_table.csv`: guide, time step, walkers, storage cadence, and grid columns | Matrix bundle and R2 supplement |
| `tab:scalar-results` | `$REPRO_ROOT/final_matrix/final_matrix_table.csv`: DMC/LDA energy and FW/LDA radius columns | Matrix bundle and R2 supplement |
| `tab:numerical-summary` | `$REPRO_ROOT/numerical_systematics/thesis_energy_table.csv`, `uncertainty_components.csv`, and `case_status.csv` | Matrix and all systematics bundles |
| `tab:energy-components` | `$REPRO_ROOT/numerical_systematics/summary.json`, `thesis_energy_table.csv`, and `uncertainty_components.csv` | Matrix and all systematics bundles |
| `tab:energy-fit-inputs` | `$REPRO_ROOT/energy_fit_inputs.csv` | Matrix bundle and time-step point tables |
| `tab:fw-lags` | `$REPRO_ROOT/final_matrix/final_matrix_table.csv`: density and R2 window and late-lag-bound columns | Matrix bundle and R2 supplement |
| `tab:density-distances` | `$REPRO_ROOT/final_matrix/final_matrix_table.csv`: FW--LDA and FW--mixed relative L2 columns | Matrix bundle and R2 supplement |
| `tab:proposal-efficiency` | `$REPRO_ROOT/proposal_efficiency.csv` | Forward-walking sensitivity summaries |
| `fig:density-profiles` | `$REPRO_ROOT/figures/final_matrix_density_profiles.pdf` | Matrix bundle and R2 supplement |
| `fig:lda-deviations` | `$REPRO_ROOT/figures/final_matrix_lda_comparison.pdf` | Matrix bundle and R2 supplement |
| Hellmann--Feynman radius `58.85127(74)` | `$REPRO_ROOT/hf_radius.json`: `rms_radius = 58.85127194991539`, `rms_radius_seed_stderr = 0.0007373005273995926` | None; input CSV is tracked |

The two compact extraction tables introduce no new estimates. They select the
time-step values and proposal telemetry already bound to the manifest-verified
systematics inputs.
