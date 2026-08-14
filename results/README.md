# Results

This directory contains a compact record of the DMC evidence reported in the
thesis. It is intended for inspection of the published numbers and figures,
not as a replacement for the full simulation archive.

## Tracked thesis evidence

- `dmc/final_matrix/thesis_5seed_all_optimized_final_v1/` contains the accepted
  eight-case summary, table, and run manifest.
- `dmc/systematics/timestep_extrapolation_v4/` contains the time-step fits and
  their recorded inputs.
- `dmc/systematics/population_systematics_v7_simultaneous/` and
  `dmc/systematics/population_systematics_v8_selected_treatment/` contain the
  walker-population comparisons used in the numerical allowance.
- `dmc/systematics/fw_sensitivity_v1/` contains the forward-walking treatment
  comparisons.
- `figures/` contains the two PDF figures used in the manuscript.

The artifacts remain at their original generated paths so that relative source
locators in the manifests retain their recorded meaning.

## External simulation archive

Per-seed DMC packets, the `N20_A10` radius supplement, arrays, seed traces,
diagnostic plots, and logs remain untracked. They are required for the complete
reassembly procedure in `docs/reproducing.md` and are retained separately in
the thesis evidence archive. The tracked subset exposes the reported values
and derived assessments but cannot reconstruct the matrix from raw samples on
its own.

Other experiment scripts may write generated output under `results/`. Such
output remains ignored unless it is explicitly admitted to this compact
evidence layer.
