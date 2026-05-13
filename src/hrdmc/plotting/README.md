# Plotting

Owner: figure rendering only.

The plotting lane consumes already-assembled workflow payloads and renders
comparison figures. It does not own physics formulas, LDA reducers, DMC
sampling, estimator assembly, or gate decisions.

Current public entry points:

- `write_benchmark_packet_plots(output_dir, payload)`
- `write_claim_matrix_plot(output_dir, rows)`

Package layout:

```text
tokens.py              design tokens and tier colors
numerics.py            display-safe sigma, y-limits, finite conversions
primitives.py          generic Matplotlib drawing atoms
components/           domain-aware Axes components
figures/              figure composition and saving
reports/              report-level wrappers
```

Expected inputs:

- `paper_values.energy`: RN-DMC mixed local-energy result and LDA reference.
- `paper_values.r2` / `paper_values.rms`: transported-FW paper coordinate
  values and LDA references.
- `paper_values.density`: transported-FW density, LDA density, and optional
  mixed diagnostic density.
- `paper_values.pair_distance_density` and `paper_values.structure_factor`:
  optional transported-FW secondary observables without LDA normalization claims.

Default outputs:

- `paper_scalar_comparison.{png,pdf}`
- `density_comparison.{png,pdf}`
- `gate_diagnostics.{png,pdf}`
- `fw_lag_diagnostics.{png,pdf}`
- `benchmark_packet_one_page.{png,pdf}`
- optional `pair_distance_density.{png,pdf}`
- optional `structure_factor.{png,pdf}`

Figure conventions:

- Scalar comparisons are reference-line plots, not bars: LDA is the reference
  line and DMC/FW is the estimator point with uncertainty.
- Sigma labels always pass through `plotting.numerics.display_sigma`; numerical
  floor cases are labelled as such instead of printing meaningless huge sigma
  ratios.
- Gate diagnostics use threshold lollipop panels so the pass/fail boundary is
  visible without reusing the paper-observable axes. Effective sample size uses
  log scale.
- Secondary vector observables are rendered as FW curves with uncertainty bands
  and no hidden LDA normalization claim.
