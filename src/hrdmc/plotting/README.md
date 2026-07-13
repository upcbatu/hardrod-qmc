# Plotting

Owner: figure rendering only.

The plotting package consumes assembled workflow payloads and renders comparison
figures. It does not own physics formulas, LDA reducers, Monte Carlo sampling,
estimator assembly, or numerical validation decisions.

Public entry points:

- `write_benchmark_packet_plots(output_dir, payload)`
- `write_benchmark_packet_comparison_plots(output_dir, packets)`
- `write_validation_matrix_plot(output_dir, rows)`

Package layout:

```text
tokens.py              design tokens and status colors
numerics.py            display-safe sigma, y-limits, finite conversions
primitives.py          generic Matplotlib drawing atoms
components/            domain-aware Axes components
figures/               figure composition and saving
reports/               report-level wrappers
```

Expected benchmark-packet inputs:

- `estimates.energy`: mixed local-energy estimate and LDA reference.
- `estimates.r2` / `estimates.rms`: transported forward-walking estimates and
  LDA references.
- `estimates.density`: transported forward-walking density, LDA density, and
  optional mixed-estimator diagnostic density.
- `estimates.pair_distance_density` and `estimates.structure_factor`: optional
  transported forward-walking secondary observables.
- `status` and `stationarity.precision_status`: assembled numerical status used
  only for labels and colors.

Default outputs:

- `scalar_comparison.{png,pdf}`
- `density_comparison.{png,pdf}`
- `numerical_diagnostics.{png,pdf}`
- `energy_stationarity_diagnostics.{png,pdf}`
- `fw_lag_diagnostics.{png,pdf}`
- `benchmark_packet_one_page.{png,pdf}`
- optional `pair_distance_density.{png,pdf}`
- optional `structure_factor.{png,pdf}`

Figure conventions:

- Scalar comparisons are reference-line plots: LDA is the reference line and
  DMC or forward walking is the estimator point with uncertainty.
- Sigma labels pass through `plotting.numerics.display_sigma`; numerical floor
  cases are labelled instead of displaying unstable ratios.
- Diagnostic panels make numerical thresholds visible without reusing the
  observable axes. Effective sample size uses a logarithmic scale.
- Secondary vector observables are rendered as forward-walking curves with
  uncertainty bands and no extra physical interpretation added by plotting.
