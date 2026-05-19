# RN-DMC Tables

This directory contains compact summaries for the RN-corrected
collective-block DMC candidate.

These files are compact summaries rather than raw run bundles. Full trace JSON,
per-seed audits, progress logs, and large reproducibility payloads belong in
separate run artifacts for thesis or paper use.

The trapped rows in the current CSV snapshots predate the harmonic-oscillator
case-id migration and use legacy `N*_a*_omega*` labels. New trapped production
runs use canonical `N*_A*` case ids, where \(A=a/a_{\mathrm{ho}}\).

## Files

- `rn_validation_summary.csv`: homogeneous hard-rod validation against exact
  finite-ring references.
- `rn_grid_summary.csv`: compact trapped DMC-versus-LDA candidate table,
  including six-case and frontier rows.
- `rn_seed_robustness_summary.csv`: seed-stress summaries for the weak `N = 4`
  trap cases.

## Claim Boundary

The tables support this label:

```text
RN-corrected collective-block DMC engine, validated candidate.
```

Final paper artifact bundles additionally need versioned code, raw
reproducibility artifacts, citation metadata, and an archived dataset.
