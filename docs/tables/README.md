# RN-DMC Tables

This directory contains compact summaries for the RN-corrected
collective-block DMC candidate.

These files are intentionally not raw run bundles. Full trace JSON, per-seed
audits, progress logs, and large reproducibility payloads should be archived as
separate run artifacts for thesis or paper use.

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

They do not by themselves constitute a final paper artifact bundle. A release
bundle still needs versioned code, raw reproducibility artifacts, citation
metadata, and an archived dataset.
