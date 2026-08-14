# Numerical inputs

This directory contains the small numerical inputs required by a clean clone.
Large stochastic run bundles belong outside the tracked repository; their
expected locations and use are documented in
[`docs/reproducing.md`](../docs/reproducing.md).

## Final-matrix guide registry

`final_matrix_guides/summary.json` records the validated reduced-TG width used
for each finite-diameter production case:

```text
N10_A0.1  1.0637325870622627
N10_A1    1.6224444406063525
N10_A10   5.5908651157560385
N20_A0.1  1.0908094794241916
N20_A1    1.8669363227063642
N20_A10   7.011111084682286
```

The two zero-diameter cases use the analytic reduced-TG default and therefore
need no fitted width record. `final_matrix_guides/run_manifest.json` binds the
registry to its byte size and SHA-256 digest. The final-matrix runner reads this
tracked registry by default, so an eight-case dry run does not depend on an
untracked calibration directory.

## Hellmann--Feynman energy response

`energy_response/N20_A10_h0025_5seed.csv` contains five paired seeds at the
relative trap-coupling offsets

```text
-0.005, -0.0025, 0, 0.0025, 0.005.
```

The symmetric five-point response and Richardson extrapolation give the pure
cloud radius reported for `N20_A10`. Recompute it from the repository root with

```bash
PYTHONPATH=src python3 experiments/dmc/local/reanalyze_energy_response.py
```

The expected result is
`rms_radius = 58.85127194991539` with seed standard error
`0.0007373005273995926` in harmonic-oscillator units.

## Data policy

Tracked files here must be compact, immutable inputs needed for a documented
reproduction path. Raw walker histories, checkpoints, per-seed production
packets, generated plots, and temporary analysis output are not stored here.
