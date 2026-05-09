# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 4
- omega: 0.1
- rod_length: 0.5
- seeds: 5101, 5102
- steps: 6000
- burn_in: 1000
- thinning: 20
- grid_extent: 40.0
- n_bins: 240

## Diagnostic Checks

- summary status: completed
- completed replicates: 8/8
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.11e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.014
- max relative-density-L2 seed spread by alpha: 0.213

## Replicate Summary

| alpha | total energy | kinetic energy | trap energy | acceptance | relative density L2 | sampled RMS radius | RMS radius error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1.28019 +/- 0.032 | 0.495681 +/- 0.099 | 0.784513 +/- 0.067 | 0.943833 +/- 0.007 | 0.447871 +/- 0.079 | 6.25732 +/- 0.27 | 0.49983 +/- 0.27 |
| 0.50 | 1.32855 +/- 0.052 | 0.784602 +/- 0.069 | 0.543945 +/- 0.016 | 0.93925 +/- 0.0024 | 0.472967 +/- 0.017 | 5.21452 +/- 0.078 | -0.542966 +/- 0.078 |
| 0.75 | 1.52213 +/- 0.086 | 1.11055 +/- 0.11 | 0.411577 +/- 0.026 | 0.921583 +/- 0.0011 | 0.488523 +/- 0.012 | 4.53414 +/- 0.14 | -1.22335 +/- 0.14 |
| 1.00 | 1.93389 +/- 0.089 | 1.62936 +/- 0.13 | 0.304537 +/- 0.043 | 0.904833 +/- 0.00083 | 0.648061 +/- 0.11 | 3.89245 +/- 0.28 | -1.86504 +/- 0.28 |

## Interpretation

This is a VMC diagnostic alpha scan.
The total-energy curve is a diagnostic variational surface for the current trial, not a production benchmark.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum unless the minimum is bracketed and uncertainties support it.
It does not validate LDA accuracy or DMC readiness.
