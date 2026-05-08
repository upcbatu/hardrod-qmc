# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 8
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
- completed replicates: 10/10
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 5.95e-14
- max |LDA integrated particles error|: 6.83e-11
- max acceptance seed spread by alpha: 0.0328
- max relative-density-L2 seed spread by alpha: 0.0617

## Replicate Summary

| alpha | acceptance | relative density L2 | sampled RMS radius | RMS radius error | sampled potential energy |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.875833 +/- 0.013 | 0.547497 +/- 0.007 | 6.27179 +/- 0.25 | -2.12881 +/- 0.25 | 1.57592 +/- 0.13 |
| 0.75 | 0.852417 +/- 0.016 | 0.669258 +/- 0.015 | 5.42558 +/- 0.078 | -2.97502 +/- 0.078 | 1.17772 +/- 0.034 |
| 1.00 | 0.828917 +/- 0.0016 | 0.813922 +/- 0.024 | 4.67595 +/- 0.1 | -3.72465 +/- 0.1 | 0.875008 +/- 0.039 |
| 1.25 | 0.812583 +/- 0.0024 | 0.890667 +/- 0.00067 | 4.33514 +/- 0.024 | -4.06547 +/- 0.024 | 0.75176 +/- 0.0084 |
| 1.50 | 0.799083 +/- 0.012 | 0.937683 +/- 0.031 | 4.14314 +/- 0.079 | -4.25747 +/- 0.079 | 0.686873 +/- 0.026 |

## Interpretation

This is a VMC diagnostic alpha scan.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum because trapped local energy is not implemented.
It does not validate LDA accuracy or DMC readiness.
