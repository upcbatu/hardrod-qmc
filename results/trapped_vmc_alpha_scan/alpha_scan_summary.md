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
- completed replicates: 10/10
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.11e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.0128
- max relative-density-L2 seed spread by alpha: 0.213

## Replicate Summary

| alpha | acceptance | relative density L2 | sampled RMS radius | RMS radius error | sampled potential energy |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.93925 +/- 0.0024 | 0.472967 +/- 0.017 | 5.21452 +/- 0.078 | -0.542966 +/- 0.078 | 0.543945 +/- 0.016 |
| 0.75 | 0.921583 +/- 0.0011 | 0.488523 +/- 0.012 | 4.53414 +/- 0.14 | -1.22335 +/- 0.14 | 0.411577 +/- 0.026 |
| 1.00 | 0.904833 +/- 0.00083 | 0.648061 +/- 0.11 | 3.89245 +/- 0.28 | -1.86504 +/- 0.28 | 0.304537 +/- 0.043 |
| 1.25 | 0.90325 +/- 0.0018 | 0.643915 +/- 0.023 | 3.86288 +/- 0.022 | -1.89461 +/- 0.022 | 0.298446 +/- 0.0034 |
| 1.50 | 0.89825 +/- 0.0064 | 0.703538 +/- 0.0052 | 3.50034 +/- 0.062 | -2.25715 +/- 0.062 | 0.245125 +/- 0.0087 |

## Interpretation

This is a VMC diagnostic alpha scan.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum because trapped local energy is not implemented.
It does not validate LDA accuracy or DMC readiness.
