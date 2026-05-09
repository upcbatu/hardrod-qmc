# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 4
- omega: 0.1
- rod_length: 0.5
- seeds: 5101, 5102, 5103, 5104, 5105, 5106
- steps: 30000
- burn_in: 6000
- thinning: 20
- grid_extent: 40.0
- n_bins: 240

## Diagnostic Checks

- summary status: completed
- completed replicates: 36/36
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.11e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.0112
- max relative-density-L2 seed spread by alpha: 0.0979

## Energy Diagnostic

- lowest sampled total energy alpha: 0.6
- lowest sampled total energy: 1.30408 +/- 0.018
- bracketed energy minimum: yes; the lowest sampled energy is inside the scanned alpha range
- max total-energy stderr by alpha: 0.021
- max total-energy seed spread by alpha: 0.143
- sampled_potential_energy_mean is harmonic trap energy only; use sampled_total_energy_mean for the VMC local-energy diagnostic.

## Replicate Summary

| alpha | total energy | kinetic energy | trap energy | acceptance | relative density L2 | sampled RMS radius | RMS radius error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.52 | 1.32553 +/- 0.014 | 0.594186 +/- 0.021 | 0.731341 +/- 0.026 | 0.940767 +/- 0.0012 | 0.230811 +/- 0.016 | 6.04236 +/- 0.11 | 0.284875 +/- 0.11 |
| 0.55 | 1.3371 +/- 0.014 | 0.603538 +/- 0.009 | 0.733565 +/- 0.021 | 0.939672 +/- 0.0016 | 0.214272 +/- 0.0088 | 6.0532 +/- 0.086 | 0.29571 +/- 0.086 |
| 0.58 | 1.31624 +/- 0.021 | 0.625661 +/- 0.046 | 0.690578 +/- 0.028 | 0.939656 +/- 0.0015 | 0.224765 +/- 0.015 | 5.87014 +/- 0.12 | 0.112656 +/- 0.12 |
| 0.60 | 1.30408 +/- 0.018 | 0.663391 +/- 0.025 | 0.64069 +/- 0.013 | 0.938906 +/- 0.0011 | 0.223008 +/- 0.0085 | 5.65852 +/- 0.056 | -0.0989726 +/- 0.056 |
| 0.62 | 1.32492 +/- 0.011 | 0.688189 +/- 0.018 | 0.636728 +/- 0.0093 | 0.936817 +/- 0.0019 | 0.218534 +/- 0.0094 | 5.64161 +/- 0.041 | -0.115874 +/- 0.041 |
| 0.65 | 1.33888 +/- 0.019 | 0.761224 +/- 0.022 | 0.577658 +/- 0.0083 | 0.935417 +/- 0.0011 | 0.23509 +/- 0.0073 | 5.37358 +/- 0.039 | -0.383909 +/- 0.039 |

## Interpretation

This is a VMC diagnostic alpha scan.
The total-energy curve is a diagnostic variational surface for the current trial, not a production benchmark.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum unless the minimum is bracketed and uncertainties support it.
It does not validate LDA accuracy or DMC readiness.
