# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 4
- omega: 0.1
- rod_length: 0.5
- seeds: 5101, 5102, 5103, 5104, 5105, 5106
- steps: 20000
- burn_in: 4000
- thinning: 20
- grid_extent: 40.0
- n_bins: 240

## Diagnostic Checks

- summary status: completed
- completed replicates: 60/60
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.15e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.0141
- max relative-density-L2 seed spread by alpha: 0.116

## Energy Diagnostic

- lowest sampled total energy alpha: 0.6
- lowest sampled total energy: 1.28665 +/- 0.014
- bracketed energy minimum: yes; the lowest sampled energy is inside the scanned alpha range
- max total-energy stderr by alpha: 0.0369
- max total-energy seed spread by alpha: 0.249
- sampled_potential_energy_mean is harmonic trap energy only; use sampled_total_energy_mean for the VMC local-energy diagnostic.

## Replicate Summary

| alpha | total energy | kinetic energy | trap energy | acceptance | relative density L2 | sampled RMS radius | RMS radius error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 1.41282 +/- 0.037 | 0.391657 +/- 0.029 | 1.02117 +/- 0.065 | 0.951992 +/- 0.0014 | 0.344166 +/- 0.016 | 7.12621 +/- 0.23 | 1.36872 +/- 0.23 |
| 0.40 | 1.37956 +/- 0.017 | 0.44921 +/- 0.02 | 0.930348 +/- 0.029 | 0.94865 +/- 0.0011 | 0.293881 +/- 0.012 | 6.8163 +/- 0.11 | 1.05881 +/- 0.11 |
| 0.45 | 1.35429 +/- 0.024 | 0.490929 +/- 0.023 | 0.863364 +/- 0.045 | 0.94645 +/- 0.0013 | 0.25969 +/- 0.018 | 6.55924 +/- 0.17 | 0.801753 +/- 0.17 |
| 0.50 | 1.32816 +/- 0.014 | 0.594491 +/- 0.042 | 0.73367 +/- 0.034 | 0.942583 +/- 0.0013 | 0.27088 +/- 0.012 | 6.04857 +/- 0.14 | 0.291086 +/- 0.14 |
| 0.55 | 1.30738 +/- 0.019 | 0.60397 +/- 0.02 | 0.703414 +/- 0.025 | 0.939617 +/- 0.0015 | 0.245783 +/- 0.0081 | 5.92577 +/- 0.11 | 0.168278 +/- 0.11 |
| 0.60 | 1.28665 +/- 0.014 | 0.652648 +/- 0.043 | 0.634006 +/- 0.033 | 0.938508 +/- 0.0013 | 0.247817 +/- 0.019 | 5.62049 +/- 0.15 | -0.136995 +/- 0.15 |
| 0.65 | 1.33718 +/- 0.026 | 0.781234 +/- 0.033 | 0.555944 +/- 0.014 | 0.935208 +/- 0.0013 | 0.255094 +/- 0.015 | 5.27016 +/- 0.067 | -0.487332 +/- 0.067 |
| 0.70 | 1.36857 +/- 0.026 | 0.828386 +/- 0.045 | 0.540187 +/- 0.021 | 0.932533 +/- 0.0022 | 0.283089 +/- 0.013 | 5.1924 +/- 0.098 | -0.565089 +/- 0.098 |
| 0.75 | 1.37693 +/- 0.022 | 0.869001 +/- 0.028 | 0.507925 +/- 0.0077 | 0.929117 +/- 0.0014 | 0.312889 +/- 0.013 | 5.03875 +/- 0.038 | -0.718742 +/- 0.038 |
| 0.80 | 1.41527 +/- 0.028 | 0.939326 +/- 0.034 | 0.47594 +/- 0.0085 | 0.929733 +/- 0.0017 | 0.34344 +/- 0.012 | 4.87722 +/- 0.044 | -0.880263 +/- 0.044 |

## Interpretation

This is a VMC diagnostic alpha scan.
The total-energy curve is a diagnostic variational surface for the current trial, not a production benchmark.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum unless the minimum is bracketed and uncertainties support it.
It does not validate LDA accuracy or DMC readiness.
