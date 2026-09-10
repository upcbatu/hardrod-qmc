# Finite-diameter hard-rod density profiles

These are the stored forward-walking densities used in the thesis density
figure, for `N = 10, 20` and `A = a/a_ho = 0.1, 1, 10`. They were exported
from the original result summaries on 10 September 2026 without smoothing,
rebinning, fitting, or replacing the reported values with new simulations.

## Files and units

Each case has two CSV files:

- `N*_A*_density.csv`: bin edges, bin centres, mean DMC density, its statistical
  standard error, and the five individual-seed density estimates.
- `N*_A*_lda.csv`: the LDA coordinate grid and density used for comparison.

`metadata.json` records selected numerical controls, forward times, and
source file hashes. It is a data-comparison description, not a complete
simulation input file. `checksums.json` records hashes of the distributed files.

| Column | Definition |
|---|---|
| `q_left`, `q_right` | Histogram bin edges in oscillator lengths: `q = x/a_ho`. |
| `q_center` | Midpoint of the corresponding bin. |
| `n_fw` | Mean number density `n_ho(q) = a_ho n(a_ho q)` in that bin. |
| `n_fw_sem` | Sample standard deviation of the five seed profiles divided by `sqrt(5)`. |
| `n_fw_seed_7001` through `n_fw_seed_7005` | Density from each independent production seed, after its selected forward-time average. |
| `q`, `n_lda` | Coordinates and number density of the smooth LDA reference. |

The density integrates to **N**, not to one. From inside this data folder:

```python
import numpy as np

d = np.genfromtxt("N10_A1_density.csv", delimiter=",", names=True)
particle_number = np.sum(d["n_fw"] * (d["q_right"] - d["q_left"]))
```

Divide the density by N if the other method reports a single-particle
probability density normalized to one. The units correspond to the Hamiltonian

`H/(hbar*omega) = -1/2 sum_i d^2/dq_i^2 + 1/2 sum_i q_i^2`

on the ordered domain `q_(i+1) - q_i >= A`, with a vanishing wavefunction at
contact. Both calculations must use this length convention and the same A.

## Comparing another method

The DMC values are **bin averages**, not exact point values at the bin centres.
For a smooth prediction, integrate it over each supplied bin and divide by
the bin width before comparing. This matters most for the narrow peaks at
large A. Do not compare a density relative to the instantaneous particle
centre directly with these data: the known harmonic centre-of-mass spreading
has already been included. Reflection symmetry has also been averaged.

The error column is statistical only. Bins are correlated; they must not be
treated as independent observations in a chi-squared calculation. The five
seed columns allow comparisons of integrated quantities without discarding
those correlations. Finite-time-step, population, and forward-time effects
are separate numerical questions. The recorded density forward-time allowance
is included in the metadata; it is not a pointwise confidence band or a bound
on every source of numerical error.

The green cell averages shown for `N20_A1` in the thesis are a derived view of
this same density, not a separate microscopic estimate. The files here retain
the full profile and its shell oscillations. The additional long-forward-time
radius calculation for `N20_A10` does not replace that case's density source.

## Archived data

The CSV files reproduce the stored means, statistical errors, and seed values
exactly when read as float64. Their source hashes match the density figure's
recorded source references. Replotting these data does not require a new
simulation. These files are an export of the thesis results, not the output
of a fresh production run.

## Plotting

The CSVs can be read by any plotting program. In the repository, use the
shared plotting command from the repository root with its environment activated:

```bash
python experiments/run.py plot \
  --input data/density_profiles/N10_A1_density.csv \
  --reference data/density_profiles/N10_A1_lda.csv \
  --output results/plots/N10_A1_density.png
```

The command reads only the specified CSVs; it does not rerun a simulation.
It requires the repository installation, which includes NumPy and Matplotlib.
The [running guide](../../docs/running/plotting.md) gives commands
for all six profiles, new VMC/DMC output and custom CSV columns.
If you received only this data folder, use your own plotting software or obtain
the repository to use that command. There is no executable code in this folder.
The complete original density figure, including the two zero-diameter reference
cases, remains under `results/figures/final_matrix_density_profiles.pdf`.
