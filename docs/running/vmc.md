# VMC

Run from the repository root with the [environment activated](../../README.md#installation).
VMC samples `Psi_T²` and averages the local energy: [§4.2, Eq. (4.7)](../Batuhan_Turgay_TFM.pdf#page=17).
`--preset thesis` loads the trial wavefunction used in the thesis. The run
lengths and checks are set separately.

## Run with MALA

```bash
python experiments/run.py vmc \
  --n 10 \
  --a 1 \
  --preset thesis \
  --sampler mala \
  --mala-step 0.04 \
  --drift-limiter umrigar \
  --walkers 256 \
  --burn-in-steps 20000 \
  --production-steps 20000 \
  --block-size 20 \
  --grid-extent 35 \
  --bins 840 \
  --initial-scale 1 \
  --cutoff-epsilons 0.01,0.02,0.04 \
  --rhat-limit 1.01 \
  --min-effective-samples 400 \
  --seeds 7001,7002,7003,7004,7005 \
  --workers 5 \
  --output results/vmc_mala
```

Results: `summary.json`; density: `density.csv` ([plotting](plotting.md)).
Check whether the convergence tests passed before using the estimates; see
[Checks and results](checks-and-results.md#energy-and-sampling-checks).

## Run with RWM

```bash
python experiments/run.py vmc \
  --n 10 \
  --a 1 \
  --preset thesis \
  --sampler rwm \
  --rwm-step 0.8 \
  --drift-limiter none \
  --walkers 256 \
  --burn-in-steps 20000 \
  --production-steps 20000 \
  --block-size 20 \
  --seeds 7001,7002,7003,7004,7005 \
  --workers 5 \
  --output results/vmc_rwm
```

MALA moves all coordinates using Gaussian noise and guide drift; RWM moves
one coordinate by a uniform displacement. Both reject rod overlaps and use
a Metropolis acceptance test. MALA includes the forward/reverse proposal ratio.
See Eqs. (4.8)–(4.9), references [12–13] Metropolis/Hastings; the optional
[Umrigar limiter](dmc-and-forward-walking.md#what-umrigar-and-fixed-population-mean)
is Eq. (4.18), reference [14]. Neither VMC sampler uses DMC weights or resampling.

## Settings

| Option | Meaning |
|---|---|
| `--walkers` | Independent walker chains per seed. |
| `--burn-in-steps`, `--production-steps` | Move attempts discarded for equilibration and used for measurement, per walker. |
| `--mala-step` | Step size for the Gaussian noise and drift move. |
| `--rwm-step` | Maximum absolute displacement of one coordinate, in oscillator lengths. |
| `--drift-limiter` | `none` or `umrigar`; RWM requires `none`. |
| `--block-size` | Successive observations per stored block; production steps must be divisible by it. Blocks can remain correlated. |
| `--grid-extent`, `--bins` | Histogram range `[-extent,+extent]` and bin count. |
| `--initial-scale` | Scale free coordinates while preserving rod lengths; 0.5/2 give compact/expanded starts. |
| `--cutoff-epsilons` | Near-contact cutoffs for the gradient kinetic estimator. |
| `--rhat-limit` | Limit on ordinary split R-hat of the seed block traces. |
| `--min-effective-samples` | Minimum sample count after accounting for time correlations in each seed trace; different from walker-weight ESS. |
| `--seeds`, `--workers` | Distinct nonnegative seeds and parallel seed processes. |
| `--alpha`, `--alpha-from` | Override the preset guide; see [alpha](guide-and-alpha.md). |

These examples store 1,000 blocks per seed (`20000/20`). The ESS can be
smaller because of correlations; an ESS threshold of 400 still needs checking.
Use `python experiments/run.py vmc --help` for defaults.
`--dry-run --verbose-json` prints the settings that will be used. Use a new or empty
output directory for each run; interrupted seeds are not resumed.

## Complete thesis VMC validation

Use the [recorded validation commands](../reproducing.md#re-running-the-vmc-validation)
to repeat the VMC checks reported in the thesis. Those calculations used
different run lengths and four kinetic cutoffs: [§4.2.2](../Batuhan_Turgay_TFM.pdf#page=18).
They also used rank-normalized/folded R-hat and bulk ESS. The general VMC
command uses ordinary split R-hat and per-seed ESS: [Appendix A.2](../Batuhan_Turgay_TFM.pdf#page=31), reference [18].

The local and gradient kinetic estimators are Eqs. (4.10)–(4.11). Comparing them requires extrapolating the gradient estimate to zero cutoff
and checking how the fit affects the answer. A value at a finite cutoff
is not the full kinetic energy: Eqs. (4.12)–(4.14).
