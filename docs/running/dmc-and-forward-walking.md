# DMC and forward walking

Run from the repository root with the [environment activated](../../README.md#installation).
DMC uses the mixed estimator for energy and forward walking for density and radius:
[§4.3–4.4, Eqs. (4.15)–(4.21)](../Batuhan_Turgay_TFM.pdf#page=18).

## Run a thesis case

```bash
python experiments/run.py dmc \
  --preset thesis \
  --case N10_A0.1 \
  --density-fw-times 0,2,4,7 \
  --rms-fw-times 0,5,10,20,30,40,50 \
  --seeds 7001,7002,7003,7004,7005 \
  --workers 5 \
  --output results/dmc_N10_A01 \
  --progress
```

Results: `summary.json`; density: `density.csv` when available.
Before using them, check whether the energy and FW tests passed:
[Checks and results](checks-and-results.md).

`--preset thesis` loads the guide and numerical settings for N=10,20 and
A=0,0.1,1,10. Options supplied in the command replace those saved settings. `--dry-run --verbose-json` shows
all settings without running. Use a new or empty output directory; interrupted
seeds are not resumed.

Units: `A=a/a_ho`, `q=x/a_ho`, energy in `hbar*Omega`, time in `1/Omega`
([§3.2](../Batuhan_Turgay_TFM.pdf#page=12)). Forward time divided by dt gives
the lag in steps: at dt=0.0025, times **0,2,4,7 → 0,800,1600,2800 steps**.
At zero lag, transport should leave the measurement unchanged. Positive lags
are used to compare estimates at different forward times.

## Change a setting

```bash
python experiments/run.py dmc \
  --preset thesis \
  --case N10_A0.1 \
  --dt 0.00125 \
  --walkers 512 \
  --burn-time 60 \
  --run-time 480 \
  --density-fw-times 0,2,4,7 \
  --workers 5 \
  --output results/dmc_finer
```

Changing dt keeps the same physical times for recording energy, starting FW
measurements and following descendants. Each must correspond to an integer number of steps. Use physical-time
options rather than mixing them with old step-count options.

## A new system

Choose or [optimize alpha](guide-and-alpha.md). The settings below are an
example for N8_A0.2; they have not been validated for that system:

```bash
python experiments/run.py dmc \
  --n 8 \
  --a 0.2 \
  --alpha 1.1 \
  --dt 0.0025 \
  --walkers 256 \
  --drift-limiter umrigar \
  --burn-time 60 \
  --run-time 480 \
  --store-time 0.025 \
  --rms-fw-times 0,5,10,20,30,40,50 \
  --density-fw-times 0,2,4,7 \
  --rms-collection-time 0.05 \
  --density-collection-time 0.1 \
  --pure-fw-observable-source r2_rb \
  --pure-fw-density-source com_rao_blackwell \
  --pure-fw-density-parity-average \
  --pure-fw-min-block-count 20 \
  --pure-fw-rms-plateau-relative-tolerance 0.001 \
  --pure-fw-density-plateau-window-lag-count 3 \
  --grid-extent 35 \
  --n-bins 840 \
  --seeds 7001,7002,7003,7004,7005 \
  --workers 5 \
  --output results/dmc_N8_A02
```

Replace `--alpha` with `--alpha-from .../summary.json` to use a candidate.

## Sampling and measurement options

| Option | What it controls |
|---|---|
| `--dt` | DMC time step Delta-tau*Omega. Smaller steps need more iterations for the same duration. |
| `--walkers` | Number of walkers M, kept fixed during the run. |
| `--burn-time`, `--run-time` | Equilibration and measurement duration in 1/Omega. |
| `--seeds`, `--workers` | Independent runs and the number run in parallel; 0 workers selects automatically. |
| `--drift-limiter` | `none` or the `umrigar` drift in Eq. (4.18). |
| `--ess-resample-fraction` | Resample when walker-weight ESS falls below this fraction of M; default 0.35. |
| `--initialization-mode` | `tight-lattice`, `lda-rms-lattice`, or `lda-rms-logspread`. |
| `--init-width-log-sigma` | Width spread for logspread initialization. |
| `--breathing-preburn-steps`, `--breathing-preburn-log-step` | Optional initial equilibration using moves that rescale gaps without violating the hard core. |
| `--store-time` | Interval between stored energy measurements. |
| `--rms-collection-time`, `--density-collection-time` | Intervals between FW measurement starts. |
| `--rms-fw-times`, `--density-fw-times` | Sorted, distinct forward times including zero. Run duration must exceed the largest time. |
| `--grid-extent`, `--n-bins` | Requested grid half-width and number of grid centers; the grid may expand to contain the LDA cloud. |
| `--pure-fw-observables` | `r2,density`, or one of these. |
| `--pure-fw-observable-source` | `raw_r2` or `r2_rb` (internal squared radius plus exact center contribution). |
| `--pure-fw-density-source` | `raw_density` or `com_rao_blackwell` (average over the exact center distribution). |
| `--pure-fw-density-parity-average` | Average the density at q and -q using trap symmetry. |
| `--skip-plots`, `--plot-formats` | Numerical output only, or requested image formats. |

Exact center averaging follows [§4.5, Eqs. (4.22)–(4.23)](../Batuhan_Turgay_TFM.pdf#page=21).
FW transport is Eq. (4.21).

## What Umrigar and fixed population mean

Near contact, `v_i=d(log Psi_T)/dq_i` becomes large. With `h=dt`, the limiter is
`limited_v_i=2*v_i/(1+sqrt(1+2*h*v_i²))`. It reduces large drift moves; both
proposal directions use it in the acceptance test. This is
[Eq. (4.18), reference [14], Umrigar et al. (1993)](../Batuhan_Turgay_TFM.pdf#page=19).
It changes the proposal, not the Hamiltonian, and also applies to VMC MALA.

DMC keeps M walkers and changes their weights. At `ESS < 0.35*M`, resampling
selects M equally weighted copies. See [§4.3.3, reference [15], Assaraf et al.
(2000)](../Batuhan_Turgay_TFM.pdf#page=19). The 0.35 threshold was chosen for the thesis calculations; the paper does
not prescribe it as a universal constant.

## Plateau and genealogy controls

| Option | Thesis preset | Meaning |
|---|---:|---|
| `--pure-fw-min-block-count` | 20 | Minimum completed blocks per seed and tested lag; blocks may remain correlated. |
| `--pure-fw-min-walker-weight-ess` | 30 | Minimum recorded walker-weight ESS per seed and tested lag. |
| `--pure-fw-min-source-ancestor-ess` | 50 | Required lower bound on ancestor ESS after combining seeds. |
| `--pure-fw-max-source-family-fraction` | 0.10 | Allowed upper bound on one family's contribution after combining seeds. |
| `--pure-fw-rms-plateau-relative-tolerance` | 0.001 | Relative RMS change allowed across lags, including uncertainty (0.1%). |
| `--pure-fw-density-relative-tolerance` | 0.03 | Relative L2 density difference allowed, including uncertainty (3%, not per-bin). |
| `--pure-fw-plateau-equivalence-confidence-level` | 0.95 | Simultaneous confidence for all lag pairs, using paired seed differences. |
| `--pure-fw-plateau-window-lag-count` | 4 | Number of R2 lags to compare; the combined result needs at least 3 lags that pass the data checks. |
| `--pure-fw-density-plateau-window-lag-count` | 3 | Number of density lags to compare; the combined result needs at least 3 lags that pass the data checks. |
| `--pure-fw-block-size-steps` | 1 | Must be 1 for positive lags; change collection-time options to space measurements. |

The 50 ancestor ESS and 10% family limits are stated in
[§4.4](../Batuhan_Turgay_TFM.pdf#page=20); selected times are in
[Appendix C.3, Table C.2](../Batuhan_Turgay_TFM.pdf#page=37).
Other thresholds above come from the [matrix configuration](../../src/hrdmc/production/matrix/final_matrix.py).
Without the thesis preset, the benchmark defaults include 30 blocks and a zero RMS margin.
Choose margins before comparing results; zero is generally unsuitable for
noisy data. Passing these checks at finite forward times does not prove convergence at infinite time.

### Walker ESS and ancestor ESS are different checks

| Quantity | Calculation and example |
|---|---|
| Walker ESS | `1/sum(pi_k²)` for normalized weights. 256 equal weights give 256; one dominant weight gives approximately 1. |
| Ancestor ESS | Sum descendant weights by source family to obtain `F_j`, then `1/sum(F_j²)`. 256 equally weighted walkers in 8 equal families give ancestor ESS 8. |
| Largest family | `max(F_j)`: fraction of weight from one source family. |

The FW walker-ESS threshold checks the smallest recorded ESS during transport
and completed blocks for each seed and lag. It does not count independent
observations. The 0.35M threshold instead triggers resampling during simulation.

For S equally weighted independent seeds, let `e_s` be each seed's minimum
ancestor ESS and `f_s` its maximum family fraction. The calculation combines the seeds using
`S²/sum(1/e_s)` as the pooled ESS lower bound and `max(f_s)/S` as the family
upper bound. Thus five seeds with `e_s=10` give pooled ESS 50. These bounds
come from recorded minima and maxima, not Student-t intervals; see [FW support](../../src/hrdmc/production/benchmark/support.py).

FW method references [10–11] are Casulleras/Boronat (1995) and Sarsa et al.
(2002): [bibliography](../Batuhan_Turgay_TFM.pdf#page=40).

Additional weight alerts are `--ess-warning-fraction` (0.20),
`--ess-invalid-fraction` (0.10), and `--log-weight-span-warning` (50).
They flag concentrated walker weights separately from the resampling trigger
and the ancestry checks.
