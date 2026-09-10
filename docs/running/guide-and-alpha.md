# Guide and alpha

For an existing thesis case, use `--preset thesis`; no optimization is needed.
For a new system, search a broad range, narrow the search around a minimum,
then check the selected alpha with independent samples. All commands run from the repository root with
the [environment activated](../../README.md#installation).

## Explore a wider alpha range

The command searches overlapping intervals covering approximately **0.335–11.935**.
It collects a fresh MALA sample at each center. Change the center list to
search another range; the centers are starting values, not assumed answers.

```bash
(
  set -e
  ALPHA_SCAN_ROOT=results/alpha_scan_N10_A1
  alpha_seed=9401
  for center in 0.5 1 2 4 8; do
    python experiments/run.py optimize-alpha \
      --n 10 \
      --a 1 \
      --alpha-center "$center" \
      --alpha-log-half-width 0.4 \
      --alpha-grid-points 41 \
      --walkers 256 \
      --dt 0.001 \
      --drift-limiter umrigar \
      --burn-in-steps 20000 \
      --production-steps 20000 \
      --sample-stride-steps 100 \
      --min-reweight-ess-fraction 0.1 \
      --max-configurations 100000 \
      --grid-extent 35 \
      --n-bins 840 \
      --seed "$alpha_seed" \
      --output "$ALPHA_SCAN_ROOT/center_${center}_seed_${alpha_seed}"
    alpha_seed=$((alpha_seed + 1))
  done
)
```

Use a new `ALPHA_SCAN_ROOT` for another scan. Output directories must be new
or empty; interrupted seeds are not resumed. `--dry-run --verbose-json` shows
settings without running. Check that the example run lengths allow the sample to equilibrate.

## Refine a promising region

Earlier N10_A1 searches found a minimum near 1.6. This example
searches **1.310–1.954**. For another system, use a center from your broad scan.

```bash
python experiments/run.py optimize-alpha \
  --n 10 \
  --a 1 \
  --alpha-center 1.6 \
  --alpha-log-half-width 0.2 \
  --alpha-grid-points 31 \
  --walkers 256 \
  --dt 0.001 \
  --drift-limiter umrigar \
  --burn-in-steps 20000 \
  --production-steps 20000 \
  --sample-stride-steps 100 \
  --min-reweight-ess-fraction 0.1 \
  --max-configurations 100000 \
  --grid-extent 35 \
  --n-bins 840 \
  --seed 9501 \
  --output results/my_alpha
```

The selected alpha and search status are in `summary.json`; the variance curve
is in `candidate_grid.csv`.
If the minimum is at the edge, extend the search. If overlap is too small,
collect a fresh sample nearer the selected alpha. Check `search.refinement_success`
and repeat the search near the minimum with new seeds; the variance curves
have no reported confidence intervals. A scan does not prove a global minimum.

The alpha values used for the thesis results are in [thesis Table 5.3](../Batuhan_Turgay_TFM.pdf#page=24)
and the [guide registry](../../data/final_matrix_guides/summary.json).

## What alpha and the controls mean

Alpha changes the guide's **internal Gaussian width**, leaving the exact
center-of-mass factor unchanged: [§4.1, Eq. (4.2)](../Batuhan_Turgay_TFM.pdf#page=16).
Both VMC and DMC use this guide. The optimizer minimizes estimated local-energy
variance under `Psi_T²`, as described after Eq. (4.6). It does not minimize
energy or the variance under the DMC mixed distribution.

| Option | Meaning |
|---|---|
| `--alpha-center` | Alpha used to collect the sample, not the answer. |
| `--alpha-log-half-width` | Search from `center*exp(-width)` to `center*exp(width)`. |
| `--alpha-grid-points` | Number of initial alpha values to compare; the search then refines within the same interval. |
| `--walkers` | Number of sampled configurations. Each contains N rods. |
| `--dt` | MALA proposal scale: drift times dt plus Gaussian noise times `sqrt(dt)`. Here it is not a DMC extrapolation step. |
| `--drift-limiter` | Limits large near-contact drifts; [Umrigar explanation](dmc-and-forward-walking.md#what-umrigar-and-fixed-population-mean). |
| `--burn-in-steps`, `--production-steps` | Discarded equilibration steps and measured steps. |
| `--sample-stride-steps` | Keep one configuration batch every this many steps. |
| `--max-configurations` | Storage limit on retained configuration statistics. |
| `--min-reweight-ess-fraction` | Minimum weight-overlap fraction; see below. |
| `--grid-extent`, `--n-bins` | Spatial grid for LDA-based initialization, not the alpha search grid. |
| `--seed` | Random seed; use new seeds for independent checks. |

The example retains `256 * 20000/100 = 51,200` configuration records.
More alpha grid points reuse these records; they do not add samples.

For saved configuration m, `U_m=sum_i(u_i²)` and the reweighting ratio is
`w_m=exp(-(alpha-alpha_center)*U_m)`. With normalized weights `p_m`,
`ESS=1/sum(p_m²)` and `ESS fraction=ESS/number_of_records`.
The 0.1 threshold requires an ESS of at least 10% of the saved record count; it is not
10% accuracy or a count corrected for time correlations. Even fraction 1
at the center does not prove equilibration. See the
[current optimizer](../../src/hrdmc/trial/relative_width.py); search bounds and
thresholds are set in the code and are not specified in the thesis.

## Check the candidate independently

Run with new seeds from compact and expanded starting configurations, as in thesis §4.1:

```bash
python experiments/run.py vmc \
  --n 10 \
  --a 1 \
  --alpha-from results/my_alpha/summary.json \
  --sampler mala \
  --mala-step 0.04 \
  --drift-limiter umrigar \
  --block-size 20 \
  --initial-scale 0.5 \
  --seeds 8101,8102,8103 \
  --workers 3 \
  --output results/alpha_compact

python experiments/run.py vmc \
  --n 10 \
  --a 1 \
  --alpha-from results/my_alpha/summary.json \
  --sampler mala \
  --mala-step 0.04 \
  --drift-limiter umrigar \
  --block-size 20 \
  --initial-scale 2 \
  --seeds 8201,8202,8203 \
  --workers 3 \
  --output results/alpha_expanded
```

Compare energy, radius, density and [trace diagnostics](checks-and-results.md#energy-and-sampling-checks)
between the two starts; low energy variance alone does not show that density has converged.

## Use the candidate in DMC

```bash
python experiments/run.py dmc \
  --preset thesis \
  --case N10_A1 \
  --alpha-from results/my_alpha/summary.json \
  --density-fw-times 0,2,4,7 \
  --output results/dmc_new_alpha
```

`--alpha-from` reads a candidate for the same N and A. Alternatively use
`--alpha VALUE`; both options work in VMC too. They override the preset guide
and remain labelled `not_independently_validated`. They do not replace the
saved thesis alpha values. At A=0 the default alpha=1 is exact and needs no search.

Changing alpha changes VMC averages. In converged DMC it should leave the
projected ground state unchanged, but can change errors due to the time step, walker population and
forward time. Recheck those before reporting results with a different guide:
[§4.3–4.4](../Batuhan_Turgay_TFM.pdf#page=18).
