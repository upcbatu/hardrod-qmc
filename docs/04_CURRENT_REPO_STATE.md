# Current Repository State

Active thesis title: **Microscopic description of trapped hard rods**.

The repository implements a local importance-sampled DMC workflow for trapped
one-dimensional hard rods. Local drift-diffusion is the default trajectory.
Collective Radon-Nikodym-corrected moves are available as an optional scheduled
extension and are disabled unless explicitly requested.

## Implemented

- periodic and open-line one-dimensional hard-rod geometry;
- reduced-coordinate hard-rod identities and hard-core validity checks;
- harmonic-oscillator units for trapped cases, with canonical case identifiers
  such as `N10_A1` where \(A=a/a_{\rm ho}\);
- homogeneous hard-rod EOS, finite-ring energy, chemical-potential inversion,
  and excluded-volume LDA support under `src/hrdmc/theory/`;
- exact and deterministic anchor workflows under `experiments/anchors/`;
- a Metropolis VMC baseline under `src/hrdmc/monte_carlo/vmc.py`;
- the default DMC engine under `src/hrdmc/monte_carlo/dmc/local/`, including
  drift-diffusion, branching, population control, streaming summaries,
  checkpoints, telemetry, and transport events;
- the optional collective move and its explicit target-to-proposal correction
  under `src/hrdmc/monte_carlo/dmc/collective_rn/`;
- DMC workflow composition under `src/hrdmc/workflows/dmc/` and thin user
  commands under `experiments/dmc/local/`;
- mixed local-energy estimates and transported forward-walking estimators for
  \(R^2\), RMS radius, density, pair-distance density, and structure factor;
- an independent Hellmann-Feynman energy-response route for trap \(R^2\) and
  RMS radius;
- blocking, autocorrelation, chain-agreement, population, mobility, and
  genealogy diagnostics;
- non-periodic density estimation and QMC-versus-LDA comparison metrics;
- canonical artifact routing plus JSON, CSV, NPZ, and plotting support.

Benchmark packets expose observable data under `estimates`. Forward-walking
\(R^2\) summaries expose the derived radius as `rms_radius` and its uncertainty
as `rms_radius_stderr`.

## Active Commands

```bash
make validate-ring
make validate-ring-grid
make validate-dmc-exact
make validate-dmc-trapped-stationarity
```

The trapped stationarity command runs local DMC by default. Its CLI documents
the explicit option for enabling collective RN transport when that comparison
is scientifically relevant.

## Validation Status

Deterministic validation covers hard-rod geometry, harmonic trapping, guide and
local-energy formulas, DMC transitions and accounting, estimator invariants,
theory formulas, LDA normalization, analysis utilities, and workflow contracts.

The main numerical anchors are:

- exact homogeneous finite-ring hard-rod energies;
- the trapped Tonks-Girardeau harmonic limit at \(A=0\);
- a deterministic finite-\(A\), \(N=2\) relative-coordinate reference;
- timestep, population, stationarity, and estimator checks for finite-\(A\)
  trapped rows.

Passing an exact limiting case validates that limit and the associated
conventions. A finite-\(A\), many-body result is reported only with its own
timestep, population, stationarity, and observable-specific estimator evidence.

## Remaining Work

- finish the high-\(A\) guide and population calibration before the final long
  matrix;
- complete the selected timestep and walker-population checks for every final
  parameter row;
- archive the final result bundle with configuration, seed-level summaries,
  diagnostics, and code revision;
- produce the final DMC/forward-walking versus LDA figures and thesis tables.

Report tables are generated from manifest-verified run artifacts. Historical
collective-run CSV snapshots are not a current local-DMC result source and are
not tracked as thesis evidence.
