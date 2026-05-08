# Equation and Method Source Map

This note links the main physics definitions, LDA formulas, and Monte Carlo support code to the papers or methodological conventions they rely on.

## Bibliography Layers

### Thesis backbone

- `[Mazzanti2008HardRods]` F. Mazzanti, G. E. Astrakharchik, J. Boronat, J. Casulleras, *Ground-State Properties of a One-Dimensional System of Hard Rods*, **Phys. Rev. Lett. 100**, 020401 (2008). DOI: [10.1103/PhysRevLett.100.020401](https://doi.org/10.1103/PhysRevLett.100.020401)
- `[AstrakharchikGiorgini2002TrappedCrossover]` G. E. Astrakharchik and S. Giorgini, *Quantum Monte Carlo study of the three- to one-dimensional crossover for a trapped Bose gas*, **Phys. Rev. A 66**, 053614 (2002). DOI: [10.1103/PhysRevA.66.053614](https://doi.org/10.1103/PhysRevA.66.053614)
- `[Astrakharchik2005LDA]` G. E. Astrakharchik, *Local density approximation for a perturbative equation of state*, **Phys. Rev. A 72**, 063620 (2005). DOI: [10.1103/PhysRevA.72.063620](https://doi.org/10.1103/PhysRevA.72.063620)
- `[GirardeauAstrakharchik2010TrappedHardSphere]` M. D. Girardeau and G. E. Astrakharchik, *Wave functions of the super-Tonks-Girardeau gas and the trapped one-dimensional hard-sphere Bose gas*, **Phys. Rev. A 81**, 061601(R) (2010). DOI: [10.1103/PhysRevA.81.061601](https://doi.org/10.1103/PhysRevA.81.061601)

### Method background

- `[FlyvbjergPetersen1989Blocking]` H. Flyvbjerg, H. G. Petersen, *Error estimates on averages of correlated data*, **J. Chem. Phys. 91**, 461-466 (1989). DOI: [10.1063/1.457480](https://doi.org/10.1063/1.457480)

### Optional extension

- `[LiebLiniger1963GroundState]` E. H. Lieb and W. Liniger, *Exact Analysis of an Interacting Bose Gas. I. The General Solution and the Ground State*, **Phys. Rev. 130**, 1605-1616 (1963). DOI: [10.1103/PhysRev.130.1605](https://doi.org/10.1103/PhysRev.130.1605)
- `[Lieb1963ExcitationSpectrum]` E. H. Lieb, *Exact Analysis of an Interacting Bose Gas. II. The Excitation Spectrum*, **Phys. Rev. 130**, 1616-1624 (1963). DOI: [10.1103/PhysRev.130.1616](https://doi.org/10.1103/PhysRev.130.1616)
- `[Astrakharchik2005Quasi1DHardRods]` G. E. Astrakharchik, J. Boronat, J. Casulleras, S. Giorgini, *Beyond the Tonks-Girardeau Gas: Strongly Correlated Regime in Quasi-One-Dimensional Bose Gases*, **Phys. Rev. Lett. 95**, 190407 (2005). DOI: [10.1103/PhysRevLett.95.190407](https://doi.org/10.1103/PhysRevLett.95.190407)

The proposal should cite only the thesis-backbone items. The method-background and Lieb-Liniger references should stay out of the main proposal unless those paths become active.

## Implemented Repository Map

| Repository location | Physics / method item | Scientific basis | Implementation note |
|---|---|---|---|
| [src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py) | Homogeneous one-dimensional hard rods on a periodic ring, including hard-core exclusion and boundary conventions | `[Mazzanti2008HardRods]` | Geometry and constraints only; EOS and LDA ownership lives in `theory/`. |
| [src/hrdmc/systems/open_line.py](src/hrdmc/systems/open_line.py) | Open-line hard-rod geometry and non-periodic hard-core constraint | Trapped hard-rod setup; contextual anchor `[GirardeauAstrakharchik2010TrappedHardSphere]` | Geometry and constraints only; no trap potential, EOS, LDA, or comparison logic. |
| [src/hrdmc/systems/external_potential.py](src/hrdmc/systems/external_potential.py) | Harmonic trap `V(x)=0.5 omega^2 (x-x0)^2` | Standard trapped-gas model; trapped QMC precedent `[AstrakharchikGiorgini2002TrappedCrossover]` | Potential owner only; systems layer does not solve LDA or estimate observables. |
| [src/hrdmc/theory/hard_rods.py](src/hrdmc/theory/hard_rods.py) | Excluded-volume mapping, finite-`N` ring energy, thermodynamic EOS, energy density, chemical potential, and chemical-potential inversion | `[Mazzanti2008HardRods]`, Eq. (4), and the excluded-volume mapping | Theory-layer homogeneous reference and LDA input. |
| [src/hrdmc/theory/lda.py](src/hrdmc/theory/lda.py) | LDA normalization, trapped density prediction, and LDA total-energy prediction | `[Astrakharchik2005LDA]` for LDA precedent; hard-rod EOS from `[Mazzanti2008HardRods]` | Theory-layer approximation, not analysis comparison logic. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py) | Reduced-coordinate all-pair hard-rod trial structure with `y_i = x_i - i a` and `L' = L - N a` | `[Mazzanti2008HardRods]` | Ring-oriented trial structure for homogeneous validation; reduced length comes from `theory/`. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L48-L59) | Nearest-neighbor Jastrow-like gap trial | Repository trial design informed by the same hard-rod constraint | Current smoke-test default; not a final trapped trial form. |
| [src/hrdmc/wavefunctions/trapped.py](src/hrdmc/wavefunctions/trapped.py) | Gaussian trap factor times hard-rod contact factor | Diagnostic VMC trial design; not a claimed exact trapped wavefunction | Used for trapped smoke diagnostics before DMC or external references are trusted. |
| [src/hrdmc/monte_carlo/vmc.py](src/hrdmc/monte_carlo/vmc.py#L27-L94) | Metropolis VMC with burn-in, thinning, and acceptance/rejection on `|Psi_T|^2` | Standard Variational Monte Carlo methodology | Current homogeneous smoke workflow. |
| [src/hrdmc/monte_carlo/dmc.py](src/hrdmc/monte_carlo/dmc.py) | DMC result contract, walker population support, and resampling support | Standard walker Monte Carlo conventions | Contract layer only; production DMC is not implemented. |
| [src/hrdmc/estimators/pair_distribution.py](src/hrdmc/estimators/pair_distribution.py#L22-L69) | Pair distribution function `g(r)` estimated from coordinate snapshots | `[Mazzanti2008HardRods]` for the observable; repository histogram normalization for implementation | Mainly useful for homogeneous validation unless adapted for trapped local analysis. |
| [src/hrdmc/estimators/structure_factor.py](src/hrdmc/estimators/structure_factor.py#L21-L54) | Static structure factor `S(k) = <|rho_k|^2>/N`, with `rho_k = sum_j exp(i k x_j)` using physical wrapped particle positions, not reduced coordinates | `[Mazzanti2008HardRods]` | Ring observable for homogeneous validation. |
| [src/hrdmc/estimators/density.py](src/hrdmc/estimators/density.py) | Periodic and open-line density profile estimators | Repository histogram normalization convention | Periodic density wraps ring coordinates; trapped density uses raw open-line coordinates on a chosen grid. |
| [src/hrdmc/analysis/blocking.py](src/hrdmc/analysis/blocking.py#L25-L64) | Blocking analysis for correlated Monte Carlo error bars | `[FlyvbjergPetersen1989Blocking]` | Used for QMC uncertainty estimates. |
| [src/hrdmc/analysis/metrics.py](src/hrdmc/analysis/metrics.py) | Bias, MSE, raw density L2 error, and relative density L2 error helpers | Standard statistical definitions | Small support utilities for comparison code. |
| [src/hrdmc/analysis/stability.py](src/hrdmc/analysis/stability.py) | Replicate spread summaries for diagnostic stability checks | Standard sample mean and sample standard deviation | Used to decide whether VMC diagnostic metrics are stable across seeds. |
| [experiments/trapped_vmc_common.py](experiments/trapped_vmc_common.py) | Shared trapped VMC diagnostic orchestration | Repository orchestration convention | Calls system, wavefunction, MC, estimator, theory, and analysis seams without owning their formulas. |
| [experiments/03_trapped_vmc_diagnostic_grid.py](experiments/03_trapped_vmc_diagnostic_grid.py) | Trapped VMC diagnostic grid over particle number and trap strength | Repository orchestration convention | Diagnostic artifact generation only; not final QMC/DMC benchmark evidence. |
| [experiments/04_trapped_vmc_seed_stability.py](experiments/04_trapped_vmc_seed_stability.py) | Trapped VMC seed-stability probe | Repository orchestration convention | Replicate diagnostic only; not final QMC/DMC benchmark evidence. |

## LDA Benchmark Formulas

The LDA itself is established background, not the thesis contribution. The thesis should implement these formulas so QMC and available benchmark data can test where excluded-volume LDA works and where it fails.

The homogeneous hard-rod equation of state is:

$$
e_{\mathrm{HR}}(\rho)
=\frac{\pi^2\rho^2}{3(1-a\rho)^2}.
$$

The homogeneous energy density is:

$$
\epsilon_{\mathrm{HR}}(\rho)
=\rho e_{\mathrm{HR}}(\rho)
=\frac{\pi^2\rho^3}{3(1-a\rho)^2}.
$$

The chemical potential is:

$$
\mu_{\mathrm{HR}}(\rho)
=\frac{d\epsilon_{\mathrm{HR}}}{d\rho}
=\frac{\pi^2\rho^2(3-a\rho)}{3(1-a\rho)^3}.
$$

For a trap:

$$
\mu_0 = V_{\mathrm{trap}}(x) + \mu_{\mathrm{HR}}\!\left(n_{\mathrm{LDA}}(x)\right).
$$

with `mu0` fixed by:

$$
\int n_{\mathrm{LDA}}(x)\,dx = N.
$$

The main benchmark errors are:

$$
\Delta E = E_{\mathrm{benchmark}} - E_{\mathrm{LDA}},
$$

$$
\Delta n_2 = \int
\left|n_{\mathrm{benchmark}}(x)-n_{\mathrm{LDA}}(x)\right|^2\,dx,
$$

with a dimensionless relative version:

$$
\delta n_2 =
\frac{
\left[\int |n_{\mathrm{benchmark}}(x)-n_{\mathrm{LDA}}(x)|^2\,dx\right]^{1/2}
}{
\left[\int |n_{\mathrm{LDA}}(x)|^2\,dx\right]^{1/2}
},
$$

$$
\Delta R = R_{\mathrm{benchmark}} - R_{\mathrm{LDA}}.
$$

## Reading Note

The literature-facing homogeneous validation code is concentrated in:

- `systems/`;
- `wavefunctions/`;
- `estimators/`.

The trapped thesis path now has an initial diagnostic route through:

- `systems/` for open-line hard rods and harmonic trapping;
- `estimators/` for non-periodic density profiles;
- `theory/` for homogeneous EOS and excluded-volume LDA predictions;
- `analysis/` for first density-error diagnostics.

It still needs parameter sweeps, stronger benchmark-tier handling, production DMC validation, and thesis-level failure-map analysis.
