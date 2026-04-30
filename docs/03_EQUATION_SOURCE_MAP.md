# Equation and Method Source Map

This note links the main physics definitions, planned LDA formulas, and Monte Carlo support code to the papers or methodological conventions they rely on.

## Bibliography Layers

### Thesis backbone

- `[Mazzanti2008HardRods]` F. Mazzanti, G. E. Astrakharchik, J. Boronat, J. Casulleras, *Ground-State Properties of a One-Dimensional System of Hard Rods*, **Phys. Rev. Lett. 100**, 020401 (2008). DOI: [10.1103/PhysRevLett.100.020401](https://doi.org/10.1103/PhysRevLett.100.020401)
- `[AstrakharchikGiorgini2002TrappedCrossover]` G. E. Astrakharchik and S. Giorgini, *Quantum Monte Carlo study of the three- to one-dimensional crossover for a trapped Bose gas*, **Phys. Rev. A 66**, 053614 (2002). DOI: [10.1103/PhysRevA.66.053614](https://doi.org/10.1103/PhysRevA.66.053614)
- `[Astrakharchik2005LDA]` G. E. Astrakharchik, *Local density approximation for a perturbative equation of state*, **Phys. Rev. A 72**, 063620 (2005). DOI: [10.1103/PhysRevA.72.063620](https://doi.org/10.1103/PhysRevA.72.063620)
- `[GirardeauAstrakharchik2010TrappedHardSphere]` M. D. Girardeau and G. E. Astrakharchik, *Wave functions of the super-Tonks-Girardeau gas and the trapped one-dimensional hard-sphere Bose gas*, **Phys. Rev. A 81**, 061601(R) (2010). DOI: [10.1103/PhysRevA.81.061601](https://doi.org/10.1103/PhysRevA.81.061601)

### Method background

- `[BoronatCasulleras1995PureEstimators]` J. Boronat, J. Casulleras, *Unbiased estimators in quantum Monte Carlo methods: Application to liquid helium*, **Phys. Rev. B 52**, 3654-3661 (1995). DOI: [10.1103/PhysRevB.52.3654](https://doi.org/10.1103/PhysRevB.52.3654)
- `[SarsaBoronatCasulleras2002QuadraticDMC]` A. Sarsa, J. Boronat, J. Casulleras, *Quadratic diffusion Monte Carlo and pure estimators for atoms*, **J. Chem. Phys. 116**, 5956-5962 (2002). DOI: [10.1063/1.1446847](https://doi.org/10.1063/1.1446847)
- `[FlyvbjergPetersen1989Blocking]` H. Flyvbjerg, H. G. Petersen, *Error estimates on averages of correlated data*, **J. Chem. Phys. 91**, 461-466 (1989). DOI: [10.1063/1.457480](https://doi.org/10.1063/1.457480)

### Optional extension

- `[LiebLiniger1963GroundState]` E. H. Lieb and W. Liniger, *Exact Analysis of an Interacting Bose Gas. I. The General Solution and the Ground State*, **Phys. Rev. 130**, 1605-1616 (1963). DOI: [10.1103/PhysRev.130.1605](https://doi.org/10.1103/PhysRev.130.1605)
- `[Lieb1963ExcitationSpectrum]` E. H. Lieb, *Exact Analysis of an Interacting Bose Gas. II. The Excitation Spectrum*, **Phys. Rev. 130**, 1616-1624 (1963). DOI: [10.1103/PhysRev.130.1616](https://doi.org/10.1103/PhysRev.130.1616)
- `[Astrakharchik2005Quasi1DHardRods]` G. E. Astrakharchik, J. Boronat, J. Casulleras, S. Giorgini, *Beyond the Tonks-Girardeau Gas: Strongly Correlated Regime in Quasi-One-Dimensional Bose Gases*, **Phys. Rev. Lett. 95**, 190407 (2005). DOI: [10.1103/PhysRevLett.95.190407](https://doi.org/10.1103/PhysRevLett.95.190407)

The proposal should cite only the thesis-backbone items plus `[BoronatCasulleras1995PureEstimators]` if DMC estimator machinery remains in the text. The Lieb-Liniger references should stay out of the main proposal unless the optional extension becomes active.

## Implemented Repository Map

| Repository location | Physics / method item | Scientific basis | Implementation note |
|---|---|---|---|
| [src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py#L12-L153) | Homogeneous one-dimensional hard rods on a periodic ring, including hard-core exclusion and excluded-length mapping `L' = L - N a` | `[Mazzanti2008HardRods]` | This is now the validation benchmark, not the final trapped-system target. |
| [src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py#L123-L153) | Finite-`N` energy from reduced-length quasi-momenta and thermodynamic energy \(E/N = \pi^2\rho^2/[3(1-\rho a)^2]\) in units \(\hbar^2/(2m)=1\) | `[Mazzanti2008HardRods]`, Eq. (4), and the excluded-volume mapping | This is the reference for homogeneous validation and the base equation of state for LDA. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L18-L27), [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L61-L76) | Reduced-coordinate all-pair hard-rod trial structure with `y_i = x_i - i a` and `L' = L - N a` | `[Mazzanti2008HardRods]` | Ring-oriented trial structure for homogeneous validation. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L48-L59) | Nearest-neighbor Jastrow-like gap trial | Repository trial design informed by the same hard-rod constraint | Current smoke-test default; not a final trapped trial form. |
| [src/hrdmc/monte_carlo/vmc.py](src/hrdmc/monte_carlo/vmc.py#L27-L94) | Metropolis VMC with burn-in, thinning, and acceptance/rejection on `|Psi_T|^2` | Standard Variational Monte Carlo methodology | Current homogeneous smoke workflow. |
| [src/hrdmc/monte_carlo/dmc.py](src/hrdmc/monte_carlo/dmc.py#L13-L126) | DMC result contract, walker population support, branching support, and optional forward-walking ancestry support | `[BoronatCasulleras1995PureEstimators]`; `[SarsaBoronatCasulleras2002QuadraticDMC]`; standard walker Monte Carlo conventions | Contract layer only; production DMC is not implemented. |
| [src/hrdmc/estimators/pair_distribution.py](src/hrdmc/estimators/pair_distribution.py#L22-L69) | Pair distribution function `g(r)` estimated from coordinate snapshots | `[Mazzanti2008HardRods]` for the observable; repository histogram normalization for implementation | Mainly useful for homogeneous validation unless adapted for trapped local analysis. |
| [src/hrdmc/estimators/structure_factor.py](src/hrdmc/estimators/structure_factor.py#L21-L54) | Static structure factor `S(k) = <|rho_k|^2>/N`, with `rho_k = sum_j exp(i k x_j)` | `[Mazzanti2008HardRods]` | Ring observable for homogeneous validation. |
| [src/hrdmc/estimators/density.py](src/hrdmc/estimators/density.py#L21-L34) | Periodic density profile estimator | Repository implementation convention | Currently wraps onto a ring; trapped density requires an open-line variant. |
| [src/hrdmc/analysis/blocking.py](src/hrdmc/analysis/blocking.py#L25-L64) | Blocking analysis for correlated Monte Carlo error bars | `[FlyvbjergPetersen1989Blocking]` | Used for QMC uncertainty estimates. |
| [src/hrdmc/analysis/estimator_families.py](src/hrdmc/analysis/estimator_families.py#L14-L103) | VMC, mixed, extrapolated, and pure estimator labels and combinations | Standard DMC estimator conventions | Support infrastructure, not the main thesis endpoint. |
| [src/hrdmc/analysis/metrics.py](src/hrdmc/analysis/metrics.py#L4-L19), [src/hrdmc/analysis/cost_accuracy.py](src/hrdmc/analysis/cost_accuracy.py#L8-L42) | Bias, MSE, and runtime-weighted score helpers | Standard statistical definitions | Support diagnostics for numerical comparisons. |

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

$$
\Delta R = R_{\mathrm{benchmark}} - R_{\mathrm{LDA}}.
$$

## Reading Note

The literature-facing homogeneous validation code is concentrated in:

- `systems/`;
- `wavefunctions/`;
- `estimators/`.

The trapped thesis path still needs new implementation in:

- `systems/` for open-line hard rods and harmonic trapping;
- `estimators/` for non-periodic density profiles;
- `analysis/` for the LDA implementation and benchmark-versus-LDA failure-map comparison.
