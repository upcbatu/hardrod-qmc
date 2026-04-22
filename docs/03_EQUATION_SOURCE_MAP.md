# Equation and Method Source Map

This note links the main physics definitions, estimator formulas, and Monte Carlo implementations in the repository to the papers or methodological conventions they rely on. The aim is to make it easy to inspect which parts of the code correspond directly to the literature and which parts belong to the benchmark implementation.

## Primary references

1. F. Mazzanti, G. E. Astrakharchik, J. Boronat, J. Casulleras, *Ground-State Properties of a One-Dimensional System of Hard Rods*, **Phys. Rev. Lett. 100**, 020401 (2008). DOI: [10.1103/PhysRevLett.100.020401](https://doi.org/10.1103/PhysRevLett.100.020401)
2. J. Boronat, J. Casulleras, *Unbiased estimators in quantum Monte Carlo methods: Application to liquid helium*, **Phys. Rev. B 52**, 3654-3661 (1995). DOI: [10.1103/PhysRevB.52.3654](https://doi.org/10.1103/PhysRevB.52.3654)
3. A. Sarsa, J. Boronat, J. Casulleras, *Quadratic diffusion Monte Carlo and pure estimators for atoms*, **J. Chem. Phys. 116**, 5956-5962 (2002). DOI: [10.1063/1.1446847](https://doi.org/10.1063/1.1446847)
4. H. Flyvbjerg, H. G. Petersen, *Error estimates on averages of correlated data*, **J. Chem. Phys. 91**, 461-466 (1989). DOI: [10.1063/1.457480](https://doi.org/10.1063/1.457480)

## Repository map

| Repository location | Physics / method item | Scientific basis | Implementation note |
|---|---|---|---|
| [src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py#L12-L153) | 1D hard-rod benchmark system on a periodic ring, including hard-core exclusion and excluded-length mapping `L' = L - N a` | Mazzanti et al. (2008) | This file contains the main physical model used in the repository. |
| [src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py#L123-L153) | Finite-`N` hard-rod energy from reduced-length quasi-momenta and thermodynamic energy `E/N = pi^2 rho^2 / [3(1-rho a)^2]` in units `hbar^2/(2m)=1` | Mazzanti et al. (2008), Eq. (4) and the associated excluded-volume mapping | The code stores both the finite-system benchmark and the thermodynamic-limit expression. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L18-L27), [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L61-L76) | Reduced-coordinate all-pair hard-rod trial structure with `y_i = x_i - i a` and `L' = L - N a` | Mazzanti et al. (2008) | This is the hard-rod trial form in the present implementation. |
| [src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py#L48-L59) | Nearest-neighbor Jastrow-like gap trial | Repository trial design informed by the same hard-rod constraint | This is the default trial form used by the present VMC implementation. |
| [src/hrdmc/monte_carlo/vmc.py](src/hrdmc/monte_carlo/vmc.py#L27-L94) | Metropolis VMC with burn-in, thinning, and acceptance/rejection on `|Psi_T|^2` | Standard Variational Monte Carlo methodology | This file contains the present VMC workflow. |
| [src/hrdmc/monte_carlo/dmc.py](src/hrdmc/monte_carlo/dmc.py#L13-L126) | DMC result contract together with walker populations, branching support, and forward-walking ancestry support | Boronat and Casulleras (1995); Sarsa, Boronat, Casulleras (2002); standard walker Monte Carlo conventions | This file contains the present DMC implementation layer. |
| [src/hrdmc/estimators/pair_distribution.py](src/hrdmc/estimators/pair_distribution.py#L22-L69) | Pair distribution function `g(r)` estimated from coordinate snapshots | Mazzanti et al. (2008) for the observable; repository histogram normalization for implementation | The physical observable is literature-facing; the binning convention is a code-level choice. |
| [src/hrdmc/estimators/structure_factor.py](src/hrdmc/estimators/structure_factor.py#L21-L54) | Static structure factor `S(k) = <|rho_k|^2>/N`, with `rho_k = sum_j exp(i k x_j)` | Mazzanti et al. (2008) | This is one of the central hard-rod benchmark observables. |
| [src/hrdmc/analysis/estimator_families.py](src/hrdmc/analysis/estimator_families.py#L14-L103) | Construction of VMC, mixed, extrapolated, and pure estimator families, including `O_ext = 2 O_mixed - O_VMC` | Standard estimator-family definitions from DMC methodology; extrapolated-estimator convention widely used in QMC benchmarking | This file contains the estimator-family combination logic used in the repository. |
| [src/hrdmc/analysis/blocking.py](src/hrdmc/analysis/blocking.py#L25-L64) | Blocking analysis for correlated Monte Carlo error bars | Flyvbjerg and Petersen (1989) | Used because VMC and DMC samples are correlated. |
| [src/hrdmc/analysis/metrics.py](src/hrdmc/analysis/metrics.py#L4-L19), [src/hrdmc/analysis/cost_accuracy.py](src/hrdmc/analysis/cost_accuracy.py#L8-L42) | Bias, mean-squared error, and runtime-weighted estimator scoring | Standard statistical definitions combined with the repository’s benchmarking objective | These quantities support the thesis comparison of mixed, extrapolated, and pure estimators. |

## Reading note

The literature-facing physics is concentrated in:

- `systems/` for the hard-rod benchmark itself;
- `wavefunctions/` for the reduced-coordinate hard-rod trial structure;
- `estimators/` for `g(r)` and `S(k)`.

The literature-facing estimator methodology is concentrated in:

- `monte_carlo/dmc.py`.

The benchmarking layer for thesis comparison is concentrated in:

- `analysis/estimator_families.py`;
- `analysis/metrics.py`;
- `analysis/cost_accuracy.py`.
