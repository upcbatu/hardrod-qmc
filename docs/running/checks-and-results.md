# Check a finished run

Open `summary.json` in the output folder. The numbers are the estimates;
the check results tell you whether there is enough evidence to use them.

## Energy and sampling checks

| What you see | What it means | What to do |
|---|---|---|
| `completed_with_diagnostics` in VMC | The run finished; this does not mean the convergence checks passed. | Read the classification for each quantity under `diagnostics`. |
| `trace_nonstationary` in DMC | At least one energy-trace check failed. | Look for drift in the energy trace; investigate equilibration and sampling length. |
| Low time-series ESS or high R-hat | Too few effectively independent observations, or traces that disagree. | Check the individual runs before collecting more samples. |

The VMC examples store 1,000 blocks per seed and request ESS of at least 400.
Correlations reduce ESS; 1,000 blocks do not guarantee 400 independent samples.
VMC reports diagnostics only with at least two seeds and eight blocks per seed.

## Forward-walking checks

For DMC density and radius, open `pure_walking.observables.density` and `.r2`.
The result combines independent seeds and compares several forward times.

- Check the plateau result and the ancestry checks, not just the plotted curve.
- If the plateau is unresolved, the forward times may disagree or the errors
  may be too large to decide. Read the reason before choosing a longer run.
- If ancestry is insufficient, too few source families contribute. Simply
  increasing forward time can make this worse.

The [DMC guide](dmc-and-forward-walking.md#plateau-and-genealogy-controls)
explains the thresholds. A single seed cannot supply the combined-seed test.
Passing at the tested times does not prove the infinite-forward-time limit.

## What the error bar covers

The error bar describes sampling uncertainty. For VMC it is the standard
deviation of seed means divided by the square root of the seed count; with
one seed it is unavailable. DMC energy uses the largest of the across-seed,
blocking and autocorrelation-based estimates.

It does not automatically include errors from the time step, walker population
or finite forward time. See [thesis §5.1.4](../Batuhan_Turgay_TFM.pdf#page=24).

## Before reporting a new result

| Check | Change | Keep fixed |
|---|---|---|
| Time step | Reduce `--dt`. | Guide, walkers, physical run durations and FW/recording intervals. |
| Walker population | Compare M with 2M walkers. | Guide, dt, durations and measurement settings. |
| Forward time | Compare the collected positive FW times. | Same observable and run; keep the ancestry requirements. |

Compare the differences with their statistical uncertainty. If a change is
larger than the precision you need, investigate it before reporting the result.
To reproduce the thesis's numerical checks and uncertainty tables, use
[Reproduction](../reproducing.md), with the methods in
[Appendix B.3](../Batuhan_Turgay_TFM.pdf#page=33).
