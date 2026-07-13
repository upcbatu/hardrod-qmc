# Monte Carlo

Owner: sampling engines and result contracts.

This package owns VMC/DMC engines, walker data structures, and run result
contracts.

`vmc.py` owns the current simple Metropolis VMC engine. `dmc/local/` owns the
default importance-sampled DMC engine, including drift-diffusion, branching,
population control, streaming summaries, checkpoints, and transport events.
`dmc/collective_rn/` owns an optional scheduled collective proposal with its
explicit target-to-proposal Radon-Nikodym correction. The local engine does not
enable that extension unless a workflow supplies it.

DMC is the target production method. Each reported parameter row still needs
the relevant timestep, population, stationarity, and estimator checks.
