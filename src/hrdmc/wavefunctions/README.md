# Wavefunctions

Owner: trial and guide amplitudes.

`api.py` defines the DMC guide interface consumed by samplers. `trials/` owns
diagnostic trial amplitudes. `guides/` owns DMC guide classes and guide-specific
parameters. `kernels/` owns hot numerical backends only; kernels do not define
physics ownership or sampler behavior.

VMC diagnostic trial states and DMC importance-sampling guides stay separate
even when they share hard-rod formulas.

DMC guides may expose batch methods for log values, derivatives, local energy,
and validity masks. Guide classes choose their backend and report it through
`batch_backend`.
