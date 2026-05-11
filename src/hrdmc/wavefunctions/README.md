# Wavefunctions

Owner: trial states.

This package owns trial amplitude/log-amplitude evaluation, guide derivatives,
and trial-state parameters used by Monte Carlo samplers.

VMC diagnostic trial states and DMC importance-sampling guides are separate
forms even when they share hard-rod physics.

DMC guides may expose batch methods for log values, derivatives, local energy,
and validity masks. `ReducedTGHardRodGuide` uses the optional `dmc` extra for
Numba kernels when available, with a Python fallback for portability.
