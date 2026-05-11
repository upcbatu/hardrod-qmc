# Monte Carlo

Owner: sampling engines and result contracts.

This package owns VMC/DMC engines, walker data structures, and run result
contracts.

`vmc.py` owns the current simple Metropolis VMC engine. `dmc/` owns the generic
DMC contract plus concrete DMC implementations such as `dmc/rn_block/`.

DMC is the target production method but should be treated as a candidate
reference until validated.
