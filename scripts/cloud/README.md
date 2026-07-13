# Cloud DMC Runs

These utilities prepare a Linux VM for local importance-sampled DMC campaigns.
The optional collective RN move remains an explicit run setting; it is not the
name of the base calculation.

## Bootstrap

```bash
bash scripts/cloud/bootstrap_vm.sh
```

This installs system build tools, creates `.venv`, installs `.[dmc,dev]`, and
runs the local checks unless `HRDMC_SKIP_CHECKS=1`.
