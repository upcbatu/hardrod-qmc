# Cloud RN-DMC Runs

These scripts are thin operators for running RN-DMC campaigns on a Linux VM.

## Bootstrap

```bash
bash scripts/cloud/bootstrap_vm.sh
```

This installs system build tools, creates `.venv`, installs `.[dmc,dev]`, and
runs the local checks unless `HRDMC_SKIP_CHECKS=1`.

