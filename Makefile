PYTHON ?= python3

.PHONY: test smoke validate-ring smoke-trap diagnose-trap-grid diagnose-trap-seeds clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) experiments/00_smoke_vmc.py

validate-ring:
	PYTHONPATH=src $(PYTHON) experiments/01_uniform_hard_rods_validation.py

smoke-trap:
	PYTHONPATH=src $(PYTHON) experiments/02_trapped_vmc_smoke.py

diagnose-trap-grid:
	PYTHONPATH=src $(PYTHON) experiments/03_trapped_vmc_diagnostic_grid.py

diagnose-trap-seeds:
	PYTHONPATH=src $(PYTHON) experiments/04_trapped_vmc_seed_stability.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
