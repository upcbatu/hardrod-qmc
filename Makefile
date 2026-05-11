PYTHON ?= python3

.PHONY: test check lint typecheck unit smoke validate-ring validate-ring-grid validate-rn-exact validate-rn-trapped-stationarity smoke-trap diagnose-trap-grid diagnose-trap-seeds scan-trap-alpha clean

test: lint unit

check: lint typecheck

lint:
	$(PYTHON) -m ruff check

typecheck:
	PYTHONPATH=src $(PYTHON) -m pyright

unit:
	PYTHONPATH=src $(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) experiments/vmc/smoke.py

validate-ring:
	PYTHONPATH=src $(PYTHON) experiments/validation/homogeneous_ring.py

validate-ring-grid:
	PYTHONPATH=src $(PYTHON) experiments/validation/homogeneous_ring_exact_grid.py

validate-rn-exact:
	PYTHONPATH=src $(PYTHON) experiments/dmc/rn_block/exact_tg_trap.py

validate-rn-trapped-stationarity:
	PYTHONPATH=src $(PYTHON) experiments/dmc/rn_block/trapped_stationarity_grid.py

smoke-trap:
	PYTHONPATH=src $(PYTHON) experiments/vmc/trapped_smoke.py

diagnose-trap-grid:
	PYTHONPATH=src $(PYTHON) experiments/vmc/trapped_diagnostic_grid.py

diagnose-trap-seeds:
	PYTHONPATH=src $(PYTHON) experiments/vmc/trapped_seed_stability.py

scan-trap-alpha:
	PYTHONPATH=src $(PYTHON) experiments/vmc/trapped_alpha_scan.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
