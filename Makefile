PYTHON ?= python3

.PHONY: test check lint typecheck unit validate-ring validate-ring-grid validate-rn-exact validate-rn-trapped-stationarity clean

test: lint unit

check: lint typecheck

lint:
	$(PYTHON) -m ruff check

typecheck:
	PYTHONPATH=src $(PYTHON) -m pyright

unit:
	PYTHONPATH=src $(PYTHON) -m pytest

validate-ring:
	PYTHONPATH=src $(PYTHON) experiments/validation/homogeneous_ring.py

validate-ring-grid:
	PYTHONPATH=src $(PYTHON) experiments/validation/homogeneous_ring_exact_grid.py

validate-rn-exact:
	PYTHONPATH=src $(PYTHON) experiments/dmc/rn_block/exact_tg_trap.py

validate-rn-trapped-stationarity:
	PYTHONPATH=src $(PYTHON) experiments/dmc/rn_block/trapped_stationarity_grid.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
