PYTHON ?= python3

.PHONY: test smoke validate-ring clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) experiments/00_smoke_vmc.py

validate-ring:
	PYTHONPATH=src $(PYTHON) experiments/01_uniform_hard_rods_validation.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
