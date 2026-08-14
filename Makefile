PYTHON ?= python3
PY := PYTHONPATH=src $(PYTHON)

.PHONY: check check-science lint typecheck deadcode structure test imports \
        surface whitespace report-duplicates security clean

check: lint typecheck structure deadcode imports test surface whitespace

# Not in `check`: runs a short DMC case, too slow for the edit loop.
check-science:
	$(PY) scripts/checks/science.py

lint:
	$(PYTHON) -m ruff check

typecheck:
	$(PY) -m pyright

structure:
	$(PYTHON) scripts/checks/structure.py

# 100% only; the 60% default flags public API and dynamic dispatch.
deadcode:
	$(PYTHON) -m vulture src experiments --min-confidence 100 --sort-by-size

imports:
	$(PY) -c "import importlib, pkgutil, hrdmc; [importlib.import_module(m.name) for m in pkgutil.walk_packages(hrdmc.__path__, 'hrdmc.')]"

# tests/ is untracked, so this binds locally only; check-science carries reproducibility.
test:
	$(PY) -m pytest tests -q

surface:
	$(PYTHON) operator/audit_public_surface.py --root .

whitespace:
	git diff --check

# Report-only until triaged.
report-duplicates:
	-$(PYTHON) -m pylint src experiments \
	  --disable=all --enable=duplicate-code \
	  --reports=no --score=no \
	  --min-similarity-lines=12 \
	  --ignore-comments=yes --ignore-docstrings=yes \
	  --ignore-imports=yes --ignore-signatures=yes

# Outside `check`: network-dependent.
security:
	$(PYTHON) -m pip_audit

clean:
	rm -rf .pytest_cache .ruff_cache
	find src experiments tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find src experiments tests -type f -name '*.py[co]' -delete
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find src experiments tests -depth -type d -empty -delete
