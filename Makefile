PYTHON ?= python

.PHONY: verify train interp samples subset webapp test clean help

help:
	@echo "Targets:"
	@echo "  make verify      - verify locked-model performance on released subsets"
	@echo "  make subset      - regenerate synthetic data/ subsets"
	@echo "  make samples     - regenerate the 3 risk-stratum test samples"
	@echo "  make train       - full retrain on the full cohorts (~15 min)"
	@echo "  make interp      - 3-layer evidence chain (SHAP + drop-one + LASSO surrogate)"
	@echo "  make webapp      - launch the research web application (docker)"
	@echo "  make test        - run pytest"
	@echo "  make clean       - remove results/ and caches"

verify:
	$(PYTHON) analysis/verify_performance.py

subset:
	$(PYTHON) analysis/make_synthetic_data.py

samples:
	$(PYTHON) analysis/make_test_samples.py

train:
	$(PYTHON) analysis/01_train_multimodal.py

interp:
	$(PYTHON) analysis/02_interp_three_layer.py

webapp:
	cd web_app && docker compose up --build

test:
	$(PYTHON) -m pytest -q tests/

clean:
	rm -rf results/ __pycache__ **/__pycache__ web_app/__pycache__ .pytest_cache
