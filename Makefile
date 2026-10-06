PYTHON ?= python
DATA_DIR ?= data/ptb-xl/1.0.3
RUN_DIR ?= artifacts/ptbxl-resnet1d-v1

.PHONY: test train evaluate dev build lint smoke
test:
	$(PYTHON) -m pytest -q
train:
	$(PYTHON) -m training.train --data-dir $(DATA_DIR) --run-dir $(RUN_DIR)
evaluate:
	$(PYTHON) -m training.evaluate --data-dir $(DATA_DIR) --run-dir $(RUN_DIR)
dev:
	$(PYTHON) -m uvicorn app.main:app --app-dir backend --env-file .env --reload --port 8000
build:
	npm --prefix frontend run build
lint:
	npm --prefix frontend run lint
smoke:
	$(PYTHON) -m scripts.smoke
