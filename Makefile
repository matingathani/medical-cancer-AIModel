# Convenience targets for medical-cancer-cnn
# Use: make install, make data, make train, etc.

PYTHON ?= python3
VENV = .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: install venv data data-pcam train eval demo test clean help

help:
	@echo "Medical Cancer CNN - targets:"
	@echo "  make install    - create venv and install dependencies"
	@echo "  make data       - create dummy data (scripts/create_dummy_data.py)"
	@echo "  make data-pcam  - download PatchCamelyon subset (scripts/download_pcam.py)"
	@echo "  make train      - train model (python train.py)"
	@echo "  make eval       - evaluate best checkpoint (python evaluate.py)"
	@echo "  make demo       - run full demo (python run_demo.py)"
	@echo "  make test       - run pytest tests"
	@echo "  make clean      - remove outputs and cache"

install: venv
	$(ACTIVATE) && pip install -r requirements.txt
	@echo "Run: source $(VENV)/bin/activate  (or .venv\\Scripts\\activate on Windows)"

venv:
	$(PYTHON) -m venv $(VENV)

data:
	$(PYTHON) scripts/create_dummy_data.py

data-pcam:
	$(PYTHON) scripts/download_pcam.py --max-train 400 --max-val 100 --max-test 100

train:
	$(ACTIVATE) && $(PYTHON) train.py

eval:
	$(ACTIVATE) && $(PYTHON) evaluate.py --checkpoint outputs/best.pt

demo:
	$(ACTIVATE) && $(PYTHON) run_demo.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf outputs outputs_demo __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
