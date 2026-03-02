# ─────────────────────────────────────────────────────────────────
# Financial PDF Pipeline — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────

PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
VENV_DIR := .venv

.DEFAULT_GOAL := help

.PHONY: help setup install env sample-data run-cli run-ui run-api run-tests clean

help:   ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment setup ─────────────────────────────────────────────
setup: ## Create venv + install deps + copy .env
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@[ -f .env ] || cp .env.example .env && echo "\n>>> Add your ANTHROPIC_API_KEY to .env <<<"

install: ## Install/update requirements only
	$(PIP) install -r requirements.txt

env: ## Copy .env.example to .env (if not exists)
	@[ -f .env ] && echo ".env already exists" || (cp .env.example .env && echo "Created .env — add your ANTHROPIC_API_KEY")

# ── Data ──────────────────────────────────────────────────────────
sample-data: ## Generate 12 sample financial PDFs
	$(PYTHON) scripts/generate_sample_pdfs.py

# ── Run ───────────────────────────────────────────────────────────
run-cli: ## Run pipeline on sample PDFs (CLI)
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py

run-cli-query: ## Run pipeline with a test query
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py \
	  --query "What was the total revenue in 2023?"

run-ui: ## Launch Streamlit UI at http://localhost:8501
	PYTHONPATH=. .venv/bin/streamlit run ui/app.py --server.port 8501

run-api: ## Launch FastAPI at http://localhost:8000/docs
	PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000

# ── Tests ─────────────────────────────────────────────────────────
run-tests: ## Run pytest suite
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short

# ── Clean ─────────────────────────────────────────────────────────
clean: ## Remove __pycache__, .pytest_cache, vectorstore
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -rf data/vectorstore/* data/pdfs/*/

clean-venv: ## Delete the virtual environment
	rm -rf $(VENV_DIR)
