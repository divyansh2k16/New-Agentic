# ─────────────────────────────────────────────────────────────────
# Financial PDF Pipeline — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────

PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
VENV_DIR := .venv

.DEFAULT_GOAL := help

.PHONY: help setup install env run-cli run-cli-query run-ui run-api run-tests clean clean-venv docs-folder

help:   ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment setup ─────────────────────────────────────────────
setup: ## Create venv + install deps + copy .env
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@mkdir -p data/pdfs/my_documents data/vectorstore
	@[ -f .env ] || (cp .env.example .env && echo "\n>>> .env created. Add your ANTHROPIC_API_KEY <<<")

install: ## Install/update requirements only
	$(PIP) install -r requirements.txt

env: ## Copy .env.example to .env (if not exists)
	@[ -f .env ] && echo ".env already exists" || (cp .env.example .env && echo "Created .env — add your ANTHROPIC_API_KEY")

docs-folder: ## Print the path to drop your PDF files
	@echo ""
	@echo "  Drop your PDF files here:"
	@echo "  $$(pwd)/data/pdfs/my_documents/"
	@echo ""
	@ls data/pdfs/my_documents/*.pdf 2>/dev/null | wc -l | xargs -I{} echo "  {} PDF(s) currently in folder"
	@echo ""

# ── Run ───────────────────────────────────────────────────────────
run-cli: ## Run pipeline on your PDFs in data/pdfs/my_documents/
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py

run-cli-query: ## Run pipeline + ask a question (edit the query below)
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py \
	  --query "What was the total revenue in 2023?"

run-cli-folder: ## Run on a custom folder: make run-cli-folder FOLDER=~/Downloads
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py --folder $(FOLDER)

run-ui: ## Launch Streamlit UI at http://localhost:8501
	PYTHONPATH=. .venv/bin/streamlit run ui/app.py --server.port 8501

run-api: ## Launch FastAPI at http://localhost:8000/docs
	PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000

# ── Tests ─────────────────────────────────────────────────────────
run-tests: ## Run pytest suite
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short

# ── Clean ─────────────────────────────────────────────────────────
clean: ## Remove cache and vector store (keeps your PDFs)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -rf data/vectorstore/*
	@echo "Vector store cleared. Your PDFs in my_documents/ are untouched."

clean-venv: ## Delete the virtual environment
	rm -rf $(VENV_DIR)
