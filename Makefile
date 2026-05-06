.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---- Local install vs cloud install ----------------------------------------
# Local Mac (arm64): default torch wheel is MPS-capable. Just `make install`.
# Cloud Linux+CUDA:  use `make install-cuda` (selects CUDA torch wheels).

.PHONY: help install install-cuda lint format typecheck test smoke train clean cloud-setup cloud-train pull-results check

help: ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## sync local env (Mac MPS / CPU)
	uv sync --extra dev --extra cpu

install-cuda: ## sync cloud env (CUDA 12.1 wheels)
	uv sync --extra dev --extra cuda

lint: ## ruff lint
	uv run ruff check .

format: ## ruff format + autofix
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## pyright (basic globally, strict on src/)
	uv run pyright

test: ## pytest (excluding gpu-marked tests)
	uv run pytest -m "not gpu"

smoke: ## one-batch overfit — must drive loss → ~0
	uv run python scripts/overfit_one_batch.py

train: ## launch training (overrides: make train OVERRIDES="trainer=cloud model.hidden_dim=256")
	uv run python -m ml_template.cli.train $(OVERRIDES)

check: lint typecheck test ## run lint + types + tests (the standard pre-push check)

clean: ## remove caches and runtime outputs
	rm -rf outputs/ mlruns/ .pytest_cache/ .ruff_cache/ .coverage htmlcov/

# ---- Cloud GPU workflow (git-based) ----------------------------------------
# Set REMOTE and REMOTE_DIR in your shell or .env file, e.g.:
#   export REMOTE=user@cloudbox
#   export REMOTE_DIR=/workspace/<project>

cloud-setup: ## ssh remote, pull latest, install CUDA env
	@test -n "$$REMOTE" || (echo "REMOTE not set" && exit 1)
	@test -n "$$REMOTE_DIR" || (echo "REMOTE_DIR not set" && exit 1)
	ssh "$$REMOTE" 'cd "$$REMOTE_DIR" && git pull && make install-cuda'

cloud-train: ## launch a training run on the remote (cloud trainer profile)
	@test -n "$$REMOTE" || (echo "REMOTE not set" && exit 1)
	@test -n "$$REMOTE_DIR" || (echo "REMOTE_DIR not set" && exit 1)
	ssh "$$REMOTE" 'cd "$$REMOTE_DIR" && make train OVERRIDES="trainer=cloud"'

pull-results: ## rsync mlruns/ from remote into local mlruns/
	@test -n "$$REMOTE" || (echo "REMOTE not set" && exit 1)
	@test -n "$$REMOTE_DIR" || (echo "REMOTE_DIR not set" && exit 1)
	rsync -av --progress "$$REMOTE":"$$REMOTE_DIR/mlruns/" ./mlruns/
