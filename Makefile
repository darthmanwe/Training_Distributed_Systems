.PHONY: install lint typecheck test test-integration format run run-churn run-cartpole clean requirements help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project with dev dependencies
	uv pip install -e ".[dev]"

requirements:  ## Generate requirements.txt from pyproject.toml
	uv pip compile pyproject.toml -o requirements.txt

lint:  ## Run ruff linter
	ruff check src/ tests/ scripts/

typecheck:  ## Run mypy type checker
	mypy src/

test:  ## Run unit tests
	pytest -q -m "not integration and not slow"

test-integration:  ## Run integration tests
	pytest -q -m integration

test-all:  ## Run all tests
	pytest -v

format:  ## Format code with ruff
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

run:  ## Run training with base config
	python scripts/run_local.py --config configs/base.yaml

run-churn:  ## Run training with worker churn enabled
	python scripts/run_local.py --config configs/churn.yaml

run-cartpole:  ## Run PPO validation on CartPole
	python scripts/run_local.py --config configs/cartpole.yaml

benchmark:  ## Run benchmarks
	python scripts/benchmark.py

clean:  ## Remove build artifacts and caches
	rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up:  ## Start observability stack (Prometheus + Grafana)
	docker compose -f docker/docker-compose.yaml up -d

docker-down:  ## Stop observability stack
	docker compose -f docker/docker-compose.yaml down
