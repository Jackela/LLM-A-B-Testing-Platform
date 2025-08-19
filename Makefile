# LLM A/B Testing Platform - Enhanced Makefile
# ================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Project configuration
PROJECT_NAME := llm-ab-testing-platform
PYTHON_VERSION := 3.11
POETRY_VERSION := 1.8.3

# Environment variables
export DATABASE_URL ?= postgresql://postgres:postgres@localhost:5432/test_db
export REDIS_URL ?= redis://localhost:6379/0
export TESTING ?= true

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)🚀 $(PROJECT_NAME) - Development Commands$(RESET)"
	@echo "$(CYAN)================================================$(RESET)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo
	@echo "$(YELLOW)📋 Quick Start:$(RESET)"
	@echo "  make setup     # Set up the development environment"
	@echo "  make test      # Run all tests"
	@echo "  make dev       # Start development server"
	@echo

# ================================================
# Environment Setup
# ================================================

.PHONY: check-system
check-system: ## Check system requirements
	@echo "$(BLUE)🔍 Checking system requirements...$(RESET)"
	@python3 --version 2>/dev/null || (echo "$(RED)❌ Python 3 not found$(RESET)" && exit 1)
	@poetry --version 2>/dev/null || (echo "$(RED)❌ Poetry not found. Install from https://python-poetry.org/$(RESET)" && exit 1)
	@docker --version 2>/dev/null || echo "$(YELLOW)⚠️ Docker not found. Some features may not work.$(RESET)"
	@echo "$(GREEN)✅ System requirements check completed$(RESET)"

install: ## Install dependencies
	poetry install

install-dev: ## Install development dependencies
	poetry install --with dev

# Core test targets
test: lint test-unit test-integration test-e2e ## Run all tests
	@echo "All tests completed!"

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	poetry run pytest tests/unit/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/unit \
		--cov-report=xml:coverage-unit.xml \
		-v \
		-m "not slow and not external" \
		--tb=short

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	poetry run pytest tests/integration/ \
		--cov=src \
		--cov-append \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/integration \
		--cov-report=xml:coverage-integration.xml \
		-v \
		-m "integration and not slow and not external" \
		--tb=short

test-e2e: ## Run end-to-end tests
	@echo "Running end-to-end tests..."
	poetry run pytest tests/e2e/ \
		--cov=src \
		--cov-append \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/e2e \
		--cov-report=xml:coverage-e2e.xml \
		-v \
		-m "e2e and not slow and not external" \
		--tb=short \
		--timeout=300

test-performance: ## Run performance tests
	@echo "Running performance tests..."
	poetry run pytest tests/performance/ \
		-v \
		-m "performance" \
		--tb=short \
		--timeout=600 \
		--performance

test-all: test test-performance ## Run all tests including performance
	@echo "All tests including performance completed!"

# Test variants
test-fast: ## Run fast tests only
	@echo "Running fast tests only..."
	poetry run pytest \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html \
		-v \
		-m "not slow and not external and not performance" \
		--tb=short \
		--maxfail=10

test-slow: ## Run slow tests only
	@echo "Running slow tests only..."
	poetry run pytest \
		--cov=src \
		--cov-append \
		--cov-report=term-missing \
		-v \
		-m "slow" \
		--tb=short \
		--timeout=1800

test-external: ## Run external service tests
	@echo "Running external service tests..."
	poetry run pytest \
		--cov=src \
		--cov-append \
		--cov-report=term-missing \
		-v \
		-m "external" \
		--tb=short \
		--runexternal

test-parallel: ## Run tests in parallel
	@echo "Running tests in parallel..."
	poetry run pytest \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html \
		-v \
		-n auto \
		--tb=short \
		-m "not slow and not external"

test-verbose: ## Run tests with verbose output
	@echo "Running tests with verbose output..."
	poetry run pytest \
		--cov=src \
		--cov-report=term-missing \
		-vv \
		-s \
		--tb=long \
		--capture=no

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	poetry run ptw \
		--runner "pytest --cov=src --cov-report=term-missing -v --tb=short -m 'not slow and not external'" \
		-- tests/

# Code quality targets
lint: lint-black lint-isort lint-flake8 lint-mypy ## Run all linting checks

lint-black: ## Check code formatting with black
	@echo "Checking code formatting with black..."
	poetry run black --check --diff src tests

lint-isort: ## Check import sorting with isort
	@echo "Checking import sorting with isort..."
	poetry run isort --check-only --diff src tests

lint-flake8: ## Run flake8 linting
	@echo "Running flake8 linting..."
	poetry run flake8 src tests

lint-mypy: ## Run mypy type checking
	@echo "Running mypy type checking..."
	poetry run mypy src

format: ## Format code with black and isort
	@echo "Formatting code with black and isort..."
	poetry run black src tests
	poetry run isort src tests

type-check: ## Run comprehensive type checking
	@echo "Running comprehensive type checking..."
	poetry run mypy src --strict --show-error-codes --show-error-context

security-scan: ## Run security scanning
	@echo "Running security scans..."
	@echo "Running bandit security scan..."
	poetry run bandit -r src/ -f json -o bandit-report.json || true
	@echo "Running safety dependency scan..."
	poetry run safety check --json --output safety-report.json || true
	@echo "Security scan reports generated: bandit-report.json, safety-report.json"

code-quality: ## Run code quality analysis
	@echo "Running code quality analysis..."
	@echo "Analyzing code complexity..."
	poetry run radon cc src/ --min B --show-complexity || true
	@echo "Analyzing maintainability index..."
	poetry run radon mi src/ --min B || true
	@echo "Running pylint analysis..."
	poetry run pylint src/ --output-format=text --reports=y || true

# Coverage targets
coverage: ## Generate coverage report
	@echo "Generating coverage report..."
	poetry run pytest \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--cov-fail-under=85 \
		-m "not slow and not external and not performance"

coverage-html: ## Generate HTML coverage report
	@echo "Generating HTML coverage report..."
	poetry run pytest \
		--cov=src \
		--cov-report=html:htmlcov \
		-m "not slow and not external and not performance"
	@echo "HTML coverage report generated in htmlcov/"

# Database targets
setup-test-db: ## Setup test database
	@echo "Setting up test database..."
	@if [ -z "$(DATABASE_URL)" ]; then \
		echo "DATABASE_URL not set, using default..."; \
		export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/test_db"; \
	fi
	poetry run alembic upgrade head

# Docker targets
docker-test-up: ## Start test services with Docker Compose
	@echo "Starting test services with Docker Compose..."
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services to be ready..."
	sleep 10

docker-test-down: ## Stop test services
	@echo "Stopping test services..."
	docker-compose -f docker-compose.test.yml down -v

docker-test: docker-test-up ## Run tests in Docker environment
	@echo "Running tests in Docker environment..."
	sleep 5  # Additional wait for services
	$(MAKE) test DATABASE_URL=postgresql://postgres:postgres@localhost:5433/test_db
	$(MAKE) docker-test-down

# Development workflow helpers
dev-setup: install-dev setup-test-db ## Setup development environment
	@echo "Development environment setup complete!"

dev-test: lint test-fast ## Run development test cycle
	@echo "Development test cycle complete!"

pre-commit: format lint test-fast ## Run pre-commit checks
	@echo "Pre-commit checks completed!"

# Quick aliases
unit: test-unit ## Alias for test-unit
integration: test-integration ## Alias for test-integration  
e2e: test-e2e ## Alias for test-e2e
perf: test-performance ## Alias for test-performance
quick: test-fast ## Alias for test-fast
watch: test-watch ## Alias for test-watch
fmt: format ## Alias for format
check: lint ## Alias for lint

quality: lint type-check ## Run all quality checks

dev: ## Start development servers
	docker-compose -f docker-compose.dev.yml up -d
	poetry run uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000 &
	poetry run streamlit run src/presentation/dashboard/main.py --server.port 8501

dev-api: ## Start API server only
	poetry run uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000

dev-dashboard: ## Start dashboard only
	poetry run streamlit run src/presentation/dashboard/main.py --server.port 8501

celery: ## Start Celery worker
	poetry run celery -A src.infrastructure.tasks.celery_app worker --loglevel=info

celery-beat: ## Start Celery beat scheduler
	poetry run celery -A src.infrastructure.tasks.celery_app beat --loglevel=info

clean: ## Clean cache and temporary files
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

migrate: ## Run database migrations
	poetry run alembic upgrade head

migrate-auto: ## Generate automatic migration
	poetry run alembic revision --autogenerate -m "Auto migration"

seed: ## Seed database with test data
	poetry run python scripts/seed_database.py

docs: ## Generate documentation
	poetry run mkdocs serve