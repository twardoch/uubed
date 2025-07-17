# Makefile for uubed project
# This file provides convenient commands for common development tasks

.PHONY: help test build clean install dev lint format type-check release version

# Default target
help:
	@echo "Available commands:"
	@echo "  test        - Run the test suite"
	@echo "  test-all    - Run tests with linting and formatting"
	@echo "  build       - Build the project"
	@echo "  clean       - Clean build artifacts"
	@echo "  install     - Install the project in development mode"
	@echo "  dev         - Set up development environment"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  type-check  - Run type checking"
	@echo "  release     - Create a release (dry run)"
	@echo "  version     - Show current version"

# Test commands
test:
	python scripts/test.py

test-all:
	python scripts/test.py --all

test-coverage:
	python scripts/test.py --coverage

# Build commands
build:
	python scripts/build.py

build-clean:
	python scripts/build.py --clean

build-release:
	python scripts/build.py --release --clean

# Development commands
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

install:
	pip install -e .

dev:
	pip install -e .
	pip install pytest pytest-cov ruff black mypy

# Code quality commands
lint:
	python scripts/test.py --lint

format:
	python scripts/test.py --format

type-check:
	python scripts/test.py --type-check

# Release commands
release:
	python scripts/release.py --dry-run

release-real:
	python scripts/release.py

# Utility commands
version:
	python scripts/get_version.py

check-deps:
	python scripts/build.py --check-deps

# CI simulation
ci: test-all build-clean