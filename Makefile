.PHONY: install format lint typecheck test check fetch-data prepare-data simulate

install:
	poetry install --with dev

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest

check: lint typecheck test

fetch-data:
	poetry run dyn-evt fetch-data --dataset all

prepare-data:
	poetry run dyn-evt prepare-metropt
	poetry run dyn-evt prepare-metropt2
	poetry run dyn-evt prepare-scania

simulate:
	poetry run dyn-evt simulate --output data/processed/synthetic_cyclic.csv --n-steps 20000 --seed 42
