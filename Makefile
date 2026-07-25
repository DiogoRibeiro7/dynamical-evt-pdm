.PHONY: install format lint typecheck test check fetch-data prepare-data simulate simulation-study experiment-matrix end-to-end-smoke paper-assets paper-assets-verify paper supplement paper-check

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

simulation-study:
	poetry run dyn-evt run-simulation-study --smoke --output artifacts/simulation_study_smoke.parquet

experiment-matrix:
	poetry run dyn-evt run-experiment-matrix --output-root artifacts/experiment_matrix

end-to-end-smoke:
	poetry run dyn-evt simulate --output data/processed/master_smoke.csv --n-steps 1000 --seed 42
	poetry run dyn-evt analyse-series --input data/processed/master_smoke.csv --output artifacts/master_smoke_summary.json --value-column observable --regime-column regime --run-length 5
	poetry run dyn-evt analyse-univariate --input data/processed/master_smoke.csv --value-column observable --output artifacts/master_smoke_univariate.json --run-length 5 --min-exceedances 3
	poetry run dyn-evt analyse-regimes --input data/processed/master_smoke.csv --value-column observable --method rules --current-column current --pressure-column pressure --output artifacts/master_smoke_regimes.json
	poetry run dyn-evt analyse-dangerous-region --input data/processed/master_smoke.csv --state-columns pressure,current,temperature --target-column is_fault --output artifacts/master_smoke_dangerous.parquet --report-output artifacts/master_smoke_dangerous.json --horizons 10,20
	poetry run dyn-evt analyse-multivariate-extremes --input data/processed/master_smoke.csv --component-columns current,temperature,pressure --regime-column regime --split-column split --output artifacts/master_smoke_multivariate.json --max-lag 2 --n-null 3 --min-regime-samples 20
	poetry run dyn-evt run-baselines --input data/processed/master_smoke.csv --feature-columns pressure,current,temperature --timestamp-column time --regime-column regime --split-column split --target-column is_fault --window-size 2 --horizon 20 --isolation-estimators 20 --output artifacts/master_smoke_baselines.parquet --metadata-output artifacts/master_smoke_baselines.json
	poetry run dyn-evt run-simulation-study --smoke --output artifacts/master_smoke_simulation_study.parquet
	poetry run dyn-evt build-paper-assets --input data/processed/master_smoke.csv --output-root reports/master_smoke --simulation-study-path artifacts/master_smoke_simulation_study.parquet

paper-assets: simulate simulation-study
	poetry run dyn-evt build-paper-assets \
		--input data/processed/synthetic_cyclic.csv \
		--output-root reports/paper \
		--simulation-study-path artifacts/simulation_study_smoke.parquet
	poetry run dyn-evt verify-paper-assets --output-root reports/paper

paper-assets-verify:
	poetry run dyn-evt verify-paper-assets --output-root reports/paper

paper: paper-assets
	$(MAKE) -C paper paper

supplement: paper-assets
	$(MAKE) -C paper supplement

paper-check: paper supplement
	poetry run dyn-evt check-paper --paper-root paper --asset-root reports/paper
