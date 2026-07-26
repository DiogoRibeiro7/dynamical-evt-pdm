# Data acquisition and governance

Large source files are excluded from Git.

## MetroPT

Record: <https://zenodo.org/records/6854240>

The accompanying Scientific Data article describes 1 Hz telemetry from an Air Production Unit, approximately 10.98 million observations, 20 variables and three catastrophic failures during six months. The company requested failure detection at least two hours before the train became non-operational.

Fetch the downloaded file with:

```bash
poetry run dyn-evt fetch-data --dataset metropt
```

The released file is stored under:

```text
data/raw/metropt/dataset_train.csv
```

Do not rename columns invisibly. Record source-to-canonical mappings in the dataset YAML and generated data dictionary.

## MetroPT2

Record: <https://zenodo.org/records/7766691>

The record exposes two large CSV files. The `MetroPT2.csv` file is reported with MD5 `056e04fd1874eaef4697d6b6657502c3` and two documented failure intervals.

Fetch the downloaded file with:

```bash
poetry run dyn-evt fetch-data --dataset metropt2
```

The released file is stored under:

```text
data/raw/metropt2/MetroPT2.csv
```

## SCANIA Component X

Article: <https://www.nature.com/articles/s41597-025-04802-6>

The dataset contains fleet-level operational readouts, repair records and truck specifications. It includes irregular entity histories and cumulative or histogram-like features, so the MetroPT pointwise pipeline must not be reused without an entity-time adapter.

Fetch the downloaded files with:

```bash
poetry run dyn-evt fetch-data --dataset scania_component_x
```

Source tables are stored under:

```text
data/raw/scania_component_x/
```

## Hydraulic Systems

Record: <https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems>

The dataset contains experimentally obtained hydraulic test-rig cycles. Each cycle repeats a constant-load operating profile and records multi-rate sensor traces with component-condition targets in `profile.txt`.

Fetch the downloaded archive with:

```bash
poetry run dyn-evt fetch-data --dataset hydraulic_systems
```

The released archive is stored under:

```text
data/raw/hydraulic_systems/condition+monitoring+of+hydraulic+systems.zip
```

## SECOM

Record: <https://archive.ics.uci.edu/dataset/179/secom>

The dataset contains semiconductor manufacturing process measurements and pass/fail yield labels for individual production examples. It broadens the study to high-dimensional tabular process monitoring with missing values and imbalanced failures.

Fetch the downloaded archive with:

```bash
poetry run dyn-evt fetch-data --dataset secom
```

The released archive is stored under:

```text
data/raw/secom/secom.zip
```

## Integrity rules

1. Keep raw files immutable.
2. Store checksums in dataset manifests.
3. Write processed data as partitioned Parquet.
4. Fit scaling, thresholds, regimes and dangerous-state prototypes on training data only.
5. Preserve timestamps and entity identifiers.
6. Save all exclusion windows and label transformations as machine-readable metadata.

## Preparation

After fetching raw files, build processed Parquet parts with:

```bash
poetry run dyn-evt prepare-metropt
poetry run dyn-evt prepare-metropt2
poetry run dyn-evt prepare-scania
poetry run dyn-evt prepare-hydraulic-systems
poetry run dyn-evt prepare-secom
```

Each command writes `part-*.parquet` files and a `manifest.json` under `data/processed/<dataset>/`. The manifests record raw hashes, schema statistics, source-to-canonical mappings, timestamp checks, failure-label provenance, Scania vehicle-split leakage checks, hydraulic cycle summaries and SECOM missingness checks.
