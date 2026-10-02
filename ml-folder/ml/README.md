# ChurnKops — ML Core (`ml/`)

Owned by the ML engineer. Nothing in this folder starts a server, builds a
Docker image, or touches CI — it's pure, importable Python that the DevOps
layer (`Dockerfile`, `docker-compose.yml`, `.github/workflows/`) wraps.

## Layout

```
ml/
├── src/
│   ├── config.py      # env-var-backed paths & constants (single source of truth)
│   ├── data.py         # load + clean the raw CSV
│   ├── preprocess.py   # feature lists, ColumnTransformer, train/test split
│   ├── train.py        # model definition + training entrypoint (CLI-runnable)
│   ├── evaluate.py      # metrics computation + JSON persistence
│   └── predicate.py     # inference: load model, predict() on raw records
├── tests/               # pytest suite, one file per src module
├── models/               # trained artifacts land here (git-ignored, built by train.py)
└── requirements.txt
```

## What each file does

- **`config.py`** — Every path and tunable (`RAW_DATA_PATH`, `MODEL_PATH`,
  `METRICS_PATH`, `RANDOM_STATE`, `TEST_SIZE`, `MODEL_TYPE` via `train.py`,
  `TARGET_COLUMN`, `ID_COLUMN`) is read from an environment variable with a
  sane default. This is the seam DevOps uses to override behaviour from a
  container (`docker run -e MODEL_PATH=/data/model.pkl ...`) without editing
  any Python.
- **`data.py`** — `load_raw_data()` reads the CSV; `clean_data()` fixes the
  known data-quality issues (blank `TotalCharges` for brand-new customers,
  duplicate `customerID`s, `Yes`/`No` target mapped to `1`/`0`). No feature
  encoding happens here on purpose.
- **`preprocess.py`** — Declares which columns are numeric vs categorical,
  builds the `ColumnTransformer` (StandardScaler + OneHotEncoder), and
  performs the stratified train/test split. Returns an *unfitted*
  transformer — it gets fitted only as part of the full pipeline in
  `train.py`, so train/test data never leaks into the scaler/encoder.
- **`train.py`** — Chooses a model via `MODEL_TYPE` (`logistic_regression`
  default, or `random_forest`), bundles `preprocessor + classifier` into one
  `sklearn.Pipeline`, fits it, evaluates it, and saves exactly one artifact
  (`ml/models/churn_model.pkl`) plus a metrics JSON. Runnable standalone:
  `python -m ml.src.train [--model-type random_forest]`.
- **`evaluate.py`** — Pure scoring functions (accuracy, precision, recall,
  F1, ROC-AUC, confusion matrix) plus JSON save/load helpers. Reusable to
  re-score a saved model later without retraining.
- **`predicate.py`** — The only file an API layer needs to import. Exposes
  `load_model()` (cached) and `predict(record_or_records)` which accepts a
  dict, list of dicts, or DataFrame of raw (unencoded) feature values and
  returns `{"churn_prediction": 0|1, "churn_probability": float}` per row.
  Deliberately has no HTTP/route code — that belongs to whatever framework
  DevOps wraps this in.

## Running it standalone

```bash
pip install -r ml/requirements.txt
python -m ml.src.train                 # trains + saves ml/models/churn_model.pkl + metrics.json
python -m pytest ml/tests/             # runs the ML test suite (32 tests)
```

```python
from ml.src.predicate import predict
predict({"gender": "Female", "SeniorCitizen": 0, ..., "TotalCharges": 29.85})
```

## Env vars a container/CI job may set

| Variable | Default | Purpose |
|---|---|---|
| `RAW_DATA_PATH` | `data/raw/Telco-Customer-Churn.csv` | where the training CSV lives |
| `MODEL_PATH` | `ml/models/churn_model.pkl` | where the trained pipeline is saved/loaded |
| `METRICS_PATH` | `ml/models/metrics.json` | where evaluation metrics are saved |
| `MODEL_TYPE` | `logistic_regression` | `logistic_regression` \| `random_forest` |
| `RANDOM_STATE` | `42` | reproducibility seed |
| `TEST_SIZE` | `0.2` | train/test split fraction |
