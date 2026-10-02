"""Inference (prediction) logic.

Deliberately a thin, dependency-light module: it only knows how to load
the single persisted `Pipeline` artifact from `MODEL_PATH` and turn raw
customer records into churn predictions. It does NOT open any network
port or define HTTP routes — wrapping this in a web framework (FastAPI/
Flask) and exposing it inside the container is DevOps' job. Anything in
this file can be imported directly by that layer, e.g.:

    from ml.src.predicate import predict

    @app.post("/predict")
    def predict_endpoint(payload: dict):
        return predict(payload)
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.src import config
from ml.src.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EXPECTED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class ModelNotFoundError(RuntimeError):
    """Raised when MODEL_PATH does not point at a trained artifact yet."""


@functools.lru_cache(maxsize=1)
def load_model(model_path: str | Path | None = None):
    """Load (and cache) the trained pipeline from disk.

    Cached with `lru_cache` so repeated predictions in the same process
    (e.g. many requests handled by one API worker) don't re-read the
    artifact from disk each time. Call `load_model.cache_clear()` if the
    artifact is retrained/replaced while the process is running.
    """
    path = Path(model_path) if model_path is not None else config.MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{path}'. Run `python -m ml.src.train` first."
        )
    return joblib.load(path)


def _to_dataframe(records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        raise TypeError(
            f"Unsupported input type {type(records)}; expected dict, list[dict], or DataFrame"
        )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Extra columns (e.g. a stray customerID passed through by a caller)
    # are silently ignored rather than crashing the request.
    return df[EXPECTED_COLUMNS]


def predict(
    records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Predict churn for one or more customer records.

    Accepts a single dict, a list of dicts, or a DataFrame of raw
    (unencoded) feature values matching the Telco schema. Returns a list
    of `{"churn_prediction": 0|1, "churn_probability": float}` dicts, one
    per input row, in the same order.
    """
    model = load_model(model_path)
    X = _to_dataframe(records)

    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = [None] * len(predictions)

    return [
        {"churn_prediction": int(pred), "churn_probability": (float(prob) if prob is not None else None)}
        for pred, prob in zip(predictions, probabilities)
    ]
