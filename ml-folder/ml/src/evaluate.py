"""Model evaluation.

Pure functions that take a fitted model/pipeline and a held-out X/y and
produce a plain-dict metrics report. Kept separate from `train.py` so the
same evaluation logic can be re-run later (e.g. re-scoring a saved model
against a fresh batch of labeled data) without re-training anything.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score `model` on `X_test`/`y_test` and return a JSON-serializable metrics dict."""
    y_pred = model.predict(X_test)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": int(len(y_test)),
    }

    # ROC-AUC needs predicted probabilities; not every estimator exposes
    # predict_proba (e.g. some SVM configs), so degrade gracefully.
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Persist a metrics dict as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)


def load_metrics(path: str | Path) -> dict:
    """Read back a previously saved metrics JSON file."""
    with open(Path(path)) as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
