"""Model definition + training entrypoint.

Running this module end-to-end (`python -m ml.src.train`) will:
  1. load + clean the raw data
  2. split it into train/test
  3. fit a full `preprocessor -> classifier` sklearn Pipeline
  4. evaluate it on the held-out test set
  5. persist the fitted pipeline to `MODEL_PATH` and the metrics to
     `METRICS_PATH`

Everything reads its paths/knobs from `config.py` (env-var backed), so a
CI job or Docker build can override behaviour without touching this file.
"""
from __future__ import annotations

import argparse
import logging
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src import config, evaluate
from ml.src.data import load_clean_data
from ml.src.preprocess import prepare_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Supported model families, keyed by the MODEL_TYPE env var so a different
# baseline can be selected without editing code.
MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=config.RANDOM_STATE
    ),
}


def build_model(model_type: str | None = None):
    """Instantiate the classifier named by `model_type` (default: logistic_regression)."""
    model_type = model_type or os.getenv("MODEL_TYPE", "logistic_regression")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_TYPE '{model_type}'. Choose one of {list(MODEL_REGISTRY)}"
        )
    return model_type, MODEL_REGISTRY[model_type]()


def train_pipeline(model_type: str | None = None) -> tuple[Pipeline, dict]:
    """Train the full pipeline and return it along with its evaluation metrics."""
    df = load_clean_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df)

    resolved_model_type, model = build_model(model_type)
    logger.info(
        "Training model_type=%s on %d rows (test=%d)",
        resolved_model_type,
        len(X_train),
        len(X_test),
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)

    metrics = evaluate.evaluate_model(pipeline, X_test, y_test)
    metrics["model_type"] = resolved_model_type
    logger.info(
        "Evaluation metrics: %s",
        {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    )

    return pipeline, metrics


def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    logger.info("Saved trained pipeline to %s", config.MODEL_PATH)

    evaluate.save_metrics(metrics, config.METRICS_PATH)
    logger.info("Saved metrics to %s", config.METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY),
        default=None,
        help="Override MODEL_TYPE env var (default: logistic_regression).",
    )
    args = parser.parse_args()

    pipeline, metrics = train_pipeline(model_type=args.model_type)
    save_artifacts(pipeline, metrics)


if __name__ == "__main__":
    main()
