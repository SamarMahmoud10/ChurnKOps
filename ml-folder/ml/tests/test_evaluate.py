from __future__ import annotations

import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src.evaluate import evaluate_model, load_metrics, save_metrics
from ml.src.preprocess import build_preprocessing_pipeline, get_feature_columns, split_features_target


def _fit_toy_pipeline(clean_sample_df):
    X, y = split_features_target(clean_sample_df)
    numeric, categorical = get_feature_columns(X)
    preprocessor = build_preprocessing_pipeline(numeric, categorical)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])
    pipeline.fit(X, y)
    return pipeline, X, y


def test_evaluate_model_returns_expected_keys(clean_sample_df):
    pipeline, X, y = _fit_toy_pipeline(clean_sample_df)

    metrics = evaluate_model(pipeline, X, y)

    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix", "n_test_samples"]:
        assert key in metrics


def test_evaluate_model_metrics_are_in_valid_range(clean_sample_df):
    pipeline, X, y = _fit_toy_pipeline(clean_sample_df)

    metrics = evaluate_model(pipeline, X, y)

    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_model_confusion_matrix_shape(clean_sample_df):
    pipeline, X, y = _fit_toy_pipeline(clean_sample_df)

    metrics = evaluate_model(pipeline, X, y)

    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)


def test_save_and_load_metrics_roundtrip(tmp_path):
    metrics = {"accuracy": 0.9, "confusion_matrix": [[10, 1], [2, 8]], "roc_auc": None}
    path = tmp_path / "metrics.json"

    save_metrics(metrics, path)
    loaded = load_metrics(path)

    assert loaded == metrics
    # sanity: it's really plain JSON on disk
    with open(path) as f:
        json.load(f)
