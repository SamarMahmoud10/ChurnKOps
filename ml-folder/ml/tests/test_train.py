from __future__ import annotations

import json

import pytest
from sklearn.pipeline import Pipeline

from ml.src import config, train


@pytest.fixture
def patched_paths(tmp_path, clean_sample_df, monkeypatch):
    """Point config at a temp raw CSV + temp model/metrics paths.

    Keeps these tests from touching the real dataset or overwriting the
    real ml/models/ artifacts on disk.
    """
    raw_csv = tmp_path / "raw.csv"
    clean_sample_df.to_csv(raw_csv, index=False)

    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    monkeypatch.setattr(config, "RAW_DATA_PATH", raw_csv)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "METRICS_PATH", metrics_path)
    return {"raw_csv": raw_csv, "model_path": model_path, "metrics_path": metrics_path}


def test_build_model_defaults_to_logistic_regression():
    name, model = train.build_model(None)
    assert name == "logistic_regression"


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError):
        train.build_model("not_a_real_model")


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_train_pipeline_produces_fitted_pipeline_and_metrics(patched_paths, model_type):
    pipeline, metrics = train.train_pipeline(model_type=model_type)

    assert isinstance(pipeline, Pipeline)
    assert metrics["model_type"] == model_type
    assert "accuracy" in metrics


def test_save_artifacts_writes_model_and_metrics_files(patched_paths):
    pipeline, metrics = train.train_pipeline()
    train.save_artifacts(pipeline, metrics)

    assert patched_paths["model_path"].exists()
    assert patched_paths["metrics_path"].exists()

    with open(patched_paths["metrics_path"]) as f:
        saved = json.load(f)
    assert saved["model_type"] == metrics["model_type"]
