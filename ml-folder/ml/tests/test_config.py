from __future__ import annotations

import importlib

from ml.src import config


def test_default_paths_are_under_repo_layout():
    assert config.MODEL_PATH.name == "churn_model.pkl"
    assert config.METRICS_PATH.name == "metrics.json"
    assert config.RAW_DATA_PATH.name == "Telco-Customer-Churn.csv"


def test_env_var_overrides_are_respected(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "custom_model.pkl"))
    monkeypatch.setenv("RANDOM_STATE", "7")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.MODEL_PATH == tmp_path / "custom_model.pkl"
        assert reloaded.RANDOM_STATE == 7
    finally:
        # restore module state for any tests that import config afterwards
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("RANDOM_STATE", raising=False)
        importlib.reload(config)
