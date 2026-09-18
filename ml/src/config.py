"""Runtime configuration for the ML package.
Paths and environment switches are read from the process environment so a
later container or CI job can override them without changing code.
"""
from __future__ import annotations
import os
from pathlib import Path
ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv")
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", ML_ROOT / "models" / "churn_model.pkl"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ML_ROOT / "models" / "metrics.json"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Churn")
ID_COLUMN = os.getenv("ID_COLUMN", "customerID")
ENV = os.getenv("ENV", "development")