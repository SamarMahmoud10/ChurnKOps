"""Shared fixtures for the ML test suite.

Tests use a small, in-memory synthetic sample rather than the full
7000-row CSV so the suite stays fast and deterministic in CI, and a
tiny copy of the real raw-data quirks (blank TotalCharges, Yes/No
target) so `data.py`'s cleaning logic is genuinely exercised.
"""
from __future__ import annotations

import pandas as pd
import pytest


RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


@pytest.fixture
def raw_sample_df() -> pd.DataFrame:
    """A tiny DataFrame shaped like the raw Telco CSV, quirks included.

    Includes: a blank TotalCharges (new customer), a duplicate customerID,
    and both target classes, so cleaning logic has something to do.
    """
    rows = [
        ["0001-AAA", "Female", 0, "Yes", "No", 1, "No", "No phone service", "DSL",
         "No", "Yes", "No", "No", "No", "No", "Month-to-month", "Yes",
         "Electronic check", 29.85, "29.85", "No"],
        ["0002-BBB", "Male", 0, "No", "No", 34, "Yes", "No", "DSL",
         "Yes", "No", "Yes", "No", "No", "No", "One year", "No",
         "Mailed check", 56.95, "1889.5", "No"],
        ["0003-CCC", "Male", 1, "No", "No", 2, "Yes", "No", "Fiber optic",
         "No", "No", "No", "No", "No", "No", "Month-to-month", "Yes",
         "Electronic check", 70.35, "151.65", "Yes"],
        ["0004-DDD", "Female", 0, "Yes", "Yes", 0, "Yes", "No", "Fiber optic",
         "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes",
         "Electronic check", 90.05, " ", "Yes"],  # blank TotalCharges, tenure=0
        ["0002-BBB", "Male", 0, "No", "No", 34, "Yes", "No", "DSL",  # duplicate ID
         "Yes", "No", "Yes", "No", "No", "No", "One year", "No",
         "Mailed check", 56.95, "1889.5", "No"],
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def clean_sample_df(raw_sample_df: pd.DataFrame) -> pd.DataFrame:
    """A larger, already-clean sample used by preprocess/train/evaluate tests.

    Built programmatically (not via `clean_data`) so these tests don't
    silently depend on `data.py`'s behaviour; each module is tested in
    isolation.
    """
    import numpy as np

    rng = np.random.RandomState(0)
    n = 60
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.randint(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No"], n),
            "OnlineBackup": rng.choice(["Yes", "No"], n),
            "DeviceProtection": rng.choice(["Yes", "No"], n),
            "TechSupport": rng.choice(["Yes", "No"], n),
            "StreamingTV": rng.choice(["Yes", "No"], n),
            "StreamingMovies": rng.choice(["Yes", "No"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
            ),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(20, 8000, n).round(2),
            "Churn": rng.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )
