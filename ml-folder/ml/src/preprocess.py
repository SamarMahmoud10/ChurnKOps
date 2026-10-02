"""Feature preprocessing.

Turns a cleaned DataFrame (from `data.py`) into model-ready X/y splits and
a fitted `sklearn` `ColumnTransformer`. The transformer is deliberately
NOT persisted on its own — `train.py` bundles it together with the
classifier into a single `sklearn.Pipeline` so exactly one artifact
(`ml/models/churn_model.pkl`) has to be loaded at inference time. That
keeps `predicate.py` and any future API wrapper trivially simple.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src import config

# Columns dropped before modeling: an opaque customer identifier carries
# no predictive signal and would only encourage the model to memorize IDs.
DROP_COLUMNS = [config.ID_COLUMN]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def get_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Intersect the expected feature lists with what's actually present.

    Makes the pipeline resilient to minor schema drift (e.g. a column
    renamed or temporarily missing) instead of hard-crashing on `KeyError`.
    """
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the cleaned DataFrame into X (features) and y (target)."""
    target_col = config.TARGET_COLUMN
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns] + [target_col]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def build_preprocessing_pipeline(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build the `ColumnTransformer`: scale numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split using project-wide config for size/seed."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def prepare_train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """End-to-end convenience: clean DataFrame in, ready-to-fit split + transformer out."""
    X, y = split_features_target(df)
    numeric_features, categorical_features = get_feature_columns(X)
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    return X_train, X_test, y_train, y_test, preprocessor
