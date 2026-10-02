"""Data loading and cleaning.

Responsible ONLY for getting the raw Telco churn CSV into a clean,
well-typed pandas DataFrame. No feature engineering / encoding happens
here — that belongs to `preprocess.py`. Keeping this boundary means the
loader can be reused for training, evaluation, and (later) batch
inference without dragging in a fitted transformer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.src import config

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw churn CSV from disk.

    Parameters
    ----------
    path:
        Optional override. Defaults to `config.RAW_DATA_PATH`, which itself
        reads the `RAW_DATA_PATH` env var so this works unchanged inside a
        container that mounts the dataset somewhere else.
    """
    csv_path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{csv_path}'. Set the RAW_DATA_PATH "
            "environment variable or place the CSV at the default location."
        )
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the Telco churn dataset.

    - `TotalCharges` is shipped as a string and contains blank entries for
      customers with `tenure == 0`; these are coerced to NaN then dropped
      (a brand-new customer has no meaningful total-charges signal yet).
    - Duplicate `customerID` rows (if any) are dropped, keeping the first.
    - The target column is mapped from "Yes"/"No" to 1/0.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    id_col = config.ID_COLUMN
    if id_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first")
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d duplicate rows on '%s'", dropped, id_col)

    target_col = config.TARGET_COLUMN
    if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
        # Checking dtype rather than `dtype == object` since pandas' string
        # backend (e.g. the pandas 3.x default) may report a StringDtype
        # instead of plain `object` for text columns.
        mapping = {"Yes": 1, "No": 0}
        unmapped = set(df[target_col].dropna().unique()) - set(mapping)
        if unmapped:
            raise ValueError(
                f"Unexpected values in target column '{target_col}': {unmapped}"
            )
        df[target_col] = df[target_col].map(mapping).astype(int)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows containing missing values after cleaning", dropped)

    df = df.reset_index(drop=True)
    return df


def load_clean_data(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    return clean_data(load_raw_data(path))
