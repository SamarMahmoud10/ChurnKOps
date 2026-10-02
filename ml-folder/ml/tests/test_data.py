from __future__ import annotations

import pandas as pd
import pytest

from ml.src import config
from ml.src.data import clean_data, load_raw_data


def test_load_raw_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_raw_data_reads_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    df = load_raw_data(csv_path)

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_clean_data_coerces_blank_total_charges_and_drops_row(raw_sample_df):
    cleaned = clean_data(raw_sample_df)

    # The row with a blank TotalCharges (customer 0004-DDD) must be gone.
    assert "0004-DDD" not in cleaned[config.ID_COLUMN].values


def test_clean_data_drops_duplicate_ids(raw_sample_df):
    cleaned = clean_data(raw_sample_df)

    assert cleaned[config.ID_COLUMN].is_unique


def test_clean_data_maps_target_to_binary(raw_sample_df):
    cleaned = clean_data(raw_sample_df)

    assert set(cleaned[config.TARGET_COLUMN].unique()) <= {0, 1}
    assert pd.api.types.is_numeric_dtype(cleaned[config.TARGET_COLUMN])


def test_clean_data_rejects_unexpected_target_values(raw_sample_df):
    bad_df = raw_sample_df.copy()
    bad_df.loc[0, config.TARGET_COLUMN] = "Maybe"

    with pytest.raises(ValueError):
        clean_data(bad_df)


def test_clean_data_is_idempotent_on_already_clean_frame(clean_sample_df):
    # clean_data should not blow up or change row count on data that's
    # already numeric/clean (Churn already 0/1 here).
    cleaned = clean_data(clean_sample_df)
    assert len(cleaned) == len(clean_sample_df)
