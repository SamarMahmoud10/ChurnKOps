from __future__ import annotations

import pandas as pd
import pytest

from ml.src import config
from ml.src.preprocess import (
    build_preprocessing_pipeline,
    get_feature_columns,
    prepare_train_test_split,
    split_features_target,
    train_test_split_data,
)


def test_split_features_target_drops_id_and_target(clean_sample_df):
    X, y = split_features_target(clean_sample_df)

    assert config.TARGET_COLUMN not in X.columns
    assert config.ID_COLUMN not in X.columns
    assert len(X) == len(y) == len(clean_sample_df)


def test_split_features_target_missing_target_raises(clean_sample_df):
    df = clean_sample_df.drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(KeyError):
        split_features_target(df)


def test_get_feature_columns_matches_present_columns(clean_sample_df):
    X, _ = split_features_target(clean_sample_df)
    numeric, categorical = get_feature_columns(X)

    assert "tenure" in numeric
    assert "MonthlyCharges" in numeric
    assert "Contract" in categorical
    # every returned column really exists in X
    assert set(numeric + categorical) <= set(X.columns)


def test_build_preprocessing_pipeline_fits_and_transforms(clean_sample_df):
    X, _ = split_features_target(clean_sample_df)
    numeric, categorical = get_feature_columns(X)
    preprocessor = build_preprocessing_pipeline(numeric, categorical)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    # more columns after one-hot encoding than raw numeric+categorical count
    assert transformed.shape[1] >= len(numeric) + len(categorical)


def test_build_preprocessing_pipeline_ignores_unknown_categories(clean_sample_df):
    X, _ = split_features_target(clean_sample_df)
    numeric, categorical = get_feature_columns(X)
    preprocessor = build_preprocessing_pipeline(numeric, categorical)
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["Contract"] = "Some Brand New Contract Type"

    # Should not raise thanks to handle_unknown="ignore" on the encoder.
    preprocessor.transform(unseen)


def test_train_test_split_data_respects_config_test_size(clean_sample_df):
    X, y = split_features_target(clean_sample_df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    total = len(X)
    expected_test = round(total * config.TEST_SIZE)
    assert abs(len(X_test) - expected_test) <= 1
    assert len(X_train) + len(X_test) == total


def test_prepare_train_test_split_end_to_end(clean_sample_df):
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(clean_sample_df)

    assert len(X_train) + len(X_test) == len(clean_sample_df)
    # preprocessor should be unfitted at this point (train.py fits it as
    # part of the full Pipeline) but must still be usable.
    preprocessor.fit(X_train)
    preprocessor.transform(X_test)
