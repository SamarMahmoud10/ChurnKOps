from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src.predicate import EXPECTED_COLUMNS, ModelNotFoundError, load_model, predict
from ml.src.preprocess import build_preprocessing_pipeline, get_feature_columns, split_features_target


@pytest.fixture
def trained_model_path(tmp_path, clean_sample_df):
    X, y = split_features_target(clean_sample_df)
    numeric, categorical = get_feature_columns(X)
    preprocessor = build_preprocessing_pipeline(numeric, categorical)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])
    pipeline.fit(X, y)

    path = tmp_path / "model.pkl"
    joblib.dump(pipeline, path)

    load_model.cache_clear()
    yield path
    load_model.cache_clear()


@pytest.fixture
def sample_record(clean_sample_df) -> dict:
    row = clean_sample_df.iloc[0]
    return {col: row[col] for col in EXPECTED_COLUMNS}


def test_load_model_missing_path_raises(tmp_path):
    load_model.cache_clear()
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "nope.pkl")


def test_load_model_loads_pipeline(trained_model_path):
    model = load_model(trained_model_path)
    assert isinstance(model, Pipeline)


def test_predict_single_dict_returns_one_result(trained_model_path, sample_record):
    result = predict(sample_record, model_path=trained_model_path)

    assert len(result) == 1
    assert result[0]["churn_prediction"] in (0, 1)
    assert 0.0 <= result[0]["churn_probability"] <= 1.0


def test_predict_list_of_dicts_returns_matching_count(trained_model_path, sample_record):
    result = predict([sample_record, sample_record, sample_record], model_path=trained_model_path)
    assert len(result) == 3


def test_predict_dataframe_input(trained_model_path, sample_record):
    df = pd.DataFrame([sample_record, sample_record])
    result = predict(df, model_path=trained_model_path)
    assert len(result) == 2


def test_predict_ignores_extra_columns(trained_model_path, sample_record):
    record_with_extra = dict(sample_record)
    record_with_extra["customerID"] = "SHOULD-BE-IGNORED"
    record_with_extra["some_junk_field"] = 123

    result = predict(record_with_extra, model_path=trained_model_path)
    assert len(result) == 1


def test_predict_missing_required_column_raises(trained_model_path, sample_record):
    incomplete = dict(sample_record)
    del incomplete["tenure"]

    with pytest.raises(ValueError):
        predict(incomplete, model_path=trained_model_path)
