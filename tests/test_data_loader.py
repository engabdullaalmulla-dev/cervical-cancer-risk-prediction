"""Unit tests for the preprocessing pipeline in src/data_loader.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    CONTINUOUS_COLS,
    DEFAULT_CSV,
    HIGH_MISSING_COLS,
    TARGET,
    clean,
    impute,
    load_features_target,
    load_raw,
)

requires_data = pytest.mark.skipif(
    not DEFAULT_CSV.exists(),
    reason="Raw data missing - run `python src/download_data.py` first",
)


@requires_data
def test_raw_shape():
    df = load_raw()
    assert df.shape == (858, 36)
    assert TARGET in df.columns


@requires_data
def test_clean_drops_high_missing_columns_and_is_numeric():
    df_clean = clean(load_raw())
    assert df_clean.shape[1] == 34
    for col in HIGH_MISSING_COLS:
        assert col not in df_clean.columns
    assert all(pd.api.types.is_numeric_dtype(df_clean[c]) for c in df_clean.columns)


@requires_data
def test_clean_converts_placeholders_to_nan():
    df_clean = clean(load_raw())
    assert not (df_clean == "?").any().any()
    assert df_clean.isna().sum().sum() > 0  # missing values survive as NaN


@requires_data
def test_load_features_target_is_fully_imputed():
    X, y = load_features_target()
    assert X.shape == (858, 33)
    assert y.shape == (858,)
    assert X.isna().sum().sum() == 0
    assert TARGET not in X.columns


@requires_data
def test_target_class_balance_matches_published_dataset():
    _, y = load_features_target()
    counts = y.value_counts().to_dict()
    assert counts[0.0] == 803
    assert counts[1.0] == 55


def test_impute_fills_missing_values():
    frame = pd.DataFrame(
        {
            "Age": [20.0, np.nan, 40.0],
            "Smokes": [0.0, 1.0, np.nan],
        }
    )
    filled = impute(frame)
    assert filled.isna().sum().sum() == 0
    assert filled.loc[1, "Age"] == 30.0  # median of 20 and 40


def test_continuous_columns_are_disjoint_from_target():
    assert TARGET not in CONTINUOUS_COLS


def test_load_raw_raises_a_helpful_error_for_a_missing_file():
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(Path("does_not_exist.csv"))
