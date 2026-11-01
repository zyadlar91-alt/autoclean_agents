import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.profiling_tools import (
    check_missing,
    check_types,
    check_outliers,
    check_duplicates,
    check_cardinality,
    run_full_profile,
)

# DataFrame de test avec problemes controles
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "periode": ["2018-01", "2018-02", "2018-03", "2018-01"],  # doublon ligne 0 et 3
        "nb_train_prevu": [100, 200, 300, 100],
        "retard_moyen_depart": [5.0, np.nan, 999.0, 5.0],  # NaN + outlier + doublon
        "cause_externe": [0.1, 0.2, 0.3, 0.1],
        "type_col_str": ["N/A", "ok", "ok", "N/A"],  # valeurs string parasites
    })



def test_missing_detects_nan(sample_df):
    result = check_missing(sample_df, threshold=0.05)
    assert "retard_moyen_depart" in result["flagged_columns"]

def test_missing_no_flag_below_threshold(sample_df):
    # nb_train_prevu n'a pas de NaN
    result = check_missing(sample_df, threshold=0.05)
    assert "nb_train_prevu" not in result["flagged_columns"]

def test_missing_returns_dict(sample_df):
    result = check_missing(sample_df)
    assert isinstance(result, dict)
    assert "flagged_columns" in result



def test_types_flags_string_in_numeric(sample_df):
    result = check_types(sample_df)
    # type_col_str contient "N/A" — doit etre signale
    assert "type_col_str" in result["suspicious_columns"]

def test_types_returns_dict(sample_df):
    result = check_types(sample_df)
    assert isinstance(result, dict)
    assert "suspicious_columns" in result



def test_outliers_detects_extreme_value(sample_df):
    result = check_outliers(sample_df, iqr_factor=1.5)
    # 999.0 est un outlier extreme dans retard_moyen_depart
    assert "retard_moyen_depart" in result["flagged_columns"]

def test_outliers_no_flag_on_normal(sample_df):
    result = check_outliers(sample_df, iqr_factor=1.5)
    # cause_externe [0.1, 0.2, 0.3, 0.1] pas d'outlier
    assert "cause_externe" not in result["flagged_columns"]

def test_outliers_returns_dict(sample_df):
    result = check_outliers(sample_df)
    assert isinstance(result, dict)
    assert "flagged_columns" in result


def test_duplicates_detects_duplicate_rows(sample_df):
    result = check_duplicates(sample_df)
    assert result["duplicate_count"] > 0

def test_duplicates_returns_dict(sample_df):
    result = check_duplicates(sample_df)
    assert isinstance(result, dict)
    assert "duplicate_count" in result


def test_cardinality_returns_dict(sample_df):
    result = check_cardinality(sample_df)
    assert isinstance(result, dict)
    assert "high_cardinality_columns" in result



def test_full_profile_contains_all_keys(sample_df):
    result = run_full_profile(
        sample_df,
        missing_threshold=0.05,
        iqr_factor=1.5,
        cardinality_threshold=0.95,
        key_cols=["periode"],
    )
    expected_keys = {"missing", "types", "outliers", "duplicates", "cardinality"}
    assert expected_keys.issubset(result.keys())