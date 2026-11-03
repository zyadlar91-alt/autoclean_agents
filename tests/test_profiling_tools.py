import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.profiling_tools import check_missing, check_outliers, check_duplicates

# test dataframe simple
@pytest.fixture
def df():
    return pd.DataFrame({
        "retard": [5.0, np.nan, 999.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "trains": [100, 200, 300, 100, 200, 300, 100, 200, 300, 100],
        "periode": ["2018-01"] * 5 + ["2018-02"] * 5,
    })

def test_missing_detects_nan(df):
    res = check_missing(df, threshold=0.05)
    assert "retard" in res["flagged_columns"]

def test_missing_no_false_positive(df):
    res = check_missing(df, threshold=0.05)
    assert "trains" not in res["flagged_columns"]

def test_outlier_detects_999(df):
    # TODO: fix -- IQR pas assez sensible sur petit dataset
    res = check_outliers(df, iqr_factor=1.5)
    assert "retard" in res["flagged_columns"]

def test_duplicates_count(df):
    res = check_duplicates(df)
    assert "duplicate_count" in res

def test_missing_returns_dict(df):
    res = check_missing(df)
    assert isinstance(res, dict)