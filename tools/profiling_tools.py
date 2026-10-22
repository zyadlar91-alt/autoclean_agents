"""
profiling_tools.py
------------------
Pure-pandas profiling functions used by Agent 1 (Profiler).
Each function returns a structured dict — no LLM involved here.
The agent uses these as its tools and assembles the final JSON report.
"""

import pandas as pd
import numpy as np
from typing import Any


def check_missing(df: pd.DataFrame, threshold: float = 0.05) -> dict[str, Any]:
    """Flag columns where missing value rate exceeds threshold."""
    missing = df.isnull().sum()
    pct = (missing / len(df)).round(4)
    flagged = pct[pct > threshold].to_dict()
    return {
        "check": "missing_values",
        "threshold": threshold,
        "flagged_columns": {
            col: {"missing_count": int(missing[col]), "missing_pct": float(pct[col])}
            for col in flagged
        },
        "total_flagged": len(flagged),
    }


def check_types(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect columns that should be numeric but contain non-numeric strings.
    Heuristic: if a column's inferred dtype is object but >80% of non-null
    values can be cast to float, flag the remaining values as corrupt.
    """
    issues = {}
    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        numeric_mask = pd.to_numeric(series, errors="coerce").notna()
        numeric_ratio = numeric_mask.sum() / len(series)
        if numeric_ratio >= 0.80:
            corrupt = series[~numeric_mask]
            if len(corrupt) > 0:
                issues[col] = {
                    "corrupt_count": int(len(corrupt)),
                    "corrupt_pct": round(len(corrupt) / len(df), 4),
                    "sample_values": corrupt.unique()[:5].tolist(),
                }
    return {
        "check": "type_corruption",
        "flagged_columns": issues,
        "total_flagged": len(issues),
    }


def check_outliers(df: pd.DataFrame, iqr_factor: float = 1.5) -> dict[str, Any]:
    """IQR-based outlier detection on numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    issues = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        if count > 0:
            issues[col] = {
                "outlier_count": count,
                "outlier_pct": round(count / len(df), 4),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "max_value": round(float(series.max()), 4),
                "min_value": round(float(series.min()), 4),
            }
    return {
        "check": "outliers",
        "iqr_factor": iqr_factor,
        "flagged_columns": issues,
        "total_flagged": len(issues),
    }


def check_duplicates(df: pd.DataFrame,
                     key_cols: list[str] | None = None) -> dict[str, Any]:
    """Detect duplicate rows, optionally scoped to key columns."""
    if key_cols:
        existing = [c for c in key_cols if c in df.columns]
        dup_mask = df.duplicated(subset=existing)
    else:
        dup_mask = df.duplicated()
    count = int(dup_mask.sum())
    return {
        "check": "duplicates",
        "key_columns": key_cols,
        "duplicate_count": count,
        "duplicate_pct": round(count / len(df), 4),
        "flagged": count > 0,
    }


def check_cardinality(df: pd.DataFrame,
                      threshold: float = 0.95) -> dict[str, Any]:
    """
    Flag object columns where unique ratio > threshold
    (likely free-text or ID columns that shouldn't be treated as categories).
    """
    issues = {}
    for col in df.select_dtypes(include="object").columns:
        ratio = df[col].nunique() / len(df)
        if ratio > threshold:
            issues[col] = {
                "unique_count": int(df[col].nunique()),
                "unique_ratio": round(ratio, 4),
            }
    return {
        "check": "cardinality",
        "threshold": threshold,
        "flagged_columns": issues,
        "total_flagged": len(issues),
    }


def run_full_profile(df: pd.DataFrame,
                     missing_threshold: float = 0.05,
                     iqr_factor: float = 1.5,
                     cardinality_threshold: float = 0.95,
                     key_cols: list[str] | None = None) -> dict[str, Any]:
    """Run all checks and return a unified quality report."""
    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "checks": [
            check_missing(df, threshold=missing_threshold),
            check_types(df),
            check_outliers(df, iqr_factor=iqr_factor),
            check_duplicates(df, key_cols=key_cols),
            check_cardinality(df, threshold=cardinality_threshold),
        ],
    }