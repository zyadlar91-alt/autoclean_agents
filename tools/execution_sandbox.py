"""
Safe execution environment for Agent 2 (Planner) generated code.

The Planner writes Python pandas code as a string.
This module runs it in a controlled namespace with no file system access,
no imports beyond what we explicitly allow and returns a structured
result with the cleaned DataFrame and any execution errors.

Think of it as the isolated room where the Validator runs the
Planner's script without risking anything outside.
"""

import traceback
import pandas as pd
import numpy as np
from typing import Any


# Allowed namespace  the only names the generated code can use
SAFE_GLOBALS = {
    "__builtins__": {
        "print": print,
        "len": len,
        "range": range,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "enumerate": enumerate,
        "zip": zip,
        "isinstance": isinstance,
        "min": min,
        "max": max,
        "abs": abs,
    },
    "pd": pd,
    "np": np,
}


def run_cleaning_code(code: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute LLM-generated cleaning code against a DataFrame copy.

    Parameters
    ----------
    code : str
        Python code produced by the Planner agent.
        Must operate on a variable named `df`.
    df : pd.DataFrame
        The noisy input dataset.

    Returns
    -------
    dict with keys:
        success     bool
        df_cleaned  pd.DataFrame | None
        error       str | None
        rows_before int
        rows_after  int | None
    """
    local_ns = {"df": df.copy()}

    try:
        exec(code, SAFE_GLOBALS, local_ns) 

        df_cleaned = local_ns.get("df")

        if df_cleaned is None or not isinstance(df_cleaned, pd.DataFrame):
            return {
                "success": False,
                "df_cleaned": None,
                "error": "Generated code did not produce a DataFrame named `df`.",
                "rows_before": len(df),
                "rows_after": None,
            }

        return {
            "success": True,
            "df_cleaned": df_cleaned,
            "error": None,
            "rows_before": len(df),
            "rows_after": len(df_cleaned),
        }

    except Exception:
        return {
            "success": False,
            "df_cleaned": None,
            "error": traceback.format_exc(),
            "rows_before": len(df),
            "rows_after": None,
        }


def compute_quality_delta(df_before: pd.DataFrame,
                          df_after: pd.DataFrame) -> dict[str, Any]:
    """
    Compare quality metrics before and after cleaning.
    Used by the Validator to build its final report.
    """
    def missing_pct(d):
        return round(d.isnull().sum().sum() / (d.shape[0] * d.shape[1]), 4)

    def duplicate_count(d):
        return int(d.duplicated().sum())

    return {
        "missing_pct_before": missing_pct(df_before),
        "missing_pct_after": missing_pct(df_after),
        "duplicates_before": duplicate_count(df_before),
        "duplicates_after": duplicate_count(df_after),
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "cols": df_after.shape[1],
    }