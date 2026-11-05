import json
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, MODEL_NAME, DATA_NOISY, DATA_CLEANED
from tools.execution_sandbox import run_cleaning_code, compute_quality_delta
from prompts.system_prompts import VALIDATOR_PROMPT


def load_ground_truth(path: str = "data/raw/ground_truth.json") -> dict:
    with open(path) as f:
        return json.load(f)


def compute_recall_precision(ground_truth: dict, corrections: dict) -> dict:
    results = {}
    total_tp = total_fp = total_fn = 0

    for col, true_indices in ground_truth.items():
        true_set = set(true_indices)
        pred_set = set(corrections.get(col, []))

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[col] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # micro-moyenne globale
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    results["_global"] = {
        "precision": round(global_precision, 3),
        "recall":    round(global_recall, 3),
    }
    return results


def run_validator(cleaning_code: str, profile_report: dict) -> dict:
    df_noisy = pd.read_csv(DATA_NOISY, sep=";", encoding="utf-8-sig")
    ground_truth = load_ground_truth()

    # executes the clean code in the sandbox
    sandbox_result = run_cleaning_code(cleaning_code, df_noisy)

    if sandbox_result["status"] != "ok":
        return {"status": "error", "message": sandbox_result["error"]}

    df_cleaned = sandbox_result["df"]
    delta = compute_quality_delta(df_noisy, df_cleaned)

    # indicex corrected by columns 
    corrections = {}
    for col in ground_truth:
        if col in df_noisy.columns and col in df_cleaned.columns:
            # row that had a problem and now is fixed
            mask = df_noisy[col].isna() & df_cleaned[col].notna()
            corrections[col] = list(df_noisy[mask].index)

    metrics = compute_recall_precision(ground_truth, corrections)

    # save
    df_cleaned.to_csv(DATA_CLEANED, sep=";", index=False, encoding="utf-8-sig")
    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump({"delta": delta, "recall_precision": metrics}, f, indent=2)

    return {
        "status": "ok",
        "delta": delta,
        "metrics": metrics,
    }