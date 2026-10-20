"""
noise_injection.py
------------------
Injects controlled, reproducible noise into the SNCF TGV dataset.

Noise types introduced:
  1. Missing values     — NaN injected in numeric columns
  2. Type corruption    — numeric cells replaced with string "N/A"
  3. Outliers           — values set 5× above the column's 99th percentile
  4. Duplicate rows     — selected rows duplicated
  5. Cause % drift      — prct_cause columns scaled so row sum ≠ 100

A ground-truth JSON is saved alongside the noisy CSV so agent output
can be evaluated with recall / precision metrics.

Usage:
    python tools/noise_injection.py --input data/raw/sncf_tgv_raw.csv \
                                    --output data/raw/sncf_tgv_noisy.csv \
                                    --seed 42
"""

import argparse
import json
import os
import numpy as np
import pandas as pd


# ── Issue definitions ──────────────────────────────────────────────────────

MISSING_COLS = [
    "retard_moyen_depart",
    "retard_moyen_arrivee",
    "nb_train_retard_sup_15",
]
MISSING_FRAC = 0.06   # inject into 6% of rows per column

TYPE_CORRUPT_COLS = [
    "nb_train_prevu",
    "nb_annulation",
]
TYPE_CORRUPT_FRAC = 0.04

OUTLIER_COLS = [
    "retard_moyen_depart",
    "nb_train_retard_sup_60",
]
OUTLIER_FRAC = 0.02

DUPLICATE_FRAC = 0.015   # duplicate ~1.5% of rows

CAUSE_DRIFT_FRAC = 0.05  # mess up cause sums in 5% of rows
CAUSE_COLS = [
    "prct_cause_externe",
    "prct_cause_infra",
    "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant",
    "prct_cause_gestion_gare",
    "prct_cause_prise_en_charge_voyageurs",
]


# ── Main ──────────────────────────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    ground_truth = {}
    n = len(df)

    # 1. Missing values
    missing_log = {}
    for col in MISSING_COLS:
        idx = rng.choice(n, size=int(n * MISSING_FRAC), replace=False)
        df.loc[idx, col] = np.nan
        missing_log[col] = sorted(int(i) for i in idx)
    ground_truth["missing"] = missing_log

    # 2. Type corruption
    type_log = {}
    for col in TYPE_CORRUPT_COLS:
        idx = rng.choice(n, size=int(n * TYPE_CORRUPT_FRAC), replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = "N/A"
        type_log[col] = sorted(int(i) for i in idx)
    ground_truth["type_corruption"] = type_log

    # 3. Outliers
    outlier_log = {}
    for col in OUTLIER_COLS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        p99 = numeric.quantile(0.99)
        idx = rng.choice(n, size=int(n * OUTLIER_FRAC), replace=False)
        df.loc[idx, col] = round(p99 * 5, 2)
        outlier_log[col] = {"injected_value": round(p99 * 5, 2),
                            "rows": sorted(int(i) for i in idx)}
    ground_truth["outliers"] = outlier_log

    # 4. Duplicates
    dup_idx = rng.choice(n, size=int(n * DUPLICATE_FRAC), replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    ground_truth["duplicates"] = {"original_rows": sorted(int(i) for i in dup_idx),
                                   "count": len(dup_idx)}

    # 5. Cause % drift
    drift_idx = rng.choice(len(df), size=int(len(df) * CAUSE_DRIFT_FRAC), replace=False)
    for i in drift_idx:
        noise = rng.uniform(0.8, 1.3, size=len(CAUSE_COLS))
        for j, col in enumerate(CAUSE_COLS):
            val = pd.to_numeric(df.at[i, col], errors="coerce")
            if pd.notna(val):
                df.at[i, col] = round(val * noise[j], 4)
    ground_truth["cause_drift"] = {"rows": sorted(int(i) for i in drift_idx)}

    return df, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Inject noise into SNCF TGV dataset")
    parser.add_argument("--input",  default="data/raw/sncf_tgv_raw.csv")
    parser.add_argument("--output", default="data/raw/sncf_tgv_noisy.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";", encoding="utf-8-sig")
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    df_noisy, ground_truth = inject_noise(df.copy(), seed=args.seed)

    df_noisy.to_csv(args.output, index=False)
    print(f"Noisy dataset saved → {args.output} ({df_noisy.shape[0]} rows)")

    gt_path = os.path.join("results", "noise_ground_truth.json")
    os.makedirs("results", exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Ground truth saved  → {gt_path}")

    # Summary
    total_issues = (
        sum(len(v) for v in ground_truth["missing"].values()) +
        sum(len(v) for v in ground_truth["type_corruption"].values()) +
        sum(len(v["rows"]) for v in ground_truth["outliers"].values()) +
        ground_truth["duplicates"]["count"] +
        len(ground_truth["cause_drift"]["rows"])
    )
    print(f"\nTotal injected issues: {total_issues}")


if __name__ == "__main__":
    main()
