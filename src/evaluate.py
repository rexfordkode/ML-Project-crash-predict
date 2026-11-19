"""Evaluation helpers: data loading, year-based split, and metrics.

Target: serious or fatal injury (INJSEV_IM codes 3, 4) = 1,
        no / minor / possible injury (codes 0, 1, 2) = 0.
        Code 5 (injured, severity unknown) is dropped.
INJSEV_IM is used only to build the target, never as a feature.

Split (chronological):
    train      2016-2021
    validation 2022 (model selection and threshold tuning)
    test       2023 (locked; used once for the final model)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
DATA_PATH = "data/processed/clean_ped_cyc_crashes.csv"

TRAIN_YEARS = list(range(2016, 2022))
VAL_YEAR = 2022
TEST_YEAR = 2023

# Crash report information available before injury outcome is known.
NUMERIC = ["AGE", "HOUR", "MONTH", "DAY_WEEK", "VSPD_LIM", "MOD_YEAR_veh"]
CATEGORICAL = [
    "SEX", "PER_TYP", "LGT_COND", "WEATHER", "URBANICITY", "REGION",
    "VTRAFWAY", "TYP_INT", "REL_ROAD", "RELJCT2", "LOCATION", "HIT_RUN",
    "BODY_TYP_veh",
]
FEATURES = NUMERIC + CATEGORICAL

# Vehicle type features (dropped later to test the vehicle type question).
VEHICLE_FEATURES = ["BODY_TYP_veh", "MOD_YEAR_veh"]

# CRSS "unknown / not reported" codes for numeric columns -> NaN.
UNKNOWN = {
    "AGE": [998, 999],
    "HOUR": [99],
    "VSPD_LIM": [98, 99],
    "MOD_YEAR_veh": [9998, 9999],
}

SERIOUS_CODES = [3, 4]
NOT_SERIOUS_CODES = [0, 1, 2]


def load_data(path=DATA_PATH):
    """Load the cleaned file and return one row per person with a target.

    The cleaned file joins persons to every vehicle in the crash, which
    duplicates persons in multi vehicle crashes. We keep only the row for
    the striking vehicle (STR_VEH == VEH_NO_veh).
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[df["STR_VEH"] == df["VEH_NO_veh"]].copy()

    df = df[df["INJSEV_IM"].isin(SERIOUS_CODES + NOT_SERIOUS_CODES)]
    df["target"] = df["INJSEV_IM"].isin(SERIOUS_CODES).astype(int)

    for col, codes in UNKNOWN.items():
        df[col] = df[col].replace(codes, np.nan)

    return df.reset_index(drop=True)


def split_by_year(df, include_test=False):
    """Split by crash year. The 2023 test set is returned only on request."""
    train = df[df["YEAR"].isin(TRAIN_YEARS)]
    val = df[df["YEAR"] == VAL_YEAR]
    if include_test:
        return train, val, df[df["YEAR"] == TEST_YEAR]
    return train, val


def compute_metrics(y_true, y_score, threshold=0.5):
    """Precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix counts."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # A constant score (majority class) has no ranking: AUC 0.5, PR-AUC = base rate.
    if np.unique(y_score).size == 1:
        roc_auc, pr_auc = 0.5, y_true.mean()
    else:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def data_summary(df):
    """Usable rows and serious/fatal share by year."""
    summary = df.groupby("YEAR")["target"].agg(rows="size", serious_fatal="sum")
    summary["serious_fatal_pct"] = (100 * summary["serious_fatal"] / summary["rows"]).round(1)
    return summary.reset_index()
