"""Baselines scored on the 2022 validation year only.

Run from the repo root:
    python -m src.run_baselines
    python -m src.run_baselines path/to/clean_ped_cyc_crashes.csv

Writes:
    results/baseline_validation_2022.csv
    results/data_summary_by_year.csv
"""
import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import (
    CATEGORICAL, DATA_PATH, FEATURES, NUMERIC, SEED,
    compute_metrics, data_summary, load_data, split_by_year,
)

RESULTS_DIR = "results"


def make_preprocessor():
    """Median impute and scale numeric columns.
    Fit on the training years only."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    df = load_data(path)
    train, val = split_by_year(df)

    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]

    models = {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([
            ("prep", make_preprocessor()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=SEED)),
        ]),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        rows.append({"model": name, **compute_metrics(y_val, scores, threshold=0.5)})

    results = pd.DataFrame(rows).round(3)
    summary = data_summary(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "baseline_validation_2022.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "data_summary_by_year.csv"), index=False)

    print(f"Train rows: {len(train)}  Validation (2022) rows: {len(val)}  "
          f"Validation serious/fatal: {y_val.mean():.1%}\n")
    print(results.to_string(index=False))
    print("\nUsable rows and serious/fatal share by year:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
