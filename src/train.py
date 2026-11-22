"""Random Forest and XGBoost scored on the 2022 validation year only.

Run from the repo root:
    python -m src.train
    python -m src.train path/to/clean_ped_cyc_crashes.csv

Writes:
    results/model_validation_2022.csv
"""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import (
    DATA_PATH, FEATURES, SEED, compute_metrics, load_data, split_by_year,
)
from src.run_baselines import RESULTS_DIR, make_preprocessor


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    df = load_data(path)
    train, val = split_by_year(df)

    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Random forest": Pipeline([
            ("prep", make_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             random_state=SEED)),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor()),
            ("model", XGBClassifier(scale_pos_weight=pos_weight, random_state=SEED)),
        ]),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        train_scores = model.predict_proba(X_train)[:, 1]
        rows.append({"model": name, **compute_metrics(y_val, scores, threshold=0.5),
                     "train_pr_auc": compute_metrics(y_train, train_scores)["pr_auc"]})

    results = pd.DataFrame(rows).round(3)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "model_validation_2022.csv"), index=False)

    print(f"Train rows: {len(train)}  Validation (2022) rows: {len(val)}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
