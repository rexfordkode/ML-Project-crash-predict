import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from src.evaluate import (
    DATA_PATH,
    FEATURES,
    NUMERIC,
    CATEGORICAL,
    SEED,
    load_data,
    split_by_year,
    compute_metrics,
)

RESULTS_DIR = "results"


def make_preprocessor():

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

    df = load_data(DATA_PATH)

    train, val = split_by_year(df)

    X_train = train[FEATURES]
    y_train = train["target"]

    X_val = val[FEATURES]
    y_val = val["target"]

    models = {

        "Random Forest": Pipeline([
            ("prep", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight="balanced",
                random_state=SEED
            ))
        ]),

        "XGBoost": Pipeline([
            ("prep", make_preprocessor()),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=SEED
            ))
        ])
    }

    results = []

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        scores = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, scores)

        metrics["model"] = name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df.to_csv(
        os.path.join(RESULTS_DIR,
                     "model_validation_2022.csv"),
        index=False
    )

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
