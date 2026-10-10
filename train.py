"""Train and evaluate a student final-grade prediction model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student-mat.csv"
# The file also works when uploaded directly to a repository root with the CSV.
if not DATA_PATH.exists():
    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "student-mat.csv"
MODEL_PATH = ROOT / "models" / "student_grade_model.joblib"
RESULTS_PATH = ROOT / "results" / "metrics.json"


def main() -> None:
    data = pd.read_csv(DATA_PATH, sep=";")

    # G1 and G2 are earlier exam grades. Excluding them makes the exercise focus
    # on the remaining student and study-related features.
    features = data.drop(columns=["G3", "G1", "G2"])
    target = data["G3"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    numeric_columns = x_train.select_dtypes(include="number").columns
    categorical_columns = x_train.select_dtypes(exclude="number").columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mean_absolute_error": round(mean_absolute_error(y_test, predictions), 3),
        "r2_score": round(r2_score(y_test, predictions), 3),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
