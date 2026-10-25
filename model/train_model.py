
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "employee.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Attrition"
IGNORED = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your employee dataset at dataset/employee.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError("Dataset must contain an 'Attrition' column.")

    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()

    y = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = df.drop(
        columns=[TARGET] + IGNORED,
        errors="ignore"
    )

    # Remove columns with only one unique value.
    X = X.loc[:, X.nunique(dropna=False) > 1]

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    trained_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        trained_models[name] = pipeline
        results.append({
            "model": name,
            "accuracy": round(accuracy, 4),
            "roc_auc": round(auc, 4),
        })

        print(
            f"{name}: "
            f"Accuracy={accuracy:.3f}, ROC-AUC={auc:.3f}"
        )

    # Select the model with the higher ROC-AUC.
    results.sort(key=lambda item: item["roc_auc"], reverse=True)
    best_name = results[0]["model"]
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "attrition_model.pkl"
    )

    joblib.dump(
        best_model.named_steps["preprocessor"],
        MODEL_DIR / "preprocessor.pkl"
    )

    with open(MODEL_DIR / "model_info.json", "w") as file:
        json.dump({
            "best_model": best_name,
            "features": X.columns.tolist(),
            "metrics": results,
        }, file, indent=4)

    print(f"\nBest model: {best_name}")
    print("Model files saved successfully.")


if __name__ == "__main__":
    main()
