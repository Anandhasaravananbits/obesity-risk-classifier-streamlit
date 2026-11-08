"""Train five classifiers on the UCI obesity dataset and save pipelines, test split and metrics."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from metrics_utils import compute_metrics

SEED = 2026
ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "raw" / "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_DIR = ROOT / "model"
TARGET = "NObeyesdad"


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=5.0, probability=True, random_state=SEED),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    X = df.drop(columns=[TARGET])
    encoder = LabelEncoder().fit(df[TARGET])
    y = encoder.transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Held-out file for the Streamlit upload; target kept as the original label text.
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    results = {}
    for name, clf in build_models().items():
        prep = ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)
        results[name] = compute_metrics(y_test, pipe.predict(X_test), proba)

        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, MODEL_DIR / file_name)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    table = pd.DataFrame(results).T.round(4)
    print(table.to_string())
    print(f"\ntrain rows: {len(X_train)}, test rows: {len(X_test)}")


if __name__ == "__main__":
    main()
