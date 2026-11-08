from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "NObeyesdad"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Obesity Level Classifier", layout="wide")


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_resource
def load_encoder():
    return joblib.load(MODEL_DIR / "label_encoder.joblib")


def evaluate(pipe, X, y_true):
    proba = pipe.predict_proba(X)
    pred = pipe.predict(X)
    return pred, compute_metrics(y_true, pred, proba, classes=pipe.classes_)


st.title("Obesity Level Prediction - Model Evaluation")
st.caption(
    "Upload a held-out CSV (features + `NObeyesdad` column). "
    "Pre-trained models are loaded from disk; nothing is retrained here."
)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Test data (CSV)", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    st.markdown("A sample file, `test_data.csv`, is in the GitHub repo.")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
encoder = load_encoder()
pipe = load_model(MODEL_FILES[model_name])

expected = list(pipe.feature_names_in_)
missing = [c for c in expected + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

unknown = set(df[TARGET].unique()) - set(encoder.classes_)
if unknown:
    st.error(f"Unknown labels in `{TARGET}`: {sorted(unknown)}")
    st.stop()

X = df[expected]
y_true = encoder.transform(df[TARGET])

st.subheader("Data preview")
st.write(f"{len(df)} rows, {len(expected)} features")
st.dataframe(df.head(10), use_container_width=True)

pred, scores = evaluate(pipe, X, y_true)

st.subheader(f"Metrics - {model_name}")
cols = st.columns(len(scores))
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
labels = list(encoder.classes_)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("All models on this file")
rows = {}
for name, f in MODEL_FILES.items():
    _, s = evaluate(load_model(f), X, y_true)
    rows[name] = s
st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)
