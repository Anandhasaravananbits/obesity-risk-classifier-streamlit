# Obesity Level Classification with Streamlit

## a. Problem statement

Predict a person's obesity level (7 ordered classes, from `Insufficient_Weight` to `Obesity_Type_III`) from physical measurements and eating/lifestyle habits. Five classifiers are trained on the same dataset, compared on a held-out test set, and exposed through a Streamlit app that evaluates the saved models on an uploaded CSV without retraining.

## b. Dataset description

- Source: UCI Machine Learning Repository, "Estimation of obesity levels based on eating habits and physical condition" (ID 544).
- Instances: 2,111 (about 77% synthetic via SMOTE, about 23% collected from a survey in Mexico, Peru and Colombia).
- Features: 16 (minimum required: 12). Target: `NObeyesdad`, 7 classes, roughly balanced (272 to 351 rows each).
- Missing values: none. Exact duplicate rows: 24 (kept).

| Type | Features |
|---|---|
| Numeric | `Age`, `Height`, `Weight`, `FCVC` (vegetable frequency), `NCP` (main meals), `CH2O` (water intake), `FAF` (physical activity), `TUE` (screen time) |
| Categorical | `Gender`, `family_history_with_overweight`, `FAVC` (high-calorie food), `CAEC` (snacking), `SMOKE`, `SCC` (calorie monitoring), `CALC` (alcohol), `MTRANS` (transport) |

Preprocessing (inside each saved sklearn `Pipeline`): `StandardScaler` on numeric columns and `OneHotEncoder` on categorical columns. The target is label-encoded (`model/label_encoder.joblib`).

Split: 80/20 stratified, `random_state=2026` (1,688 train / 423 test). The test split is saved as `test_data.csv`.

## c. GitHub repository

https://github.com/Anandhasaravananbits/obesity-risk-classifier-streamlit

```
obesity-risk-classifier-streamlit/
|-- app.py              Streamlit app
|-- train_models.py     training and evaluation script
|-- metrics_utils.py    shared metric calculation
|-- requirements.txt
|-- test_data.csv       held-out test split (features + target)
|-- data/raw/           original UCI CSV
|-- model/              5 saved pipelines, label encoder, metrics.json
```

Run locally:

```
pip install -r requirements.txt
python train_models.py      # optional, regenerates model/ and test_data.csv
streamlit run app.py
```

Live app: STREAMLIT_LINK_HERE

## d. Models used

Multi-class metrics: Precision, Recall and F1 are weighted averages; AUC is one-vs-rest macro averaged.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8794 | 0.9879 | 0.8785 | 0.8794 | 0.8782 | 0.8594 |
| Decision Tree | 0.9267 | 0.9562 | 0.9263 | 0.9267 | 0.9264 | 0.9144 |
| KNN | 0.8463 | 0.9677 | 0.8516 | 0.8463 | 0.8335 | 0.8240 |
| Naive Bayes | 0.5626 | 0.8894 | 0.5887 | 0.5626 | 0.5122 | 0.5098 |
| Support Vector Machine | 0.9574 | 0.9973 | 0.9582 | 0.9574 | 0.9576 | 0.9504 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline (about 88% accuracy) with the second-best AUC. Class boundaries are largely linear in weight, height and age, but neighbouring BMI-based classes overlap, so some are confused. |
| Decision Tree | Second-best accuracy and MCC (about 93%). Splits on `Weight` and `Height` recover the BMI-style class cut-offs well. AUC is lowest of the strong models because leaf probabilities are coarse, and the tree may overfit with limited regularisation. |
| KNN | Weakest of the non-Naive-Bayes models (about 85%). Distance is affected by the 16 one-hot and scaled features, so irrelevant lifestyle columns dilute the weight and height signal. |
| Naive Bayes | Clearly the worst (about 56%). The Gaussian and conditional-independence assumptions fail because Height and Weight jointly determine the class, and the one-hot features are not Gaussian. |
| Support Vector Machine | Best on every metric (about 95.7% accuracy, MCC 0.95, AUC 0.997). The RBF kernel captures the non-linear boundary between adjacent classes. |
| **Overall winner** | **Support Vector Machine (RBF)**, followed by Decision Tree. |

Note: the target is defined from BMI (derived from `Height` and `Weight`), so these two features dominate and scores are high. Most of the dataset is synthetic (SMOTE), which can make results optimistic compared with real data.

## Streamlit app features

- CSV upload of the test data (`test_data.csv` works as a sample).
- Model selection dropdown.
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC).
- Confusion matrix and classification report.
- A comparison table of all five models on the uploaded file.
- Saved models are loaded from `model/` and nothing is retrained.
