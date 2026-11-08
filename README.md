# Obesity Level Classification - ML Assignment 2

M.Tech (AIML), Machine Learning, Assignment 2

## a. Problem statement

The goal is to predict a person's obesity level from their body measurements and eating / lifestyle habits. The target has 7 classes (Insufficient Weight, Normal Weight, Overweight Level I and II, Obesity Type I, II and III). I trained 5 classification models on the same dataset, compared them using Accuracy, AUC, Precision, Recall, F1 and MCC, and built a Streamlit app where a test CSV can be uploaded and evaluated with the saved models.

## b. Dataset description

- Dataset: "Estimation of obesity levels based on eating habits and physical condition" from the UCI ML Repository.
- Number of rows: 2111
- Number of features: 16 (the assignment needs at least 12), plus 1 target column `NObeyesdad`
- Classes: 7 multi-class target, fairly balanced (272 to 351 rows per class)
- No missing values. There are 24 exact duplicate rows which I left as they are.
- About 77% of the rows are synthetic (made with SMOTE) and the rest come from an online survey.

Numeric features: Age, Height, Weight, FCVC (vegetable eating), NCP (number of main meals), CH2O (water intake), FAF (physical activity), TUE (time using technology).

Categorical features: Gender, family_history_with_overweight, FAVC (high calorie food), CAEC (eating between meals), SMOKE, SCC (calorie monitoring), CALC (alcohol), MTRANS (transport used).

Preprocessing: numeric columns are scaled with StandardScaler and categorical columns are one-hot encoded. This is done inside a sklearn Pipeline for every model, so the saved model files already include the preprocessing. The target labels are encoded with a LabelEncoder (saved as `model/label_encoder.joblib`).

Train/test split: 80/20 stratified with random_state 2026, giving 1688 training rows and 423 test rows. The test part is saved as `test_data.csv` and this is the file to upload in the app.

## c. GitHub repository

https://github.com/Anandhasaravananbits/obesity-risk-classifier-streamlit

Files in the repo:

```
app.py               streamlit app
train_models.py      trains the 5 models and saves them
metrics_utils.py     function used to calculate the metrics
requirements.txt
test_data.csv        test split (features + target)
data/raw/            original csv from UCI
model/               saved models (.joblib), label encoder, metrics.json
```

To run it locally:

```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Live Streamlit app: STREAMLIT_LINK_HERE

## d. Models used

All 5 models were trained on the same training split and evaluated on the same 423 test rows. Since this is a multi-class problem, Precision, Recall and F1 are weighted averages and AUC is one-vs-rest macro average.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8794 | 0.9879 | 0.8785 | 0.8794 | 0.8782 | 0.8594 |
| Decision Tree | 0.9267 | 0.9562 | 0.9263 | 0.9267 | 0.9264 | 0.9144 |
| KNN | 0.8463 | 0.9677 | 0.8516 | 0.8463 | 0.8335 | 0.8240 |
| Naive Bayes | 0.5626 | 0.8894 | 0.5887 | 0.5626 | 0.5122 | 0.5098 |
| Support Vector Machine | 0.9574 | 0.9973 | 0.9582 | 0.9574 | 0.9576 | 0.9504 |

Settings used: Logistic Regression (max_iter=2000), Decision Tree (max_depth=12), KNN (k=7, distance weights), Gaussian Naive Bayes (default), SVM (RBF kernel, C=5).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gave about 88% accuracy and a high AUC (0.988), so it ranks the classes well even though some predictions are wrong. Most mistakes are between neighbouring classes, for example Overweight Level I and II. Recall is lowest for Overweight Level I (0.71) and Normal Weight (0.76). |
| Decision Tree | Second best with 92.7% accuracy. The most important features were Weight, Height and Gender, which makes sense because the class is based on BMI. Its AUC (0.956) is lower than the others because tree probabilities are not smooth. |
| KNN | 84.6% accuracy, the lowest among the four models that worked well. It had trouble with Normal Weight (recall 0.38), which it often mixed up with Overweight Level I. This may be because all 16 features (after one-hot encoding) count in the distance, including lifestyle ones that do not matter much. |
| Naive Bayes | Worst model by a big margin (56% accuracy, MCC 0.51). It almost fails on Normal Weight, Overweight Level I and II (recall 0.21, 0.14, 0.16). Probably because it treats features as independent, but the class really depends on Height and Weight together. |
| Support Vector Machine | Best on every metric: 95.7% accuracy, AUC 0.997, MCC 0.950. Every class had recall of 0.90 or more. The RBF kernel seems to handle the non-linear boundaries between the neighbouring classes well. |
| **Overall winner for this dataset** | **Support Vector Machine**, then Decision Tree. |

One thing to keep in mind: the target is basically calculated from BMI (Weight and Height), so these features make the task easier. Also most of the data is synthetic, so the scores may be higher than what we would get on fully real data.

## Streamlit app

The app (`app.py`) has:
- CSV upload for the test data
- a dropdown to choose the model
- the 6 evaluation metrics for the chosen model
- confusion matrix and classification report
- a table comparing all 5 models on the uploaded file

The saved models are loaded from the `model/` folder and nothing is retrained in the app.
