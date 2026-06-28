# Credit Card Fraud Detection

## Problem
Credit card fraud is rare but costly — fraudulent transactions make up a tiny
fraction of all transactions, so a model needs to catch as many frauds as
possible without flagging so many legitimate transactions that the system
becomes unusable. This project builds and compares several models to detect
fraudulent transactions in real time.

## Data
Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`) — 283,726
transactions by European cardholders over 2 days. Features `V1`-`V28` are
PCA-anonymized for privacy; `Time` and `Amount` are the only raw features.
Target: `Class` (1 = fraud, 0 = legitimate). Frauds make up just 0.17% of
transactions (473 out of 283,726) — a severe class imbalance that shapes
every modeling decision below.

## Approach
- Verified no missing values; removed duplicate rows
- Scaled `Amount`; converted `Time` to hour-of-day
- Split data with stratification to preserve class ratio in train/test
- Applied SMOTE to the training set only (test set kept untouched and
  imbalanced, to reflect real-world conditions)
- Trained and compared Logistic Regression, Random Forest, and XGBoost
- Tuned XGBoost's hyperparameters via RandomizedSearchCV, optimizing
  for PR-AUC rather than accuracy
- Used SHAP for model interpretability

## Results

| Model | ROC-AUC | PR-AUC | Recall (fraud) | Precision (fraud) |
|---|---|---|---|---|
| Logistic Regression | 0.9641 | 0.6543 | 0.8737 | 0.1030 |
| Random Forest | 0.9653 | 0.8052 | 0.7684 | 0.9012 |
| XGBoost | 0.9725 | 0.7892 | 0.8000 | 0.8085 |

XGBoost performed best overall, with the highest ROC-AUC and a strong
balance between catching fraud and avoiding false alarms — Logistic
Regression, by contrast, caught more fraud cases by recall but did so at
the cost of an unusable precision of just 10%. SHAP analysis shows `V14` as
the strongest driver of fraud risk, followed by `V4` and transaction `Hour`.
Lower values of `V14` and higher values of `V4` both push predictions toward
fraud, and certain hours — particularly the early-morning (~2-3 AM) and
mid-morning (~11 AM) windows — show disproportionately higher fraud rates
than the rest of the day.

## Business takeaway
At the chosen threshold, the model catches 80% of fraudulent transactions
while correctly clearing 99.97% of legitimate ones — meaning only about
0.03% of legitimate transactions would ever get incorrectly flagged for
review. That's a far better tradeoff than reviewing transactions at random,
and one that scales to the volume a real card network would need to handle.

## Files
- `fraud_detection_model.ipynb` — full notebook (EDA, cleaning, modeling, SHAP)
- `fraud_model.pkl` — trained final model
- `scaler.pkl` — fitted Amount scaler
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies

## Live demo
https://credit-fraud-app-2026.streamlit.app/

## Tools
Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP,
Streamlit, matplotlib/seaborn