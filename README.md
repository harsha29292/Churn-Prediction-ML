# Data-Centric Churn Prediction with Error Analysis

**Predict customer churn and diagnose why models fail, not just how they perform.**

---

## 1. Problem Framing

Customer churn refers to the event where a subscriber discontinues their service with a provider. Telecommunications companies use churn prediction to proactively retain customers before they leave, directing targeted interventions such as discounts or loyalty offers.

In this domain, **false negatives are substantially more costly than false positives**. Missing a churner means losing months of future revenue — often $50–$100/month per customer — with no opportunity to intervene. Incorrectly flagging a loyal customer, by contrast, incurs only the cost of an unnecessary retention incentive. The asymmetry in business cost makes recall on the churn class the primary performance criterion, not overall accuracy.

Missing a churner leads to revenue loss, while wrongly flagging a loyal customer incurs only incentive cost.

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| Dataset | Telco Customer Churn |
| Size | 7,043 rows × 21 columns |
| Target | `Churn` (Yes / No) |
| Class distribution | ~74% No, ~26% Yes |
| Cold-start customers | 11 customers with `tenure = 0` |

The dataset contains demographic features (gender, senior citizen status, dependents), account features (tenure, contract type, payment method), and service features (phone, internet, streaming, tech support).

**Accuracy alone is misleading here.** A naive model that always predicts "No Churn" achieves 73.5% accuracy by exploiting class imbalance — higher than most trained models at face value. This makes recall on the minority class the only honest signal of model usefulness.

---

## 3. Exploratory Insights

Analysis was done prior to modeling to form hypotheses about churn drivers. Key findings:

- **Low tenure → high churn.** Customers in their first 12 months are far more likely to churn than established customers. Average tenure for churners is significantly lower than non-churners.
- **Month-to-month contracts → ~43% churn rate.** Customers on flexible contracts churn at nearly twice the rate of those on annual or two-year contracts, likely reflecting lower switching costs.
- **Higher monthly charges correlate with churn.** Churners spend more per month on average, suggesting price sensitivity or dissatisfaction with value-to-cost ratio.
- **Long-tenure churners exist and are difficult to detect.** A subset of customers with 20+ months of tenure still churn. These cases lack obvious behavioral signals in the available features and represent the hard ceiling on model recall.

These patterns informed feature selection: `tenure`, `MonthlyCharges`, and `Contract` type were used as the core predictors.

---

## 4. Models Compared

| Model | Recall (Churn) | Precision (Churn) | Key Behavior |
|---|---|---|---|
| Logistic Regression | ~0.49 | ~0.60 | Conservative, linear decision boundary |
| Decision Tree (shallow) | ~0.47 | ~0.49 | Interpretable but misses complex patterns |
| Decision Tree (deep) | ~0.47 | ~0.49 | Overfits training data, poor generalization |
| Random Forest | ~0.46 | ~0.55 | Stable and consistent, recall ceiling hit |
| XGBoost (Boosting) | ~0.49 | ~0.64 | Cleaner precision, not broader recall |

No single model is declared "best." The variation in recall across all models is minimal (~0.46–0.49), which is a diagnostic signal in itself.

**Implementation note:** Logistic Regression was implemented from scratch (gradient descent with sigmoid activation and binary cross-entropy loss) before using the sklearn version to validate the approach.

---

## 5. Why Accuracy is Misleading

The baseline accuracy — achieved by predicting "No Churn" for every customer — is **73.5%**. Every trained model in this project exceeds that baseline, which might suggest success. It does not.

Despite outperforming the baseline on accuracy, all models still miss approximately **50% of actual churners**. A recall of ~0.49 on the churn class means roughly one in two customers who will leave is not flagged. In a business context, this is operationally equivalent to flipping a coin for churn detection. Accuracy masks this failure by rewarding correct predictions on the majority class, which requires no modeling at all. Recall on the churn class is the only metric that reflects whether the model is doing useful work.

---

## 6. Error Analysis

This section examines *who* the models fail on, not just *how often*.

**False Negatives — Churners the model missed — share consistent characteristics:**

- **Mid-to-long tenure customers** (average ~26 months). These are not new customers; they have history with the company, which makes their churn behaviorally unexpected based on available signals.
- **Moderate spenders** with monthly charges in the mid-range. They are not outliers in either direction — neither the cheapest nor the most expensive customers.
- **No obvious churn risk markers.** Their contract type, service profile, and demographic features do not clearly distinguish them from loyal customers.
- **Model uncertainty was high.** XGBoost predicted churn probabilities of ~0.30–0.40 for false negatives — just below the 0.5 decision threshold. The model was not confidently wrong; it was uncertain and defaulted to the majority class.
- **Boosting did not significantly reduce false negatives** compared to simpler models. Switching from Logistic Regression to XGBoost improved precision modestly but did not move recall beyond ~0.49.

**This indicates a data limitation rather than a modeling limitation.**

The features in this dataset do not contain enough signal to distinguish a long-tenure churner from a long-tenure loyal customer at prediction time. No amount of algorithm tuning will resolve a feature gap.

---

## 7. What Data Would Fix This

The following specific data types — not "more data" in general — would directly address the false negative problem:

| Data Type | Why It Helps |
|---|---|
| **Usage trends over time** (call volume, data consumption month-over-month) | Declining usage before churn is a leading indicator not captured in point-in-time snapshots |
| **Customer support interactions** (number of calls, ticket types, resolution satisfaction) | Frequent or unresolved complaints precede voluntary churn and are invisible in billing data |
| **Billing events and disputes** (late payments, payment failures, plan downgrades) | Payment friction signals financial stress or dissatisfaction before the customer formally churns |
| **Service quality indicators** (outage history, network speed complaints, coverage issues) | Churn driven by service degradation cannot be detected from demographic or contract data alone |

Each of these represents a *behavioral* signal — something the customer does in the period leading up to churn — rather than a *static* attribute. The current dataset contains almost exclusively static attributes. The gap between static and behavioral data explains the ~0.50 recall ceiling observed across all models.

---

## 8. Final Takeaway

This project demonstrates that churn prediction fails not because of insufficient modeling, but because key behavioral signals are absent from the dataset. Four distinct model families — logistic regression, decision trees, random forests, and gradient boosting — all converge to approximately the same recall ceiling (~0.49), ruling out algorithmic limitation as the cause. Error analysis reveals that false negatives are systematically mid-tenure, moderate-spending customers whose churn the available features simply cannot anticipate. This points to a data-centric diagnosis: the recall ceiling is imposed by the dataset itself, not by the models trained on it. Improving churn prediction in production requires richer behavioral data, not a better algorithm — a distinction that only disciplined error analysis can surface.

---

## Project Structure

```
dataclean/
├── churn.csv           # Telco Customer Churn dataset (7043 × 21)
├── main.py             # EDA, model training, evaluation, and error analysis
├── preprocess.py       # Feature engineering and manual normalization
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install xgboost
python main.py
```

## Dependencies

```
pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost
```
