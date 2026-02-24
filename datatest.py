import numpy as np
import pandas as pd
#initial understanding of the data
df = pd.read_csv("churn.csv")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())


print("\nTarget distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

baseline_accuracy = df["Churn"].value_counts(normalize=True)["No"]
print("Baseline (always No) accuracy:", baseline_accuracy)



print("\nMissing values:")
#print(df.isna().sum())
print((df["TotalCharges"] == " ").sum())
print(df[df["tenure"] == 0])
print(df[df["tenure"] == 0][["tenure","TotalCharges","Churn"]]
)
#exploratory data analysis

import seaborn as sns
import matplotlib.pyplot as plt

#sns.boxplot(x="Churn", y="tenure", data=df)
#plt.title("Tenure vs Churn")
#plt.show()

#df.groupby("Churn")["tenure"].mean()

#sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
#plt.title("Monthly Charges vs Churn")
#plt.show()

#df.groupby("Churn")["MonthlyCharges"].mean()
#x=pd.crosstab(df["Contract"], df["Churn"], normalize="index")
#print(x)

df["Churn_binary"] = df["Churn"].map({"No": 0, "Yes": 1})
print(df[["Churn", "Churn_binary"]].head())

X = pd.get_dummies(
    df[["tenure", "MonthlyCharges", "Contract"]],
    drop_first=True
)

y = df["Churn_binary"]

print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(y_train.mean(), y_test.mean())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    eps = 1e-15
    return -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )

def train_logistic(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for _ in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)

        loss = compute_loss(y, y_hat)
        losses.append(loss)

        dw = (1 / n_samples) * (X.T @ (y_hat - y))
        db = np.mean(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return w, b, losses
w, b, losses = train_logistic(
    X_train_scaled,
    y_train.values,
    lr=0.1,
    epochs=1000
)
#import matplotlib.pyplot as plt

#plt.plot(losses)
#plt.xlabel("Epoch")
#plt.ylabel("Loss")
#plt.title("Training Loss")
#plt.show()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
import numpy as np

print(np.percentile(y_prob, [10, 50, 90]))


from sklearn.tree import DecisionTreeClassifier

tree_shallow = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_shallow.fit(X_train, y_train)

y_pred_shallow = tree_shallow.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print("Shallow Tree")
print(confusion_matrix(y_test, y_pred_shallow))
print(classification_report(y_test, y_pred_shallow))
tree_deep = DecisionTreeClassifier(
    random_state=42
)

tree_deep.fit(X_train, y_train)

y_pred_deep = tree_deep.predict(X_test)
print("Deep Tree")
print(confusion_matrix(y_test, y_pred_deep))
print(classification_report(y_test, y_pred_deep))
print("Train accuracy (deep):", tree_deep.score(X_train, y_train))
print("Test accuracy (deep):", tree_deep.score(X_test, y_test))
import pandas as pd

feature_importance = pd.Series(
    tree_deep.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print("Random Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

print("Train accuracy:", rf.score(X_train, y_train))
print("Test accuracy:", rf.score(X_test, y_test))

rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(rf_importance)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
from sklearn.metrics import confusion_matrix, classification_report

print("Boosting Model")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
from sklearn.metrics import confusion_matrix, classification_report

xgb_importance = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(xgb_importance)