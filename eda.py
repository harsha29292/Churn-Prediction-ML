import pandas as pd
import numpy as np


df = pd.read_csv('churn.csv')
# Force numeric conversion (robust)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check how many became NaN
#print(df["TotalCharges"].isna().sum())

df["TotalCharges"]=df["TotalCharges"].fillna(0)
#print(df.dtypes)
#print(df["TotalCharges"].isna().sum())

df["avg_monthly_spend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, df["tenure"].max()],
    labels=["new", "mid", "long"]
)
df["is_long_contract"] = np.where(
    df["Contract"] == "Month-to-month",
    0,
    1
)

#print(df[["tenure", "tenure_group", "avg_monthly_spend", "is_long_contract"]].head())

tenure_summary = (
    df
    .groupby("tenure_group")
    .agg(
        avg_spend_mean=("avg_monthly_spend", "mean"),
        avg_spend_std=("avg_monthly_spend", "std"),
        customer_count=("customerID", "count")
    )
)

#print(tenure_summary)


contract_summary = (
    df
    .groupby("is_long_contract")
    .agg(
        tenure_mean=("tenure", "mean"),
        monthly_charge_mean=("MonthlyCharges", "mean"),
        count=("customerID", "count")
    )
)

#print(contract_summary)

features=df[["avg_monthly_spend", "tenure","MonthlyCharges" ]]
X=features.to_numpy()

mean = X.mean(axis=0)
std = X.std(axis=0)
x_normalized = (X - mean) / std
print(x_normalized.mean(axis=0))
print(x_normalized.std(axis=0))
numeric_cols = ["avg_monthly_spend", "tenure", "MonthlyCharges", "TotalCharges"]

print(df[numeric_cols].describe())
