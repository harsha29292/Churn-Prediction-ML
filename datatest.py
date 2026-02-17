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
x=pd.crosstab(df["Contract"], df["Churn"], normalize="index")
print(x)

