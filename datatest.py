import pandas as pd

df = pd.read_csv("churn.csv")

#print("Shape:", df.shape)
#print("\nColumns:\n", df.columns)
#print("\nInfo:\n")
#print(df.info())


#print("\nTarget distribution:")
#print(df["Churn"].value_counts())
#print(df["Churn"].value_counts(normalize=True))

#baseline_accuracy = df["Churn"].value_counts(normalize=True)["No"]
#print("Baseline (always No) accuracy:", baseline_accuracy)



print("\nMissing values:")
#print(df.isna().sum())
print((df["TotalCharges"] == " ").sum())
print(df[df["tenure"] == 0])
print(df[df["tenure"] == 0][["tenure","TotalCharges","Churn"]]
)
