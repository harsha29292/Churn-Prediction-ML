import pandas as pd
#initial understanding of the data
df = pd.read_csv("churn.csv")
x=pd.crosstab(df["Contract"], df["Churn"], normalize="index")
print(x)