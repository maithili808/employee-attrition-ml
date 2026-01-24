import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Total rows & columns:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nTarget column value counts:\n")
print(df["Attrition"].value_counts())
