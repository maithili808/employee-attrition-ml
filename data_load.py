print("Script started")
import pandas as pd

# load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# show first 5 rows
print(df.head())

# dataset info
print(df.info())
