import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

print("Starting SHAP...")

# Load model
model = joblib.load("attrition_model_xgb_final.pkl")

# Load dataset
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
X_test = data.drop("Attrition", axis=1)

print("Data loaded:", X_test.shape)

# Sample (important)
X_sample = X_test.sample(50)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Create folder
os.makedirs("static", exist_ok=True)

# Plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("static/shap_summary.png")
plt.close()

print("DONE. Check static folder.")