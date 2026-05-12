import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import numpy as np

print("Starting SHAP...")

# -----------------------------
# Load model (pipeline)
# -----------------------------
loaded = joblib.load("attrition_model_xgb_final.pkl")
pipeline = loaded["model"]

# Extract model + preprocessor
model = list(pipeline.named_steps.values())[-1]
preprocessor = list(pipeline.named_steps.values())[0]

print("Using model:", type(model))
print("Using preprocessor:", type(preprocessor))

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
X = data.drop("Attrition", axis=1)

# -----------------------------
# Apply preprocessing
# -----------------------------
X_processed = preprocessor.transform(X)

print("Processed shape:", X_processed.shape)

# -----------------------------
# Take sample (for speed)
# -----------------------------
idx = np.random.choice(X_processed.shape[0], 50, replace=False)
X_sample = X_processed[idx]

# -----------------------------
# SHAP calculation
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Create static folder
os.makedirs("static", exist_ok=True)

# -----------------------------
# 1. Summary Plot
# -----------------------------
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("static/shap_summary.png")
plt.close()

# -----------------------------
# 2. Bar Plot (Feature Importance)
# -----------------------------
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.savefig("static/shap_bar.png")
plt.close()

# -----------------------------
# 3. Force Plot (Single Prediction)
# -----------------------------
try:
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_sample[0]
    )
except:
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0][0],
        X_sample[0]
    )

shap.save_html("static/shap_force.html", force_plot)

print("DONE ✅ Check static folder")