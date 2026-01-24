import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("attrition_model_xgb_final.pkl")

# Load dataset
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Target
target = "Attrition"
y = data[target].map({"Yes": 1, "No": 0})
X = data.drop(columns=[target])

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 1️⃣ Global feature importance
shap.summary_plot(shap_values, X)

# 2️⃣ Bar plot (clean for reports)
shap.summary_plot(shap_values, X, plot_type="bar")

# 3️⃣ Single prediction explanation
shap.plots.waterfall(shap_values[0])
