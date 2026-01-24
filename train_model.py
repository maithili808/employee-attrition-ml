# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from xgboost import XGBClassifier

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# =========================
# 2. BASIC CLEANING
# =========================
# Drop unnecessary columns
data = data.drop(columns=["EmployeeNumber", "Over18", "StandardHours"])

# Encode target
data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})
target = "Attrition"

# =========================
# 3. FEATURE GROUPS
# =========================
categorical_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

numerical_cols = [col for col in data.columns if col not in categorical_cols + [target]]

X = data.drop(columns=[target])
y = data[target]

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 5. PREPROCESSING PIPELINE
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# =========================
# 6. HANDLE CLASS IMBALANCE
# =========================
# Compute scale_pos_weight for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# =========================
# 7. XGBOOST MODEL
# =========================
xgb_model = XGBClassifier(
    n_estimators=350,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

# Combine preprocessing + model in a pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# =========================
# 8. TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# 9. THRESHOLD OPTIMIZATION
# =========================
y_probs = model.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.1, 0.9, 0.01):
    preds = (y_probs >= t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

y_pred = (y_probs >= best_threshold).astype(int)

# =========================
# 10. EVALUATION
# =========================
print(f"\nOptimal Threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}\n")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 11. SAVE MODEL & THRESHOLD
# =========================
joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    "attrition_model_xgb_final.pkl"
)

print("\n✅ Final model saved as attrition_model_xgb_final.pkl")
