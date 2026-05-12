from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
saved     = joblib.load("attrition_model_xgb_final.pkl")
model     = saved["model"]          # sklearn Pipeline
threshold = saved["threshold"]

# ── Extract pieces needed for SHAP ───────────────────────────────────
# Pipeline steps: "preprocessor" (ColumnTransformer) + "classifier" (XGBClassifier)
preprocessor = model.named_steps["preprocessor"]
xgb_clf      = model.named_steps["classifier"]

# Build SHAP TreeExplainer once at startup — fast, exact for XGBoost
explainer = shap.TreeExplainer(xgb_clf)

def get_feature_names(prep):
    """Reconstruct output feature names from ColumnTransformer (num + OHE cat)."""
    num_features = list(prep.transformers_[0][2])          # numerical col names
    ohe          = prep.transformers_[1][1]                # OneHotEncoder
    cat_features = list(ohe.get_feature_names_out(
        prep.transformers_[1][2]                           # categorical col names
    ))
    return num_features + cat_features

def humanize(fname):
    """Turn a pipeline feature name into a readable label."""
    label = (fname
        .replace("OverTime_Yes",  "OverTime: Yes")
        .replace("OverTime_No",   "OverTime: No")
        .replace("_", " ")
    )
    # Remove OHE prefixes like "x0 " or "BusinessTravel "
    # Get just the last meaningful chunk for OHE columns
    parts = label.split()
    return " ".join(parts).title()

def explain_prediction(raw_df):
    """
    Given a 1-row raw DataFrame (before preprocessing), return a list of dicts
    sorted by |SHAP value| descending — top 12 only.
    """
    X_transformed = preprocessor.transform(raw_df)
    feature_names = get_feature_names(preprocessor)

    # shap_values: (n_samples, n_features) for binary XGBoost
    sv_raw = explainer.shap_values(X_transformed)
    if isinstance(sv_raw, list):
        sv_raw = sv_raw[1]   # take class=1 (Attrition=Yes)
    sv = sv_raw[0]

    results = []
    for fname, sval, fval in zip(feature_names, sv, X_transformed[0]):
        results.append({
            "feature":    humanize(fname),
            "raw_name":   fname,
            "shap_value": float(round(sval, 4)),
            "abs_shap":   float(round(abs(sval), 4)),
            "raw_value":  float(round(fval, 3)),
            "direction":  "risk" if sval > 0 else "safe",
        })

    results.sort(key=lambda x: x["abs_shap"], reverse=True)
    top = results[:12]

    # Normalise bar widths
    max_abs = max(s["abs_shap"] for s in top) if top else 1
    for s in top:
        s["bar_pct"] = round(s["abs_shap"] / max_abs * 100, 1)

    return top

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.secret_key = "attritioniq-shap-2025"

PREDICTION_HISTORY = []

USERS = {
    "hr@company.com":      {"password": "password", "role": "HR Manager",  "initials": "HR"},
    "analyst@company.com": {"password": "password", "role": "Analyst",     "initials": "AN"},
    "admin@company.com":   {"password": "password", "role": "Admin",       "initials": "AD"},
}

# =========================
# AUTH
# =========================
@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pwd   = request.form.get("password", "")
        user  = USERS.get(email)
        if user and user["password"] == pwd:
            session["user"] = {"email": email, "role": user["role"], "initials": user["initials"]}
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    recent = PREDICTION_HISTORY[-5:][::-1]
    high   = sum(1 for p in PREDICTION_HISTORY if p["risk"] == "High")
    med    = sum(1 for p in PREDICTION_HISTORY if p["risk"] == "Medium")
    low    = sum(1 for p in PREDICTION_HISTORY if p["risk"] == "Low")
    return render_template("dashboard.html",
        user=session["user"], page="dashboard",
        total_preds=len(PREDICTION_HISTORY),
        high_risk=high, med_risk=med, low_risk=low,
        recent=recent)

# =========================
# PREDICT + SHAP
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction  = None
    probability = None
    risk_level  = None
    shap_data   = []
    form_data   = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            data = {
                "Age":                      int(form_data["Age"]),
                "DailyRate":                int(form_data["DailyRate"]),
                "DistanceFromHome":         int(form_data["DistanceFromHome"]),
                "Education":                int(form_data["Education"]),
                "EmployeeCount":            1,
                "EnvironmentSatisfaction":  int(form_data["EnvironmentSatisfaction"]),
                "HourlyRate":               int(form_data["HourlyRate"]),
                "JobInvolvement":           int(form_data["JobInvolvement"]),
                "JobLevel":                 int(form_data["JobLevel"]),
                "JobSatisfaction":          int(form_data["JobSatisfaction"]),
                "MonthlyIncome":            int(form_data["MonthlyIncome"]),
                "MonthlyRate":              int(form_data["MonthlyRate"]),
                "NumCompaniesWorked":       int(form_data["NumCompaniesWorked"]),
                "PercentSalaryHike":        int(form_data["PercentSalaryHike"]),
                "PerformanceRating":        int(form_data["PerformanceRating"]),
                "RelationshipSatisfaction": int(form_data["RelationshipSatisfaction"]),
                "StockOptionLevel":         int(form_data["StockOptionLevel"]),
                "TotalWorkingYears":        int(form_data["TotalWorkingYears"]),
                "TrainingTimesLastYear":    int(form_data["TrainingTimesLastYear"]),
                "WorkLifeBalance":          int(form_data["WorkLifeBalance"]),
                "YearsAtCompany":           int(form_data["YearsAtCompany"]),
                "YearsInCurrentRole":       int(form_data["YearsInCurrentRole"]),
                "YearsSinceLastPromotion":  int(form_data["YearsSinceLastPromotion"]),
                "YearsWithCurrManager":     int(form_data["YearsWithCurrManager"]),
                "BusinessTravel":           form_data["BusinessTravel"],
                "Department":               form_data["Department"],
                "EducationField":           form_data["EducationField"],
                "Gender":                   form_data["Gender"],
                "JobRole":                  form_data["JobRole"],
                "MaritalStatus":            form_data["MaritalStatus"],
                "OverTime":                 form_data["OverTime"],
            }
            df = pd.DataFrame([data])

            # Predict
            probability = round(model.predict_proba(df)[0][1] * 100, 2)
            prediction  = "Yes — Likely to Leave" if probability / 100 >= threshold else "No — Likely to Stay"
            risk_level  = "High" if probability >= 70 else ("Medium" if probability >= 40 else "Low")

            # SHAP
            shap_data = explain_prediction(df)

            # Save history
            PREDICTION_HISTORY.append({
                "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "employee_id": form_data.get("employee_id", "N/A"),
                "department":  data["Department"],
                "job_role":    data["JobRole"],
                "prediction":  prediction,
                "probability": probability,
                "risk":        risk_level,
                "overtime":    data["OverTime"],
                "satisfaction":data["JobSatisfaction"],
                "run_by":      session["user"]["email"],
                "top_reason":  shap_data[0]["feature"] if shap_data else "—",
            })

        except Exception as e:
            import traceback
            prediction = f"Error: {e}"
            print(traceback.format_exc())

    return render_template("predict.html",
        user=session["user"], page="predict",
        prediction=prediction, probability=probability,
        risk_level=risk_level, shap_data=shap_data,
        form_data=form_data)

# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    risk_filter = request.args.get("risk", "all")
    data = PREDICTION_HISTORY[::-1]
    if risk_filter != "all":
        data = [p for p in data if p["risk"].lower() == risk_filter.lower()]
    return render_template("history.html",
        user=session["user"], page="history",
        history=data, risk_filter=risk_filter,
        total=len(PREDICTION_HISTORY))

# =========================
# API
# =========================
@app.route("/api/chart-data")
def chart_data():
    from collections import Counter
    dept_counts = Counter(p["department"] for p in PREDICTION_HISTORY)
    risk_counts = Counter(p["risk"] for p in PREDICTION_HISTORY)
    return jsonify({
        "departments": dict(dept_counts),
        "risk_dist": {
            "High":   risk_counts.get("High", 0),
            "Medium": risk_counts.get("Medium", 0),
            "Low":    risk_counts.get("Low", 0),
        },
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)))
