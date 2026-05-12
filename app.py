from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import joblib
from datetime import datetime
import json, os

# =========================
# LOAD MODEL
# =========================
saved = joblib.load("attrition_model_xgb_final.pkl")
model  = saved["model"]
threshold = saved["threshold"]

app = Flask(__name__)
app.secret_key = "attritioniq-secret-2025"

# In-memory prediction history (replace with DB in production)
PREDICTION_HISTORY = []

# =========================
# MOCK USERS (replace with DB)
# =========================
USERS = {
    "hr@company.com":      {"password": "password", "role": "HR Manager",  "initials": "HR"},
    "analyst@company.com": {"password": "password", "role": "Analyst",     "initials": "AN"},
    "admin@company.com":   {"password": "password", "role": "Admin",       "initials": "AD"},
}

# =========================
# AUTH ROUTES
# =========================
@app.route("/", methods=["GET"])
def root():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

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
# PREDICTION FORM
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    probability = None
    risk_level  = None
    form_data   = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            data = {
                "Age":                      int(form_data["Age"]),
                "DailyRate":               int(form_data["DailyRate"]),
                "DistanceFromHome":        int(form_data["DistanceFromHome"]),
                "Education":               int(form_data["Education"]),
                "EnvironmentSatisfaction": int(form_data["EnvironmentSatisfaction"]),
                "HourlyRate":              int(form_data["HourlyRate"]),
                "JobInvolvement":          int(form_data["JobInvolvement"]),
                "JobLevel":                int(form_data["JobLevel"]),
                "JobSatisfaction":         int(form_data["JobSatisfaction"]),
                "MonthlyIncome":           int(form_data["MonthlyIncome"]),
                "MonthlyRate":             int(form_data["MonthlyRate"]),
                "NumCompaniesWorked":      int(form_data["NumCompaniesWorked"]),
                "PercentSalaryHike":       int(form_data["PercentSalaryHike"]),
                "PerformanceRating":       int(form_data["PerformanceRating"]),
                "RelationshipSatisfaction":int(form_data["RelationshipSatisfaction"]),
                "StockOptionLevel":        int(form_data["StockOptionLevel"]),
                "TotalWorkingYears":       int(form_data["TotalWorkingYears"]),
                "TrainingTimesLastYear":   int(form_data["TrainingTimesLastYear"]),
                "WorkLifeBalance":         int(form_data["WorkLifeBalance"]),
                "YearsAtCompany":          int(form_data["YearsAtCompany"]),
                "YearsInCurrentRole":      int(form_data["YearsInCurrentRole"]),
                "YearsSinceLastPromotion": int(form_data["YearsSinceLastPromotion"]),
                "YearsWithCurrManager":    int(form_data["YearsWithCurrManager"]),
                "BusinessTravel":          form_data["BusinessTravel"],
                "Department":              form_data["Department"],
                "EducationField":          form_data["EducationField"],
                "Gender":                  form_data["Gender"],
                "JobRole":                 form_data["JobRole"],
                "MaritalStatus":           form_data["MaritalStatus"],
                "OverTime":                form_data["OverTime"],
                "EmployeeCount":           1,
            }
            df = pd.DataFrame([data])
            probability = round(model.predict_proba(df)[0][1] * 100, 2)
            prediction  = "Yes — Likely to Leave" if probability/100 >= threshold else "No — Likely to Stay"
            risk_level  = "High" if probability >= 70 else ("Medium" if probability >= 40 else "Low")

            # Save to history
            PREDICTION_HISTORY.append({
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                "employee_id": form_data.get("employee_id", "N/A"),
                "department":  data["Department"],
                "job_role":    data["JobRole"],
                "prediction":  prediction,
                "probability": probability,
                "risk":        risk_level,
                "overtime":    data["OverTime"],
                "satisfaction":data["JobSatisfaction"],
                "run_by":      session["user"]["email"],
            })
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("predict.html",
        user=session["user"], page="predict",
        prediction=prediction, probability=probability,
        risk_level=risk_level, form_data=form_data)

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
# API: chart data
# =========================
@app.route("/api/chart-data")
def chart_data():
    from collections import Counter
    dept_counts  = Counter(p["department"] for p in PREDICTION_HISTORY)
    risk_counts  = Counter(p["risk"] for p in PREDICTION_HISTORY)
    return jsonify({
        "departments": dict(dept_counts),
        "risk_dist":   {"High": risk_counts.get("High",0),
                        "Medium": risk_counts.get("Medium",0),
                        "Low": risk_counts.get("Low",0)},
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
