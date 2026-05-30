"""
app.py  —  AttritionIQ Flask application
All prediction history is persisted to SQLite via database.py
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
import json
from datetime import datetime

from database import (
    init_db, verify_user,
    save_prediction,
    get_predictions, get_prediction_by_id, count_predictions,
    get_dashboard_stats, get_chart_data,
)

# ══════════════════════════════════════════════
# LOAD MODEL  (once at startup)
# ══════════════════════════════════════════════
saved     = joblib.load("attrition_model_xgb_final.pkl")
model     = saved["model"]       # sklearn Pipeline
threshold = saved["threshold"]

def clean_row(row):
    """Convert a DB row to a plain dict, dropping any binary/bytes fields."""
    result = {}
    for k, v in dict(row).items():
        if isinstance(v, bytes):
            try:
                result[k] = v.decode("utf-8")
            except Exception:
                result[k] = None  # skip undecodable binary (e.g. pickled SHAP data)
        else:
            result[k] = v
    return result

preprocessor = model.named_steps["preprocessor"]
xgb_clf      = model.named_steps["classifier"]
explainer    = shap.TreeExplainer(xgb_clf)


# ── SHAP helpers ──────────────────────────────
def get_feature_names(prep):
    num_features = list(prep.transformers_[0][2])
    ohe          = prep.transformers_[1][1]
    cat_features = list(ohe.get_feature_names_out(prep.transformers_[1][2]))
    return num_features + cat_features

def humanize(fname):
    return (fname
        .replace("OverTime_Yes",  "OverTime: Yes")
        .replace("OverTime_No",   "OverTime: No")
        .replace("_", " ")
    ).title()

def explain_prediction(raw_df):
    X_t = preprocessor.transform(raw_df)
    names = get_feature_names(preprocessor)
    sv_raw = explainer.shap_values(X_t)
    if isinstance(sv_raw, list):
        sv_raw = sv_raw[1]
    sv = sv_raw[0]

    results = []
    for fname, sval, fval in zip(names, sv, X_t[0]):
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
    max_abs = max(s["abs_shap"] for s in top) if top else 1
    for s in top:
        s["bar_pct"] = round(s["abs_shap"] / max_abs * 100, 1)
    return top


# ══════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "attritioniq-sqlite-2025"

with app.app_context():
    init_db()


# ── Auth ──────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pwd   = request.form.get("password", "")
        user  = verify_user(email, pwd)
        if user:
            session["user"] = {
                "email":    user["email"],
                "role":     user["role"],
                "initials": user["initials"],
            }
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    stats = get_dashboard_stats()

    # Convert any bytes fields in recent predictions
    recent_raw = get_predictions(limit=5)
    recent = [clean_row(r) for r in recent_raw]

    return render_template("dashboard.html",
        user=session["user"], page="dashboard",
        stats=stats,
        recent=recent,
        total_preds=stats["total"],
    )


# ── Prediction + SHAP ─────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction  = None
    probability = None
    risk_level  = None
    shap_data   = []
    form_data   = {}
    pred_id     = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            # Map new Department/JobRole values to closest model-known values
            dept_map = {
                "Customer Support":       "Sales",
                "Finance":                "Research & Development",
                "Information Technology": "Research & Development",
                "Legal":                  "Human Resources",
                "Marketing":              "Sales",
                "Operations":             "Research & Development",
                "Human Resources":        "Human Resources",
                "Research & Development": "Research & Development",
                "Sales":                  "Sales",
            }
            role_map = {
                "Accountant":             "Manager",
                "Data Analyst":           "Research Scientist",
                "Finance Analyst":        "Manager",
                "HR Manager":             "Human Resources",
                "IT Support":             "Laboratory Technician",
                "Operations Manager":     "Manager",
                "Software Engineer":      "Research Scientist",
                "Supply Chain Analyst":   "Research Scientist",
                # originals pass through unchanged
                "Healthcare Representative": "Healthcare Representative",
                "Human Resources":           "Human Resources",
                "Laboratory Technician":     "Laboratory Technician",
                "Manager":                   "Manager",
                "Manufacturing Director":    "Manufacturing Director",
                "Research Director":         "Research Director",
                "Research Scientist":        "Research Scientist",
                "Sales Executive":           "Sales Executive",
                "Sales Representative":      "Sales Representative",
            }

            dept     = form_data.get("Department", "Sales")
            job_role = form_data.get("JobRole", "Sales Executive")

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
                "MonthlyRate":              int(form_data.get("MonthlyRate", 14000)),
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
                "Department":               dept_map.get(dept, "Sales"),
                "EducationField":           form_data["EducationField"],
                "Gender":                   form_data["Gender"],
                "JobRole":                  role_map.get(job_role, "Sales Executive"),
                "MaritalStatus":            form_data["MaritalStatus"],
                "OverTime":                 form_data["OverTime"],
            }
            df = pd.DataFrame([data])

            probability = round(model.predict_proba(df)[0][1] * 100, 2)
            prediction  = "Yes — Likely to Leave" if probability / 100 >= threshold else "No — Likely to Stay"
            risk_level  = "High" if probability >= 70 else ("Medium" if probability >= 40 else "Low")

            shap_data = explain_prediction(df)

            pred_row = {
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M"),
                "employee_id":        form_data.get("employee_id") or None,
                "run_by":             session["user"]["email"],
                "department":         dept,           # store display value
                "job_role":           job_role,       # store display value
                "overtime":           data["OverTime"],
                "age":                data["Age"],
                "monthly_income":     data["MonthlyIncome"],
                "job_satisfaction":   data["JobSatisfaction"],
                "work_life_balance":  data["WorkLifeBalance"],
                "years_at_company":   data["YearsAtCompany"],
                "distance_from_home": data["DistanceFromHome"],
                "total_working_years":data["TotalWorkingYears"],
                "job_level":          data["JobLevel"],
                "marital_status":     data["MaritalStatus"],
                "business_travel":    data["BusinessTravel"],
                "probability":        probability,
                "risk_level":         risk_level,
                "prediction":         prediction,
            }
            pred_id = save_prediction(pred_row, shap_data, threshold)

        except Exception as e:
            import traceback
            prediction = f"Error: {e}"
            print(traceback.format_exc())

    return render_template("predict.html",
        user=session["user"], page="predict",
        prediction=prediction, probability=probability,
        risk_level=risk_level, shap_data=shap_data,
        form_data=form_data, pred_id=pred_id,
        total_preds=count_predictions(),
    )


# ── History ───────────────────────────────────
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    risk_filter = request.args.get("risk", "all")
    page        = max(1, int(request.args.get("page", 1)))
    per_page    = 20
    offset      = (page - 1) * per_page

    rows  = get_predictions(risk_filter=risk_filter, limit=per_page, offset=offset)
    total = count_predictions(risk_filter=risk_filter)
    pages = max(1, (total + per_page - 1) // per_page)

    # Convert any bytes fields to strings to avoid JSON serialization errors
    clean_rows = [clean_row(r) for r in rows]

    return render_template("history.html",
        user=session["user"], page="history",
        history=clean_rows, risk_filter=risk_filter,
        total=count_predictions(),
        filtered_total=total,
        current_page=page, total_pages=pages,
        total_preds=count_predictions(),
    )


# ── Prediction detail ─────────────────────────
@app.route("/prediction/<int:pred_id>")
def prediction_detail(pred_id):
    if "user" not in session:
        return redirect(url_for("login"))
    pred, shap_rows = get_prediction_by_id(pred_id)
    if not pred:
        return redirect(url_for("history"))
    shap_data = json.loads(pred["shap_json"]) if pred.get("shap_json") else []
    return render_template("detail.html",
        user=session["user"], page="history",
        pred=pred, shap_data=shap_data,
        total_preds=count_predictions(),
    )


# ── API endpoints ─────────────────────────────
@app.route("/api/chart-data")
def chart_data_api():
    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(get_chart_data())

@app.route("/api/stats")
def stats_api():
    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401
    s = get_dashboard_stats()
    return jsonify({
        "total": s["total"], "high": s["high"],
        "medium": s["medium"], "low": s["low"],
        "leave": s["leave"], "stay": s["stay"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


