from flask import Flask, render_template, request
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
saved = joblib.load("attrition_model_xgb_final.pkl")
model = saved["model"]
threshold = saved["threshold"]

app = Flask(__name__)

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    # ---- collect form data ----
    data = {
        "Age": int(request.form["Age"]),
        "DailyRate": int(request.form["DailyRate"]),
        "DistanceFromHome": int(request.form["DistanceFromHome"]),
        "Education": int(request.form["Education"]),
        "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),
        "HourlyRate": int(request.form["HourlyRate"]),
        "JobInvolvement": int(request.form["JobInvolvement"]),
        "JobLevel": int(request.form["JobLevel"]),
        "JobSatisfaction": int(request.form["JobSatisfaction"]),
        "MonthlyIncome": int(request.form["MonthlyIncome"]),
        "MonthlyRate": int(request.form["MonthlyRate"]),
        "NumCompaniesWorked": int(request.form["NumCompaniesWorked"]),
        "PercentSalaryHike": int(request.form["PercentSalaryHike"]),
        "PerformanceRating": int(request.form["PerformanceRating"]),
        "RelationshipSatisfaction": int(request.form["RelationshipSatisfaction"]),
        "StockOptionLevel": int(request.form["StockOptionLevel"]),
        "TotalWorkingYears": int(request.form["TotalWorkingYears"]),
        "TrainingTimesLastYear": int(request.form["TrainingTimesLastYear"]),
        "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
        "YearsAtCompany": int(request.form["YearsAtCompany"]),
        "YearsInCurrentRole": int(request.form["YearsInCurrentRole"]),
        "YearsSinceLastPromotion": int(request.form["YearsSinceLastPromotion"]),
        "YearsWithCurrManager": int(request.form["YearsWithCurrManager"]),

        "BusinessTravel": request.form["BusinessTravel"],
        "Department": request.form["Department"],
        "EducationField": request.form["EducationField"],
        "Gender": request.form["Gender"],
        "JobRole": request.form["JobRole"],
        "MaritalStatus": request.form["MaritalStatus"],
        "OverTime": request.form["OverTime"],

        # constant column required by pipeline
        "EmployeeCount": 1
    }

    df = pd.DataFrame([data])

    # ---- prediction ----
    probability = model.predict_proba(df)[0][1]
    prediction = "Yes (Likely to Leave)" if probability >= threshold else "No (Likely to Stay)"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(probability * 100, 2)
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=5000)
