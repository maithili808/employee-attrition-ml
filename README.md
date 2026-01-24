# Employee Attrition Prediction System

This project predicts whether an employee is likely to leave the company using Machine Learning.

## 🔍 Project Overview
Employee attrition is a major problem for organizations.  
This system uses an XGBoost machine learning model to predict employee attrition based on key employee attributes.

## 🛠 Tech Stack
- Python
- Flask
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- HTML (Frontend)
- Render (Deployment)

## 🚀 Live Demo
https://employee-attrition-ml.onrender.com

⚠️ Note: The app may take 30–60 seconds to load on first request because it is deployed on Render free tier.

## 📊 Model
- Algorithm: XGBoost Classifier
- Output:
  - Yes (Likely to Leave)
  - No (Likely to Stay)

## 🧠 Features Used
The model uses important employee-related features such as:
- Age
- Job Level
- Monthly Income
- Job Satisfaction
- Work-Life Balance
- Years at Company
- OverTime
- Department
- Job Role  
(and other relevant features)

## ⚙️ How to Run Locally
```bash
git clone https://github.com/maithili808/employee-attrition-ml.git
cd employee-attrition-ml
pip install -r requirements.txt
python app.py
