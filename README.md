# EMIPredict AI

A machine learning web app that predicts EMI eligibility and the maximum affordable monthly EMI for an applicant, based on their personal, employment, and expense details. Built with Python and deployed as a live multi-page Streamlit app.

🔗 Live App: https://devadharshini-emi-prediction.streamlit.app/
🔗 GitHub Repo: https://github.com/devadharshinimurugan06-commits/EMI-prediction-AI



##  About the Project

EMIPredict AI helps estimate whether a person is eligible for an EMI-based loan/purchase, and how much monthly EMI they can realistically afford — using their income, employment details, and existing monthly expenses (school/college fees, travel, groceries, etc).

Two ML models work together in the app:
1. Classification model → predicts EMI eligibility 
2. Regression model → predicts the maximum affordable monthly EMI amount

## Problem Statement
Build a comprehensive financial risk assessment platform that integrates machine learning models with MLflow experiment tracking to create an interactive web application for EMI prediction.
Nowadays, people struggle to pay EMI due to poor financial planning and inadequate risk assessment. This project aims to solve this critical issue by providing data-driven insights for better loan decisions.
The platform should deliver:
●	Dual ML problem solving: Classification (EMI eligibility) and Regression (maximum EMI amount)
●	Real-time financial risk assessment using 400,000 records
●	Advanced feature engineering from 22 financial and demographic variables
●	ML flow integration for model tracking and comparison
●	Streamlit Cloud deployment for production-ready access
●	Complete CRUD operations for financial data management

##  App Features (4 Sections)

The app has a navigation sidebar with four pages:

- Real-Time Prediction — enter applicant details (Personal, Employment & Income, Expenses & Credit) and instantly get the eligibility prediction (classification) and the maximum affordable EMI (regression)
- Data Exploration — 3 interactive charts/graphs and dataset
- Model Performance — evaluation metrics for both the classification and regression models
- Admin (CRUD) — update and delete rows in the underlying dataset directly from the app

# 🧠 Model Details

1. Task 1 - Eligibility prediction — Classification (best_classification_model.pk) |
2. Task 2 - Affordable EMI prediction — Regression (best_regression_model.pkl ) |
 Preprocessing | Label encoding (label_encoders.pkl) + feature scaling (manual_scaler.pkl) 
| Experiment tracking | MLflow (mlflow.db) |
Results.

Both classification model and regression models - actual accuracy / R² score here — Classification accuracy: 97.25% | Regression R²: 0.97

##  Tech Stack

- Python
- Pandas / NumPy — data handling
- Scikit-learn** — model building (classification + regression)
- MLflow — experiment tracking and model comparison
- Streamlit — multi-page web app framework & deployment
- Google Colab — model training environment

## Project Structure
Dataset (400K Records) 
        ↓
Data Quality Assessment & Preprocessing
        ↓  
Feature Engineering & Exploratory Analysis
        ↓
ML Model Training & MLflow Tracking
        ↓
Model Evaluation & Selection
        ↓
Streamlit Application Development
        ↓
Cloud Deployment & Performance Testing


##  What I Learned

- Building and comparing both classification and regression models for a single real-world problem
- Handling categorical features with label encoding and numerical features with scaling
- Tracking and comparing model experiments using **MLflow**
- Building a multi-page Streamlit app with real-time prediction, data exploration, model evaluation, and an admin CRUD panel

##  Future Improvements

- Add SHAP/feature-importance explanations for each prediction
- Add user authentication for the Admin panel
- Try additional models (XGBoost, Gradient Boosting) and compare in MLflow

⭐ If you found this project useful, consider giving it a star on GitHub!
