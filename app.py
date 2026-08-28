import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EMIPredict AI", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/devadharshinimurugan06-commits/EMI-prediction-AI/main"


# =========================================================
# CACHED LOADERS
# =========================================================
@st.cache_resource
def load_pickle(filename):
    url = f"{GITHUB_BASE}/{filename}"
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

@st.cache_data
def load_dataset():
    return pd.read_csv(f"{GITHUB_BASE}/emi_prediction_dataset.csv", low_memory=False)

@st.cache_data
def load_results_csv(filename):
    return pd.read_csv(f"{GITHUB_BASE}/{filename}", index_col=0)


try:
    clf_pipeline = load_pickle("best_classification_model.pkl")
    reg_pipeline = load_pickle("best_regression_model.pkl")
    class_mapping = load_pickle("classification_class_mapping.pkl")
    label_encoders = load_pickle("label_encoders.pkl")
    manual_scaler = load_pickle("manual_scaler.pkl")
except Exception as e:
    st.error(f"Failed to load models from GitHub. ({e})")
    st.stop()

NUMERIC_COLS_TO_SCALE = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
    'dependents', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'debt_to_income_ratio', 'total_monthly_expenses', 'expense_to_income_ratio',
    'disposable_income', 'affordability_ratio', 'risk_score',
    'salary_per_dependent', 'loan_to_salary_ratio'
]


# =========================================================
# FEATURE PIPELINE — mirrors the notebook exactly
# =========================================================
def build_features(raw_input: dict):
    data = pd.DataFrame([raw_input])

    # --- Engineered ratios (same formulas as notebook) ---
    data['debt_to_income_ratio'] = data['current_emi_amount'] / (data['monthly_salary'] + 1)
    data['total_monthly_expenses'] = (
        data['monthly_rent'] + data['school_fees'] + data['college_fees'] +
        data['travel_expenses'] + data['groceries_utilities'] +
        data['other_monthly_expenses'] + data['current_emi_amount']
    )
    data['expense_to_income_ratio'] = data['total_monthly_expenses'] / (data['monthly_salary'] + 1)
    data['disposable_income'] = data['monthly_salary'] - data['total_monthly_expenses']
    data['affordability_ratio'] = data['disposable_income'] / (data['requested_amount'] / data['requested_tenure'] + 1)

    credit_score_norm = (data['credit_score'] - 300) / (850 - 300)
    employment_stability = data['years_of_employment'] / 40  # approx normalization
    data['risk_score'] = (
        (1 - credit_score_norm) * 0.4 +
        data['debt_to_income_ratio'].clip(0, 1) * 0.4 +
        (1 - employment_stability) * 0.2
    )

    data['salary_per_dependent'] = data['monthly_salary'] / (data['dependents'] + 1)
    data['loan_to_salary_ratio'] = data['requested_amount'] / (data['monthly_salary'] + 1)

    # --- Log-transformed columns (computed from RAW values, before scaling) ---
    data['monthly_salary_log'] = np.log1p(data['monthly_salary'])
    data['bank_balance_log'] = np.log1p(data['bank_balance'])
    data['requested_amount_log'] = np.log1p(data['requested_amount'])

    # --- Categorical encoding ---
    for col, le in label_encoders.items():
        data[col + '_encoded'] = le.transform(data[col])

    # --- Apply the SAME manual scaler used in training (in place) ---
    data[NUMERIC_COLS_TO_SCALE] = manual_scaler.transform(data[NUMERIC_COLS_TO_SCALE])

    return data


def predict_all(raw_input: dict):
    features = build_features(raw_input)

    # ---- STEP 1: Regression features (44 columns, no max_monthly_emi) ----
    reg_feature_cols = [
        'age', 'gender', 'marital_status', 'education', 'monthly_salary', 'employment_type',
        'years_of_employment', 'company_type', 'house_type', 'monthly_rent', 'family_size',
        'dependents', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
        'other_monthly_expenses', 'existing_loans', 'current_emi_amount', 'credit_score',
        'bank_balance', 'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure',
        'gender_encoded', 'marital_status_encoded', 'education_encoded', 'employment_type_encoded',
        'company_type_encoded', 'house_type_encoded', 'existing_loans_encoded', 'emi_scenario_encoded',
        'debt_to_income_ratio', 'total_monthly_expenses', 'expense_to_income_ratio',
        'disposable_income', 'affordability_ratio', 'risk_score', 'salary_per_dependent',
        'loan_to_salary_ratio', 'monthly_salary_log', 'bank_balance_log', 'requested_amount_log'
    ]

    reg_input = features[reg_feature_cols]
    predicted_emi = reg_pipeline.predict(reg_input)[0]
    predicted_emi = max(predicted_emi, 0)

    # ---- STEP 2: Add predicted EMI + its log as extra features for classification ----
    features['max_monthly_emi'] = predicted_emi
    features['max_monthly_emi_log'] = np.log1p(predicted_emi)

    clf_feature_cols = reg_feature_cols + ['max_monthly_emi', 'max_monthly_emi_log']
    clf_input = features[clf_feature_cols]

    clf_pred = clf_pipeline.predict(clf_input)[0]
    clf_label = class_mapping[clf_pred]

    return clf_label, predicted_emi


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("EMIPredict AI")
page = st.sidebar.radio("Navigate", [
    "🏠 Real-Time Prediction",
    "📊 Data Exploration",
    "📈 Model Performance",
    "🗂️ Admin (CRUD)"
])


# =========================================================
# PAGE 1: REAL-TIME PREDICTION
# =========================================================
if page == "🏠 Real-Time Prediction":
    st.title("EMI Eligibility & Affordability Predictor")
    st.write("Enter applicant details to predict EMI eligibility and maximum affordable monthly EMI.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal")
        age = st.number_input("Age", 18, 70, 35)
        gender = st.selectbox("Gender", label_encoders['gender'].classes_)
        marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)
        education = st.selectbox("Education", label_encoders['education'].classes_)
        dependents = st.number_input("Dependents", 0, 10, 1)
        family_size = st.number_input("Family Size", 1, 15, 3)

    with col2:
        st.subheader("Employment & Income")
        monthly_salary = st.number_input("Monthly Salary (INR)", 5000, 500000, 50000, step=1000)
        employment_type = st.selectbox("Employment Type", label_encoders['employment_type'].classes_)
        years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0)
        company_type = st.selectbox("Company Type", label_encoders['company_type'].classes_)
        house_type = st.selectbox("House Type", label_encoders['house_type'].classes_)
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100000, 0, step=500)

    with col3:
        st.subheader("Expenses & Credit")
        school_fees = st.number_input("School Fees (INR)", 0, 50000, 0, step=500)
        college_fees = st.number_input("College Fees (INR)", 0, 100000, 0, step=500)
        travel_expenses = st.number_input("Travel Expenses (INR)", 0, 30000, 3000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (INR)", 0, 50000, 8000, step=500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", 0, 50000, 2000, step=500)
        existing_loans = st.selectbox("Existing Loans", label_encoders['existing_loans'].classes_)
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100000, 0, step=500)
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        bank_balance = st.number_input("Bank Balance (INR)", 0, 5000000, 100000, step=1000)
        emergency_fund = st.number_input("Emergency Fund (INR)", 0, 2000000, 50000, step=1000)

    st.subheader("Loan Request")
    c1, c2, c3 = st.columns(3)
    with c1:
        emi_scenario = st.selectbox("EMI Scenario", label_encoders['emi_scenario'].classes_)
    with c2:
        requested_amount = st.number_input("Requested Amount (INR)", 5000, 2000000, 100000, step=1000)
    with c3:
        requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 24)

    if st.button("Predict", type="primary"):
        raw_input = dict(
            age=age, gender=gender, marital_status=marital_status, education=education,
            monthly_salary=monthly_salary, employment_type=employment_type,
            years_of_employment=years_of_employment, company_type=company_type,
            house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
            dependents=dependents, school_fees=school_fees, college_fees=college_fees,
            travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
            other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
            current_emi_amount=current_emi_amount, credit_score=credit_score,
            bank_balance=bank_balance, emergency_fund=emergency_fund,
            emi_scenario=emi_scenario, requested_amount=requested_amount,
            requested_tenure=requested_tenure
        )

        try:
            clf_label, reg_pred = predict_all(raw_input)
        except Exception as e:
            st.error(f"Something went wrong while predicting. Please check inputs. ({e})")
            st.stop()

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            if clf_label == "Eligible":
                st.success(f"### Eligibility: {clf_label}")
            elif clf_label == "High_Risk":
                st.warning(f"### Eligibility: {clf_label}")
            else:
                st.error(f"### Eligibility: {clf_label}")
        with r2:
            st.info(f"### Max Affordable Monthly EMI: ₹{reg_pred:,.0f}")

        # Log this prediction into the admin CSV
        log_row = raw_input.copy()
        log_row['predicted_eligibility'] = clf_label
        log_row['predicted_max_emi'] = reg_pred
        log_df = pd.DataFrame([log_row])
        log_file = 'prediction_log.csv'
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)


# =========================================================
# PAGE 2: DATA EXPLORATION
# =========================================================
elif page == "📊 Data Exploration":
    st.title("Data Exploration")
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Could not load the dataset from GitHub. ({e})")
        st.stop()

    st.write(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("EMI Eligibility Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='emi_eligibility', order=['Eligible', 'High_Risk', 'Not_Eligible'],
                      palette=['#FF1493', '#DB7093', '#FFC0CB'], ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['monthly_salary'], bins=30, color='#FF69B4', kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("EMI Scenario Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='emi_scenario', palette='PiYG', ax=ax)
    st.pyplot(fig)


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================
elif page == "📈 Model Performance":
    st.title("Model Performance (MLflow Results)")

    try:
        st.subheader("Classification Models")
        clf_results = load_results_csv("classification_results.csv")
        st.dataframe(clf_results)
        if 'Test Accuracy' in clf_results.columns:
            st.bar_chart(clf_results[['Test Accuracy']])

        st.subheader("Regression Models")
        reg_results = load_results_csv("regression_results.csv")
        st.dataframe(reg_results)
        if 'Test RMSE' in reg_results.columns:
            st.bar_chart(reg_results[['Test RMSE']])
    except Exception as e:
        st.error(f"Could not load model performance results from GitHub. ({e})")


# =========================================================
# PAGE 4: ADMIN (CRUD)
# =========================================================
elif page == "🗂️ Admin (CRUD)":
    st.title("Admin Panel — Prediction Records (CRUD)")
    log_file = 'prediction_log.csv'

    if os.path.exists(log_file):
        records = pd.read_csv(log_file)
    else:
        records = pd.DataFrame()

    st.subheader("Read: All Prediction Records")
    if records.empty:
        st.info("No records yet. Make a prediction on the Real-Time Prediction page first.")
    else:
        st.dataframe(records)

        st.subheader("Delete a Record")
        row_to_delete = st.number_input("Row index to delete", 0, max(len(records) - 1, 0), 0)
        if st.button("Delete Row"):
            records = records.drop(index=row_to_delete).reset_index(drop=True)
            records.to_csv(log_file, index=False)
            st.success(f"Row {row_to_delete} deleted.")
            st.rerun()

        st.subheader("Update a Record's Requested Amount")
        row_to_update = st.number_input("Row index to update", 0, max(len(records) - 1, 0), 0, key="upd")
        new_amount = st.number_input("New Requested Amount (INR)", 5000, 2000000, 100000, step=1000)
        if st.button("Update Row"):
            records.loc[row_to_update, 'requested_amount'] = new_amount
            records.to_csv(log_file, index=False)
            st.success(f"Row {row_to_update} updated.")
            st.rerun()

    
# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px;
        margin-top: 50px;
        border-top: 1px solid #444;
        color: #AAAAAA;
        font-size: 14px;
    ">
        <b>EMIPredict AI</b><br>
        Built by <b>Devadharshini</b>
    </div>
    """,
    unsafe_allow_html=True
)
