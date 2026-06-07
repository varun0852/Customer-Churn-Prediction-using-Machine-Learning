import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model.json")
    return model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Customer Churn Predictor")
st.markdown("Predict whether a bank customer will churn using an **XGBoost** model trained with SMOTE balancing.")
st.divider()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=60000.0)
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    active_member = st.selectbox("Active Member?", ["Yes", "No"])
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

def preprocess(age, tenure, estimated_salary, products_number,
               credit_card, active_member, country, gender):

    # Encode
                   
    credit_card_val = 1 if credit_card == "Yes" else 0
    active_member_val = 1 if active_member == "Yes" else 0
    country_France = 1 if country == "France" else 0
    country_Germany = 1 if country == "Germany" else 0
    gender_Male = 1 if gender == "Male" else 0

    # Normalize age and estimated_salary same as training MinMaxScaler
                   
    age_min, age_max = 18, 92
    salary_min, salary_max = 11.58, 199992
    age_n = (age - age_min) / (age_max - age_min)
    salary_n = (estimated_salary - salary_min) / (salary_max - salary_min)

    # Exact 9 features the model was trained on — in exact order
                   
    input_data = pd.DataFrame([[
        age_n, salary_n, tenure, products_number,
        credit_card_val, active_member_val,
        country_France, country_Germany, gender_Male
    ]], columns=[
        "age", "estimated_salary", "tenure", "products_number",
        "credit_card", "active_member",
        "country_France", "country_Germany", "gender_Male"
    ])

    return input_data

if st.button("🔍 Predict Churn", use_container_width=True):
    model = load_model()
    input_df = preprocess(age, tenure, estimated_salary, products_number,
                          credit_card, active_member, country, gender)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Churn Risk — This customer is likely to leave")
    else:
        st.success("✅ Low Churn Risk — This customer is likely to stay")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stay Probability", f"{probability[0]*100:.1f}%")
    with col2:
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")

st.divider()
st.markdown("**Model:** XGBoost + SMOTE | **Built by:** [Varun](https://github.com/varun0852)")
