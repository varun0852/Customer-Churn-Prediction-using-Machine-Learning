import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and vectorizer

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Customer Churn Predictor")
st.markdown("Predict whether a bank customer will churn using a **Tuned Random Forest** model (GridSearchCV, F1: 86.61%)")
st.divider()

# Input form

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0)

with col2:
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
    credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    active_member = st.selectbox("Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=60000.0)

country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

# Preprocess input

def preprocess(credit_score, age, tenure, balance, products_number,
               credit_card, active_member, estimated_salary, country, gender):
    from sklearn.preprocessing import MinMaxScaler

    credit_card = 1 if credit_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0
    country_France = 1 if country == "France" else 0
    country_Germany = 1 if country == "Germany" else 0
    country_Spain = 1 if country == "Spain" else 0
    gender_Female = 1 if gender == "Female" else 0
    gender_Male = 1 if gender == "Male" else 0

    # Use the same 7 features selected by Sequential Feature Selector
    # Features: age, estimated_salary, tenure, products_number, active_member, country_Germany, gender_Female


    input_data = pd.DataFrame([[age, estimated_salary, tenure, products_number,
                                 active_member, country_Germany, gender_Female]],
                               columns=["age", "estimated_salary", "tenure", "products_number",
                                        "active_member", "country_Germany", "gender_Female"])
    return input_data

if st.button("🔍 Predict Churn", use_container_width=True):
    try:
        model = load_model()
        input_df = preprocess(credit_score, age, tenure, balance, products_number,
                              credit_card, active_member, estimated_salary, country, gender)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.divider()

        if prediction == 1:
            st.error("⚠️ High Churn Risk — This customer is likely to leave")
            st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        else:
            st.success("✅ Low Churn Risk — This customer is likely to stay")
            st.metric("Retention Probability", f"{probability[0]*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stay Probability", f"{probability[0]*100:.1f}%")
        with col2:
            st.metric("Churn Probability", f"{probability[1]*100:.1f}%")

    except FileNotFoundError:
        st.warning("⚠️ model.pkl not found. Please export and save your trained model first (see instructions below).")
        st.code("""
# Run this in your Colab notebook to save the model:
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(best_rf_tuned, f)
# Then download model.pkl from Colab files panel
        """)

st.divider()
st.markdown("**Model:** Tuned Random Forest | **Best CV F1:** 86.61% | **Built by:** [Varun](https://github.com/varun0852)")
