import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Smart Credit Risk Prediction",
    page_icon=":credit_card:",
    layout="wide",
    )

model = pickle.load(open("loan.pkl", "rb"))

st.title(":credit_card: Smart Credit Risk Prediction System")
st.markdown("### Predict whether a loan applicant is likely to default or not based on their financial and personal information.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income :dollar:", min_value=0, value=50000)
    person_home_ownership = st.selectbox("Home Ownership",["Rent", "Own", "Mortage", "Other"]) 
    person_emp_length = st.number_input("Employment Length")
    loan_intent = st.selectbox("Loan Intent", ["Personal", "Education","Medical", "Home Improvement", "Debt Consolidation"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

with col2:
    loan_amount = st.number_input("Loan Amount :dollar:")
    loan_int_rate = st.number_input("Loan Interest Rate (%)")
    loan_percent_income = st.number_input("Loan Percent of Income (%)")
    cb_person_default_on_file = st.selectbox("Default on File", ["No", "Yes"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)",value=0)

st.markdown("---")

home_map = {"Rent": 0, "Own": 1, "Mortage": 2, "Other": 3}
loan_intent_map = {"Personal": 0, "Education": 1, "Medical": 2, "Home Improvement": 3, "Debt Consolidation": 4}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"No": 0, "Yes": 1}

if st.button(":credit_card: Predict Credit Risk"):
    input_data = np.array([[person_age, person_income, home_map[person_home_ownership], person_emp_length, loan_intent_map[loan_intent], grade_map[loan_grade], loan_amount, loan_int_rate, loan_percent_income, default_map[cb_person_default_on_file], cb_person_cred_hist_length]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The applicant is likely to default on the loan.")
    else:
        st.success("The applicant is unlikely to default on the loan.")