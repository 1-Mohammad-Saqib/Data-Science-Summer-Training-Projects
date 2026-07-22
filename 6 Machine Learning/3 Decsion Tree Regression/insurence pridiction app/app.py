import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("insurance_model.pkl","rb")

st.set_page_config(
    page_title="Insurance Prediction",
    layout = "centered"
)

st.title("Insurance Prediction App")
st.write("This app predicts the insurance charges based on user input.")

age = st.number_input("Age", min_value = 18, max_value = 100, value = 25)

bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 25.0)

children = st.number_input("Children", min_value = 0, max_value = 10, value = 0)

sex = st.selectbox("Sex", ["Female","Male"])

smoker = st.selectbox("Smoker", ["No","Yes"])

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

if st.button("Predict Insurance Cost"):
    if sex == "Male":
        sex = 1
    else:
        sex = 0

    if smoker == "Yes":
        smoker = 1
    else:
        smoker = 0

    region_dict = {
        "Northeast": 0,
        "Northwest": 1,
        "Southeast": 2,
        "Southwest": 3
    }
    region = region_dict[region]

    input_data = np.array([[age, sex, bmi, children, smoker, region]])

    prediction = model.predict(input_data)

    st.success(f"The predicted insurance cost is: ${prediction[0]:.2f}")