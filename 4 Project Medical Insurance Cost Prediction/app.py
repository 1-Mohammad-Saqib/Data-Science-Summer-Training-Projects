import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("insurence_model.pkl","rb"))

st.title("Medical Insurance Cost Prediction")
st.write("Enter the details below to predict insurance charges.")

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