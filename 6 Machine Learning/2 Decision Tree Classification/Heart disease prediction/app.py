import streamlit as st
import pickle 
import numpy as np

model = pickle.load(open("Heart_disease_model.pkl", "rb"))
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️Heart Disease Prediction Web App")

st.write("Enter patient details below:")

age = st.number_input("Age", min_value=1, max_value=120,value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3, 4])
bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=250,value=120)
chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600,value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
ekg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220,value=150)
angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
num_vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.number_input("Thallium Stress Test Result", min_value=0, max_value=10,value=3)

if sex == "Male":
    sex = 1
else:  
    sex = 0

if fbs == "Yes":
    fbs = 1
else:
    fbs = 0

if angina == "Yes":
    angina = 1
else:
    angina = 0



if st.button("Predict"):
    input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg, max_hr, angina, depression, slope, num_vessels, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")