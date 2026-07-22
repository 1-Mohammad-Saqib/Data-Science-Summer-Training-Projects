import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model('diabetes_ann_model.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title('🩺 Diabetes Prediction using ANN')

preg = st.number_input('Pregnancies', value=0, min_value=0)
glu = st.number_input('🍬Glucose', value=0, min_value=0)
bp = st.number_input('❤️Blood Pressure', value=0, min_value=0)
skin = st.number_input('🩹Skin Thickness', value=0, min_value=0)
insulin = st.number_input('💉Insulin', value=0, min_value=0)
bmi = st.number_input('BMI')
diabetes_pedigree = st.number_input('🧬Diabetes Pedigree Function')
age = st.number_input('🎂Age', value=0, min_value=0)

if st.button('Predict'):
    data = np.array([[preg, glu, bp, skin, insulin, bmi, diabetes_pedigree, age]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    if prediction[0][0] > 0.5:
        st.error('The person is likely to have diabetes.')
    else:
        st.success('The person is not likely to have diabetes.')