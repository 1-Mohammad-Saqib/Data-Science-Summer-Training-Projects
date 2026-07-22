import streamlit as st
import pickle
import numpy as np

# ---------------- Page Configuration ---------------- #
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Load Model ---------------- #
model = pickle.load(open("house.pkl", "rb"))

# ---------------- Title ---------------- #
st.title("🏠 Smart House Price Prediction System")
st.markdown("### Predict the selling price of a house using Machine Learning")

st.markdown("---")

# ---------------- Inputs ---------------- #
col1, col2 = st.columns(2)

with col1:
    overall_quality = st.slider("Overall Quality", 1, 10, 5)

    ground_living_area = st.number_input(
        "Ground Living Area (sq ft)",
        min_value=200,
        max_value=6000,
        value=1500
    )

    garage_cars = st.slider("Garage Capacity", 0, 5, 2)

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=500
    )

    basement = st.number_input(
        "Total Basement Area",
        min_value=0,
        max_value=6000,
        value=900
    )

with col2:

    first_floor = st.number_input(
        "First Floor Area",
        min_value=200,
        max_value=5000,
        value=1200
    )

    full_bath = st.slider("Full Bathrooms", 0, 5, 2)

    year_built = st.number_input(
        "Year Built",
        min_value=1870,
        max_value=2025,
        value=2000
    )

    remodel = st.number_input(
        "Year Remodeled",
        min_value=1950,
        max_value=2025,
        value=2005
    )

    rooms = st.slider(
        "Total Rooms Above Ground",
        2,
        15,
        6
    )

st.markdown("---")

# ---------------- Prediction ---------------- #

if st.button("💰 Predict House Price"):

    features = np.array([[
        overall_quality,
        ground_living_area,
        garage_cars,
        garage_area,
        basement,
        first_floor,
        full_bath,
        year_built,
        remodel,
        rooms
    ]])

    prediction = model.predict(features)

    st.success(f"🏠 Estimated House Price : ${prediction[0]:,.2f}")

st.markdown("---")
st.caption("Developed by Mohammad Saqib | Random Forest Regressor | Streamlit")