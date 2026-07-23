"""Streamlit dashboard for predicting customer churn.

Run with: streamlit run app.py
"""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """Load the trained model and its expected one-hot encoded columns."""
    with open(BASE_DIR / "customer_churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open(BASE_DIR / "feature_names.pkl", "rb") as file:
        feature_names = list(pickle.load(file))
    with open(BASE_DIR / "label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)
    return model, feature_names, encoder


def create_model_input(values: dict, feature_names: list[str]) -> pd.DataFrame:
    """Turn form values into the exact dummy-variable layout used in training."""
    # get_dummies on a one-row frame drops every category, so construct the
    # saved one-hot layout directly from the training column names instead.
    encoded = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)
    for name, value in values.items():
        if name in feature_names:  # Numeric columns.
            encoded.loc[0, name] = value
        else:  # Categorical dummy columns, e.g. Contract_Two year.
            dummy_name = f"{name}_{value}"
            if dummy_name in feature_names:
                encoded.loc[0, dummy_name] = 1
    return encoded


def risk_drivers(values: dict) -> list[str]:
    """Provide readable, directional guidance alongside the model score."""
    drivers = []
    if values["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contracts are commonly associated with higher churn.")
    if values["InternetService"] == "Fiber optic":
        drivers.append("Fiber-optic customers merit a service-experience check-in.")
    if values["PaymentMethod"] == "Electronic check":
        drivers.append("Electronic-check billing can indicate a less sticky payment relationship.")
    if values["tenure"] < 12:
        drivers.append("Newer customers benefit most from early onboarding and retention outreach.")
    if values["TechSupport"] == "No" and values["InternetService"] != "No":
        drivers.append("No technical support is a useful opportunity for a proactive support offer.")
    return drivers or ["This profile has no major rule-of-thumb risk signals from the inputs selected."]


st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; color: #17233b; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #101b36 0%, #192c58 100%); }
      [data-testid="stSidebar"] * { color: #eef4ff !important; }
      .hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
              background: linear-gradient(115deg, #172b58 0%, #315fba 58%, #42a8c5 100%);
              box-shadow: 0 12px 30px rgba(31, 63, 123, .20); margin-bottom: 1.35rem; }
      .hero h1 { margin: 0; font-size: 2.15rem; letter-spacing: -.04em; }
      .hero p { margin: .4rem 0 0; opacity: .87; font-size: 1rem; }
      .eyebrow { font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .78; }
      .risk-card { background: white; border-radius: 16px; padding: 1.5rem; border: 1px solid #e6ebf3;
                   box-shadow: 0 4px 16px rgba(21, 42, 81, .07); text-align: center; }
      .risk-number { font-size: 3.1rem; font-weight: 800; line-height: 1; color: #203c78; }
      .risk-label { margin-top: .45rem; color: #61708a; font-size: .95rem; }
      .section-note { color: #68758d; margin-top: -.55rem; margin-bottom: .7rem; }
      /* Streamlit renders widget labels with nested elements; colour every
         level so the questions remain clearly readable in the main panel. */
      .main label, .main label *, .main [data-testid="stWidgetLabel"],
      .main [data-testid="stWidgetLabel"] *, .stMain label, .stMain label *,
      .stMain [data-testid="stWidgetLabel"], .stMain [data-testid="stWidgetLabel"] * {
          color: #172b4d !important; opacity: 1 !important; font-weight: 700 !important;
          font-size: .88rem !important; line-height: 1.35 !important;
      }
      div[data-testid="stForm"] { background: white; padding: 1.15rem 1.25rem .35rem; border-radius: 16px;
                                   border: 1px solid #e6ebf3; }
      .stButton > button, .stFormSubmitButton > button { background: #315fba !important; color: white !important;
          border: 0 !important; border-radius: 9px !important; font-weight: 700 !important; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model, feature_names, label_encoder = load_artifacts()
except Exception as error:
    st.error(f"Model files could not be loaded: {error}")
    st.stop()

st.sidebar.markdown("## ◈ Customer Churn Predictor")
st.sidebar.caption("Predict customer retention risk")
st.sidebar.divider()
st.sidebar.markdown("### How to use")
st.sidebar.markdown("1. Complete the customer details\n2. Click **Assess churn risk**\n3. Review the prediction and suggestions")
st.sidebar.divider()
st.sidebar.caption("Model: Random Forest classifier")

st.markdown(
    """<div class="hero"><div class="eyebrow">Machine learning prediction</div>
    <h1>Customer Churn Predictor</h1>
    <p>Enter a customer's account, service, and billing information to estimate their chance of leaving.</p></div>""",
    unsafe_allow_html=True,
)

form_col, result_col = st.columns([1.65, 1], gap="large")

with form_col:
    st.subheader("Customer profile")
    st.caption("Select the options that match the customer you want to assess.")
    with st.form("churn_form"):
        st.markdown("#### Account details")
        st.caption("Tell us about the customer's household and how long they have been with the company.")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender (Male / Female)", ["Female", "Male"])
        with c2:
            partner = st.selectbox("Partner? (Yes / No)", ["No", "Yes"])
        with c3:
            tenure = st.slider("Tenure (Months with us)", 0, 72, 12)
        c1, c2, c3 = st.columns(3)
        with c1:
            senior = st.selectbox("Senior citizen? (Yes / No)", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            dependents = st.selectbox("Dependents? (Yes / No)", ["No", "Yes"])
        with c3:
            paperless = st.selectbox("Paperless billing? (Yes / No)", ["No", "Yes"])

        st.markdown("#### Services")
        st.caption("Choose the communication, internet, support, and entertainment services currently used by the customer.")
        c1, c2, c3 = st.columns(3)
        with c1:
            phone = st.selectbox("Phone service? (Yes / No)", ["Yes", "No"])
        with c2:
            multiple = st.selectbox("Multiple phone lines?", ["No", "Yes", "No phone service"])
        with c3:
            internet = st.selectbox("Internet service type", ["DSL", "Fiber optic", "No"])
        c1, c2, c3 = st.columns(3)
        with c1:
            security = st.selectbox("Online security?", ["No", "Yes", "No internet service"])
        with c2:
            backup = st.selectbox("Online backup?", ["No", "Yes", "No internet service"])
        with c3:
            protection = st.selectbox("Device protection?", ["No", "Yes", "No internet service"])
        c1, c2, c3 = st.columns(3)
        with c1:
            support = st.selectbox("Tech support?", ["No", "Yes", "No internet service"])
        with c2:
            streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
        with c3:
            streaming_movies = st.selectbox("Streaming movies?", ["No", "Yes", "No internet service"])

        st.markdown("#### Plan & billing")
        st.caption("Enter the customer's agreement, payment method, and charges from their current account.")
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox("Contract term", ["Month-to-month", "One year", "Two year"])
        with c2:
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with c3:
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.05)
            total = st.number_input("Total charges ($)", 0.0, 10000.0, 840.0, 0.05)

        submitted = st.form_submit_button("Assess churn risk")

    values = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

with result_col:
    st.subheader("Assessment")
    if submitted:
        model_input = create_model_input(values, feature_names)
        probability = float(model.predict_proba(model_input)[0][1])
        predicted_class = int(model.predict(model_input)[0])
        prediction = label_encoder.inverse_transform([predicted_class])[0]
        risk_text = "High risk" if probability >= 0.60 else "Watchlist" if probability >= 0.35 else "Low risk"
        card_color = "#d64c5b" if probability >= 0.60 else "#d99026" if probability >= 0.35 else "#16856b"

        st.markdown(
            f'''<div class="risk-card"><div class="eyebrow" style="color:#61708a;opacity:1">{risk_text}</div>
            <div class="risk-number" style="color:{card_color}">{probability:.0%}</div>
            <div class="risk-label">Likelihood that this customer will churn</div></div>''',
            unsafe_allow_html=True,
        )
        st.progress(int(probability * 100))
        if prediction == "Yes":
            st.error("Model recommendation: prioritize retention outreach.")
        else:
            st.success("Model recommendation: customer is likely to remain.")

        st.markdown("#### Suggested focus areas")
        for driver in risk_drivers(values):
            st.markdown(f"- {driver}")
    else:
        st.markdown("""<div class="risk-card"><div class="risk-number">—</div>
        <div class="risk-label">Submit a profile to view the churn probability and suggested actions.</div></div>""", unsafe_allow_html=True)
        st.info("The score estimates churn likelihood; use it to support—not replace—retention decisions.")
