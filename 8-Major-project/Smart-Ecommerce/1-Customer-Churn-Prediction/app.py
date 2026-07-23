"""Streamlit interface for the e-commerce customer churn prediction model."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ecommerce_churn_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"


@st.cache_resource
def load_artifacts():
    """Load the trained classifier and its feature order once."""
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with FEATURES_PATH.open("rb") as features_file:
        features = list(pickle.load(features_file))
    return model, features


def build_feature_frame(values: dict, feature_names: list[str]) -> pd.DataFrame:
    """Convert form values into the exact one-hot encoded training format."""
    row = {feature: 0 for feature in feature_names}
    numeric_features = [
        "Age", "CityTier", "Tenure", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromLastYear", "CouponUsed", "OrderCount",
        "DaySinceLastOrder", "CashbackAmount", "AverageOrderValue", "ReturnRate",
        "DiscountUsage", "CustomerSupportCalls", "DeliveryDelay",
    ]
    for feature in numeric_features:
        row[feature] = values[feature]

    for prefix, selected_value in {
        "Gender": values["Gender"],
        "PreferredLoginDevice": values["PreferredLoginDevice"],
        "PreferredPaymentMode": values["PreferredPaymentMode"],
        "PreferredOrderCategory": values["PreferredOrderCategory"],
        "MaritalStatus": values["MaritalStatus"],
        "Membership": values["Membership"],
    }.items():
        encoded_feature = f"{prefix}_{selected_value}"
        if encoded_feature in row:
            row[encoded_feature] = 1
    return pd.DataFrame([row], columns=feature_names)


def input_heading(label: str) -> None:
    """Show a consistent, visible heading above every form control."""
    st.markdown(f'<p class="input-heading">{label}</p>', unsafe_allow_html=True)


def labeled_number(label: str, *args, **kwargs):
    input_heading(label)
    return st.number_input(label, *args, label_visibility="collapsed", **kwargs)


def labeled_select(label: str, options: list, **kwargs):
    input_heading(label)
    return st.selectbox(label, options, label_visibility="collapsed", **kwargs)


def labeled_slider(label: str, *args, **kwargs):
    input_heading(label)
    return st.slider(label, *args, label_visibility="collapsed", **kwargs)


def labeled_radio(label: str, options: list, **kwargs):
    input_heading(label)
    return st.radio(label, options, label_visibility="collapsed", **kwargs)


def app():
    st.set_page_config(page_title="ChurnGuard | Customer Churn Prediction", page_icon="🛒", layout="wide")
    st.markdown("""
        <style>
        .stApp { background: radial-gradient(circle at 15% 0%, #171a42 0, #090f20 34%, #060b18 80%); color: #e7ecfa; }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.75rem 2rem; border: 1px solid #263252; border-radius: 22px; color: #f4f6ff;
                background: linear-gradient(120deg, #111a35, #10172a 64%, #171535); margin-bottom: 1.5rem;
                box-shadow: 0 18px 42px rgba(1, 4, 16, .42); }
        .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -.03em; }
        .hero p { margin: .4rem 0 0; opacity: .88; font-size: 1.05rem; }
        .section-title { color: #cdd7f5; margin: .4rem 0 1rem; font-size: 1.05rem; font-weight: 750; }
        .input-heading { color: #aebbd7; font-size: .82rem; font-weight: 700; margin: .8rem 0 .28rem; }
        div[data-testid="stForm"] { background: rgba(15, 24, 45, .92); border: 1px solid #263252; border-radius: 20px;
                                    padding: 1.3rem 1.5rem; box-shadow: 0 12px 32px rgba(1, 4, 16, .28); }
        div[data-testid="stNumberInput"] > div > div, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #111b31 !important; border: 1px solid #344363 !important; border-radius: 10px !important; }
        div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: #e7ecfa !important; }
        div[data-testid="stNumberInputContainer"] > div[data-baseweb="input"],
        div[data-testid="stNumberInputContainer"] [data-baseweb="base-input"] {
            background: #111b31 !important; border-color: #344363 !important; }
        input[data-testid="stNumberInputField"] { color: #e7ecfa !important; }
        button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
            background: #1d2942 !important; color: #cdd7f5 !important; }
        div[data-testid="stRadio"] label, div[data-testid="stRadio"] label *,
        div[role="radiogroup"] label, div[role="radiogroup"] label * {
            color: #dbe3f6 !important; opacity: 1 !important; }
        button[data-baseweb="tab"] { font-size: .93rem; font-weight: 650; padding: .6rem 1rem; }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .6rem; border-bottom: 1px solid #263252; }
        div[data-testid="stTabs"] [aria-selected="true"] { color: #a995ff !important; border-bottom-color: #7c5cff !important; }
        div[data-testid="stTabs"] [aria-selected="false"] { color: #91a0bf !important; }
        div[data-testid="stFormSubmitButton"] button { background: linear-gradient(110deg, #7c5cff, #6042e8) !important; border-color: #8b70ff !important; border-radius: 10px; box-shadow: 0 8px 22px rgba(96, 66, 232, .3); }
        div[data-testid="stMetric"] { background: #101a2f; border: 1px solid #263252; border-radius: 16px; padding: 1rem; }
        .result-card { padding: 1.25rem 1.4rem; border-radius: 16px; margin-bottom: 1rem; }
        .risk-high { background: #301a33; border: 1px solid #91415c; color: #ffd7e2; }
        .risk-low { background: #102d30; border: 1px solid #27766d; color: #c5fff2; }
        .probability-card { background: linear-gradient(135deg, #382b82, #1c2551); border: 1px solid #5e51ae; border-radius: 16px; padding: 1rem 1.25rem;
                            color: #ffffff !important; min-height: 92px; }
        .probability-card span { display: block; color: #d9d6ff !important; font-size: .86rem;
                                 font-weight: 600; margin-bottom: .2rem; }
        .probability-card strong { display: block; color: #ffffff !important; font-size: 2.25rem;
                                   font-weight: 800; line-height: 1; letter-spacing: -.04em; }
        section[data-testid="stSidebar"] { background: #0d162a; border-right: 1px solid #263252; }
        section[data-testid="stSidebar"] * { color: #dbe3f6; }
        div[data-testid="stAlert"] { background: #16213b; color: #dbe3f6; border-color: #33466c; }
        .stMarkdown, .stText, p, label { color: #c2cce2; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="hero"><h1>🛒 ChurnGuard</h1>
        <p>✨ Estimate the likelihood that an e-commerce customer will churn.</p></div>
    """, unsafe_allow_html=True)

    try:
        model, feature_names = load_artifacts()
    except (FileNotFoundError, ModuleNotFoundError, pickle.UnpicklingError) as error:
        st.error(f"Could not load the prediction model: {error}")
        st.stop()

    with st.sidebar:
        st.header("💡 How to use")
        st.write("Complete the customer profile, then select **Predict churn risk**.")
        st.info("This estimate supports customer-retention decisions; it is not a final decision.")

    st.markdown('<p class="section-title">🧾 Customer profile</p>', unsafe_allow_html=True)
    with st.form("churn_form"):
        account_tab, shopping_tab, service_tab = st.tabs([
            "👤  Account", "🛍️  Shopping behaviour", "🤝  Experience & service"
        ])
        with account_tab:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                age = labeled_number("Age", 18, 100, 35)
                city_tier = labeled_select("City tier", [1, 2, 3], index=1)
                tenure = labeled_number("Tenure (months)", 0, 120, 12)
                gender = labeled_select("Gender", ["Female", "Male"])
            with col2:
                marital_status = labeled_select("Marital status", ["Divorced", "Married", "Single"])
                membership = labeled_select("Membership", ["Bronze", "Silver", "Gold", "Platinum"])
                device_count = labeled_number("Registered devices", 1, 20, 3)
                address_count = labeled_number("Saved addresses", 1, 20, 3)

        with shopping_tab:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                login_device = labeled_select("Preferred login device", ["Computer", "Mobile Phone", "Tablet"])
                payment_mode = labeled_select("Preferred payment mode", ["Cash on Delivery", "Credit Card", "Debit Card", "UPI", "Wallet"])
                order_category = labeled_select("Preferred order category", ["Electronics", "Fashion", "Grocery", "Home & Kitchen", "Laptop & Accessories", "Mobile"])
                hours_on_app = labeled_number("Hours spent on app", 0.0, 24.0, 2.5, 0.1)
            with col2:
                order_count = labeled_number("Order count", 0, 200, 25)
                coupon_used = labeled_number("Coupons used", 0, 100, 10)
                discount_usage = labeled_number("Discount usage (%)", 0.0, 100.0, 50.0, 1.0)
                average_order_value = labeled_number("Average order value", 0.0, 100000.0, 10000.0, 100.0)

        with service_tab:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                satisfaction = labeled_slider("Satisfaction score", 1, 5, 3)
                complained = labeled_radio("Has complained?", ["No", "Yes"], horizontal=True)
                warehouse_distance = labeled_number("Warehouse-to-home distance", 0.0, 200.0, 25.0, 1.0)
                last_order_days = labeled_number("Days since last order", 0, 365, 15)
                order_hike = labeled_number("Order amount hike from last year (%)", 0.0, 100.0, 15.0, 1.0)
            with col2:
                cashback = labeled_number("Cashback amount", 0.0, 10000.0, 250.0, 10.0)
                return_rate = labeled_number("Return rate (%)", 0.0, 100.0, 20.0, 1.0)
                support_calls = labeled_number("Customer support calls", 0, 100, 3)
                delivery_delay = labeled_number("Delivery delay (days)", 0, 100, 3)

        submitted = st.form_submit_button("✨ Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        values = {
            "Age": age, "CityTier": city_tier, "Tenure": tenure,
            "WarehouseToHome": warehouse_distance, "HourSpendOnApp": hours_on_app,
            "NumberOfDeviceRegistered": device_count, "SatisfactionScore": satisfaction,
            "NumberOfAddress": address_count, "Complain": int(complained == "Yes"),
            "OrderAmountHikeFromLastYear": order_hike, "CouponUsed": coupon_used,
            "OrderCount": order_count, "DaySinceLastOrder": last_order_days,
            "CashbackAmount": cashback, "AverageOrderValue": average_order_value,
            "ReturnRate": return_rate, "DiscountUsage": discount_usage,
            "CustomerSupportCalls": support_calls, "DeliveryDelay": delivery_delay,
            "Gender": gender, "PreferredLoginDevice": login_device,
            "PreferredPaymentMode": payment_mode, "PreferredOrderCategory": order_category,
            "MaritalStatus": marital_status, "Membership": membership,
        }
        input_frame = build_feature_frame(values, feature_names)
        prediction = int(model.predict(input_frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            churn_index = list(model.classes_).index(1)
            probability = float(model.predict_proba(input_frame)[0][churn_index])

        st.divider()
        if probability is None:
            st.warning("The model returned a class prediction but does not provide a churn percentage.")
        else:
            risk_class = "risk-high" if prediction == 1 else "risk-low"
            risk_label = "High churn risk" if prediction == 1 else "Low churn risk"
            risk_message = "This customer is predicted to be likely to churn." if prediction == 1 else "This customer is predicted to remain active."
            st.markdown(f'<div class="result-card {risk_class}"><strong>{risk_label}</strong><br>{risk_message}</div>', unsafe_allow_html=True)
            metric_col, detail_col = st.columns([1, 2])
            metric_col.markdown(
                f'<div class="probability-card"><span>Churn probability</span>'
                f'<strong>{probability * 100:.1f}%</strong></div>',
                unsafe_allow_html=True,
            )
            detail_col.info("The percentage is the model's estimated likelihood that this customer will churn.")
            st.progress(int(probability * 100))
        with st.expander("View model input"):
            st.dataframe(input_frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    app()
