
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Retention Intelligence Platform",
    page_icon="📉",
    layout="wide",
)

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

@st.cache_resource
def load_pipeline():
    with open("gradientboosting.pkl", "rb") as f:
        pipe = pickle.load(f)
    return pipe


def predict(pipe, df: pd.DataFrame) -> pd.DataFrame:
    preds = pipe.predict(df)
    probs = pipe.predict_proba(df)[:, 1]
    out = df.copy()
    out["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
    out["Churn_Probability"] = (probs * 100).round(2)
    return out


def risk_bucket(prob: float) -> str:
    if prob >= 70:
        return "🔴 High Risk"
    elif prob >= 40:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def gauge_chart(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 40], "color": "#bbf7d0"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": prob,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig

try:
    pipeline = load_pipeline()
    model_load_error = None
except Exception as e:  
    pipeline = None
    model_load_error = str(e)

st.title("Customer Retention Intelligence Platform")
st.subheader("Enter customer details")

with st.form("single_customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        st.markdown("**Phone & Internet**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"],
        )
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        internet_opts = ["No internet service", "No", "Yes"] if internet_service == "No" else ["No", "Yes"]
        online_security = st.selectbox("Online Security", internet_opts)
        online_backup = st.selectbox("Online Backup", internet_opts)

    with col3:
        st.markdown("**Add-ons**")
        device_protection = st.selectbox("Device Protection", internet_opts)
        tech_support = st.selectbox("Tech Support", internet_opts)
        streaming_tv = st.selectbox("Streaming TV", internet_opts)
        streaming_movies = st.selectbox("Streaming Movies", internet_opts)

        st.markdown("**Charges**")
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0,
            value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0,
        )

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

if submitted:
    if internet_service == "No":
        online_security = online_backup = device_protection = tech_support = "No internet service"
        streaming_tv = streaming_movies = "No internet service"
    if phone_service == "No":
        multiple_lines = "No phone service"

    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])[FEATURE_COLUMNS]

    try:
        result = predict(pipeline, row)
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.stop()

    prob = result["Churn_Probability"].iloc[0]
    pred_label = result["Churn_Prediction"].iloc[0]

    st.divider()
    rcol1, rcol2 = st.columns([1, 1])

    with rcol1:
        st.plotly_chart(gauge_chart(prob), use_container_width=True)

    with rcol2:
        st.markdown("### Result")
        if pred_label == "Yes":
            st.error(f"**Prediction: Likely to churn**  ({prob:.1f}% probability)")
        else:
            st.success(f"**Prediction: Likely to stay**  ({prob:.1f}% probability of churn)")
        st.metric("Risk Level", risk_bucket(prob))

        st.markdown("#### Key factors typically driving churn")
        notes = []
        if contract == "Month-to-month":
            notes.append("Month-to-month contracts churn far more than 1–2 year contracts.")
        if tenure <= 6:
            notes.append("Low tenure customers (new signups) are higher risk.")
        if internet_service == "Fiber optic":
            notes.append("Fiber optic customers show higher churn in this dataset.")
        if payment_method == "Electronic check":
            notes.append("Electronic check payers churn more than automatic payment methods.")
        if online_security == "No" and internet_service != "No":
            notes.append("No online security add-on correlates with higher churn.")
        if not notes:
            notes.append("No strong risk flags — profile looks similar to retained customers.")
        for n in notes:
            st.markdown(f"- {n}")

    with st.expander("Show input row sent to the model"):
        st.dataframe(row, use_container_width=True)
