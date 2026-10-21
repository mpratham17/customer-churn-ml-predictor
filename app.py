
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load saved model artifacts
# -----------------------------
MODEL_PATH = Path(__file__).parent / "models" / "churn_model.pkl"


@st.cache_resource
def load_artifacts():
    # Only load pickle files that you created or otherwise trust.
    with open(MODEL_PATH, "rb") as file:
        artifacts = pickle.load(file)

    return (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["threshold"]
    )


st.title("📊 Customer Churn Prediction")
st.write(
    "Estimate customer churn risk using the trained "
    "Logistic Regression model."
)

try:
    model, preprocessor, threshold = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "'models/churn_model.pkl' exists."
    )
    st.stop()
except (KeyError, pickle.UnpicklingError, EOFError) as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()


st.info(
    f"Classification threshold: {threshold:.2f} "
    "| Customers with predicted churn probability at or above "
    f"{threshold:.0%} are flagged as potential churn risks."
)

# -----------------------------
# Customer input form
# -----------------------------
with st.form("customer_form"):

    st.subheader("1. Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        tenure_months = st.number_input(
            "Tenure (months)",
            min_value=0,
            value=12,
            step=1
        )

        monthly_fee = st.number_input(
            "Monthly Fee",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        region = st.selectbox(
            "Region",
            [
                "Asia-Pacific",
                "Europe",
                "Latin America",
                "Middle East & Africa",
                "North America"
            ]
        )

        device_type = st.selectbox(
            "Device Type",
            ["Desktop", "Mobile", "Smart TV", "Tablet"]
        )

    with col2:
        customer_segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Enterprise", "SMB"]
        )

        preferred_contact = st.selectbox(
            "Preferred Contact",
            ["Email", "In-App", "Phone", "SMS"]
        )

        is_premium_plan = st.checkbox("Premium Plan")

        discount_received_last_6m = st.checkbox(
            "Received Discount in Last 6 Months"
        )

    st.subheader("2. Usage and Engagement")

    col3, col4 = st.columns(2)

    with col3:
        total_usage_last_30d = st.number_input(
            "Total Usage in Last 30 Days",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        support_tickets_last_90d = st.number_input(
            "Support Tickets in Last 90 Days",
            min_value=0,
            value=0,
            step=1
        )

        num_competing_offers_in_region = st.number_input(
            "Competing Offers in Region",
            min_value=0,
            value=1,
            step=1
        )

        has_outstanding_invoice = st.checkbox(
            "Has Outstanding Invoice"
        )

    with col4:
        engagement_score = st.number_input(
            "Engagement Score",
            value=50.0,
            step=1.0
        )

        csat_score = st.number_input(
            "Customer Satisfaction Score",
            value=3.0,
            step=0.5
        )

        cltv_estimated = st.number_input(
            "Estimated Customer Lifetime Value",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------
if submitted:

    customer_data = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_fee": monthly_fee,
        "total_usage_last_30d": total_usage_last_30d,
        "support_tickets_last_90d": support_tickets_last_90d,
        "num_competing_offers_in_region":
            num_competing_offers_in_region,
        "engagement_score": engagement_score,
        "csat_score": csat_score,
        "cltv_estimated": cltv_estimated,
        "has_outstanding_invoice":
            int(has_outstanding_invoice),
        "discount_received_last_6m":
            int(discount_received_last_6m),
        "is_premium_plan": int(is_premium_plan),
        "region": region,
        "device_type": device_type,
        "customer_segment": customer_segment,
        "preferred_contact": preferred_contact
    }])

    try:
        # Use the original fitted preprocessing pipeline.
        transformed_data = preprocessor.transform(customer_data)

        # Probability of class 1 = churn.
        churn_probability = model.predict_proba(
            transformed_data
        )[0, 1]

        # Apply the selected threshold.
        prediction = int(churn_probability >= threshold)

        st.divider()
        st.subheader("3. Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.1%}"
            )

            st.progress(float(churn_probability))

        with result_col2:
            if prediction == 1:
                st.error("⚠️ Potential Churn Risk")
                st.write(
                    "The estimated probability meets or exceeds "
                    "the selected classification threshold."
                )
            else:
                st.success("✅ Lower Predicted Churn Risk")
                st.write(
                    "The estimated probability is below the "
                    "selected classification threshold."
                )

        st.caption(
            "This is a model prediction, not a guarantee of "
            "future customer behavior. Results depend on input "
            "quality and model performance."
        )

    except (ValueError, TypeError) as error:
        st.error(f"Could not generate prediction: {error}")
