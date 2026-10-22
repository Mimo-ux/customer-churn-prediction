import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# 2. LOAD THE TRAINED MODEL
# -------------------------------

MODEL_PATH = Path(__file__).parent / "src" / "random_forest_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# 3. APP TITLE
# -------------------------------

st.title("📊 Customer Churn Prediction")
st.subheader("Predict whether a telecom customer is likely to leave.")

st.write(
    "Enter the customer's details below. "
    "The machine learning model will estimate their likelihood of churning."
)

st.divider()

# -------------------------------
# 4. CUSTOMER INPUT FORM
# -------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen?",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Has a Partner?",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.divider()

# -------------------------------
# 5. CONTRACT AND BILLING DETAILS
# -------------------------------

st.header("💳 Contract and Billing")

col4, col5, col6 = st.columns(3)

with col4:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col5:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col6:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=2000.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=100000.0,
        value=float(70 * tenure),
        step=10.0
    )

st.divider()

# -------------------------------
# 6. PREDICTION
# -------------------------------

if st.button("🔍 Predict Customer Churn", use_container_width=True):

    # Convert SeniorCitizen into the numeric format used by the dataset.
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0

    # Create a row containing the customer's original features.
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen_value,
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
        "TotalCharges": total_charges
    }])

    # Apply the same one-hot encoding approach used during training.
    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True,
        dtype=int
    )

    # Make the input columns match the training feature columns.
    expected_columns = model.feature_names_in_

    customer_encoded = customer_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Predict churn and its estimated probability.
    prediction = model.predict(customer_encoded)[0]
    churn_probability = model.predict_proba(customer_encoded)[0][1]

    st.divider()
    st.header("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Prediction: Customer is likely to churn")
        else:
            st.success("✅ Prediction: Customer is likely to stay")

    with result_col2:
        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    st.progress(float(churn_probability))

    st.caption(
        "This is a machine learning estimate, not a guarantee "
        "of the customer's future behaviour."
    )

