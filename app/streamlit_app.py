import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")

model = joblib.load("models/churn_model.joblib")
metrics = json.load(open("reports/metrics.json"))

tab1, tab2 = st.tabs(["Predict", "Model performance"])

with tab1:
    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c2.number_input("Monthly charges", 10.0, 150.0, 70.0)
    senior = c3.selectbox("Senior citizen", [0, 1])
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = c2.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = c3.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    row = pd.DataFrame([{"SeniorCitizen": senior, "tenure": tenure, "Contract": contract,
                         "InternetService": internet, "PaymentMethod": payment,
                         "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}])
    p = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(float(p))
    st.write("Risk level:", "High" if p > 0.6 else "Medium" if p > 0.3 else "Low")

with tab2:
    st.write(f"Best model: **{metrics['best']}**")
    st.dataframe(pd.DataFrame(metrics["results"]).T)
    a, b, c = st.columns(3)
    a.image("reports/roc.png"); b.image("reports/confusion.png"); c.image("reports/drivers.png")
