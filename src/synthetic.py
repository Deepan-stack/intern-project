"""Synthetic Telco-like churn data so the project runs before you download the real dataset."""
import numpy as np
import pandas as pd

def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    monthly = np.round(20 + 50 * (internet == "Fiber optic") + 25 * (internet == "DSL") + rng.normal(0, 5, n), 2)
    total = np.round(monthly * tenure, 2).astype(object)
    total[rng.random(n) < 0.01] = " "          # real dataset has blank TotalCharges
    logit = (-1.0 + 1.6 * (contract == "Month-to-month") - 0.04 * tenure
             + 0.8 * (internet == "Fiber optic") + 0.5 * (payment == "Electronic check")
             + 0.02 * (monthly - 60) + 0.3 * senior)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "SeniorCitizen": senior, "tenure": tenure, "Contract": contract,
        "InternetService": internet, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })
