import joblib
from src.data import clean, split_xy
from src.synthetic import make_synthetic

def test_clean_handles_blank_total_charges():
    df = clean(make_synthetic(500))
    assert df["TotalCharges"].dtype.kind == "f"
    assert set(df["Churn"].unique()) <= {0, 1}
    assert "customerID" not in df.columns

def test_model_predicts_probability():
    model = joblib.load("models/churn_model.joblib")
    X, _ = split_xy(clean(make_synthetic(50, seed=1)))
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
