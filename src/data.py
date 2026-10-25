from pathlib import Path
import pandas as pd
from src.synthetic import make_synthetic

DATA_PATH = Path("data/telco_churn.csv")   # put the Kaggle "WA_Fn-UseC_-Telco-Customer-Churn.csv" here
TARGET = "Churn"

def load_raw() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    print("[info] real dataset not found, using synthetic data")
    return make_synthetic()

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    df = df.drop(columns=["customerID"], errors="ignore").drop_duplicates()
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df

def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]
