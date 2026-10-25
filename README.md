# Customer Churn Prediction

Predict which telecom customers will leave, find the drivers, and serve predictions in a Streamlit dashboard.

## Run
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m src.train            # trains LR + RF, saves model, metrics, plots
    pytest                         # tests
    streamlit run app/streamlit_app.py

## Real data
Download "Telco Customer Churn" (IBM sample, Kaggle) and save it as `data/telco_churn.csv`.
Then add its extra columns (gender, Partner, OnlineSecurity, ...) automatically: the pipeline
detects numeric vs categorical columns itself, so no code change is needed except the dashboard form.
Without the file, the project falls back to synthetic data so everything still runs.
