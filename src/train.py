import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import load_raw, clean, split_xy

def build_preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

def main():
    df = clean(load_raw())
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "cv_auc": round(cv_auc, 4),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
        }
        fitted[name] = pipe
        print(name, results[name])

    best = max(results, key=lambda k: results[k]["roc_auc"])
    pipe = fitted[best]
    joblib.dump(pipe, "models/churn_model.joblib")
    json.dump({"best": best, "results": results}, open("reports/metrics.json", "w"), indent=2)

    RocCurveDisplay.from_estimator(pipe, X_te, y_te); plt.savefig("reports/roc.png", dpi=150); plt.close()
    ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te); plt.savefig("reports/confusion.png", dpi=150); plt.close()

    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    top = np.argsort(imp)[-10:]
    plt.figure(figsize=(7, 4)); plt.barh(names[top], imp[top]); plt.title(f"Top churn drivers ({best})")
    plt.tight_layout(); plt.savefig("reports/drivers.png", dpi=150); plt.close()
    print("Best model:", best)

if __name__ == "__main__":
    main()
