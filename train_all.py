"""
Trains and compares 3 models (Logistic Regression, Random Forest, XGBoost)
for each disease using 5-fold cross-validation, picks the best by F1 score,
and saves the final model + scaler + a SHAP background sample.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from config import DISEASES

RANDOM_STATE = 42


def load_dataset(cfg):
    if cfg["has_header"]:
        df = pd.read_csv(cfg["csv"])
    else:
        df = pd.read_csv(cfg["csv"], names=cfg["columns"])

    for col in cfg["zero_as_missing"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]
    return X, y


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    }


def train_for_disease(name, cfg):
    print(f"\n{'=' * 60}\n{cfg['label']} ({name})\n{'=' * 60}")
    X, y = load_dataset(cfg)
    print(f"Dataset shape: {X.shape}, class balance: {dict(y.value_counts())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  eval_metric="logloss", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("\n--- 5-Fold Cross-Validation (F1 score) ---")
    for model_name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        cv_results[model_name] = scores.mean()
        print(f"  {model_name:22s} F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nBest model by CV: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X_train_scaled, y_train)

    metrics = evaluate(best_model, X_test_scaled, y_test)
    print(f"\n--- Test Set Performance ({best_name}) ---")
    for k, v in metrics.items():
        print(f"  {k:12s} {v}")

    joblib.dump(best_model, f"models/{name}_model.pkl")
    joblib.dump(scaler, f"models/{name}_scaler.pkl")
    joblib.dump(X_train_scaled[:100], f"models/{name}_background.pkl")

    result = {
        "disease": cfg["label"],
        "best_model": best_name,
        "cv_f1_scores": {k: round(v, 3) for k, v in cv_results.items()},
        "test_metrics": metrics,
        "features": list(X.columns),
    }
    return result


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    all_results = {}
    for disease_key, disease_cfg in DISEASES.items():
        all_results[disease_key] = train_for_disease(disease_key, disease_cfg)

    with open("models/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}\nAll models trained and saved to models/\n{'=' * 60}")
