import os
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

from config import DISEASES
from train_all import load_dataset, RANDOM_STATE

os.makedirs("plots", exist_ok=True)


def plot_for_disease(name, cfg):
    print(f"\nGenerating plots for {cfg['label']}...")
    X, y = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(f"models/{name}_model.pkl")
    scaler = joblib.load(f"models/{name}_scaler.pkl")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Confusion Matrix ----
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{cfg['label']} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_confusion_matrix.png", dpi=150)
    plt.close()

    # ---- ROC Curve ----
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
    plt.title(f"{cfg['label']} — ROC Curve")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_roc_curve.png", dpi=150)
    plt.close()

    # ---- SHAP Summary ----
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        sv = shap_values[1] if isinstance(shap_values, list) else (
            shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
        )
    else:
        explainer = shap.Explainer(model, X_train_scaled)
        sv_full = explainer(X_test_scaled)
        sv = sv_full.values

    plt.figure()
    shap.summary_plot(sv, X_test, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {name}_confusion_matrix.png, {name}_roc_curve.png, {name}_shap_summary.png")


if __name__ == "__main__":
    for disease_key, disease_cfg in DISEASES.items():
        plot_for_disease(disease_key, disease_cfg)
    print("\nAll plots generated in plots/")
