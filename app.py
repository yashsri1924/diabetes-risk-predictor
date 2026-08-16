from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import joblib
import shap

from config import DISEASES

app = Flask(__name__)
app.json.sort_keys = False

# Load models, scalers, and build SHAP explainers ONCE at startup (not per-request)
MODELS = {}
for key, cfg in DISEASES.items():
    model = joblib.load(f"models/{key}_model.pkl")
    scaler = joblib.load(f"models/{key}_scaler.pkl")
    model_type = type(model).__name__
    features = list(cfg["fields"].keys())

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        background = joblib.load(f"models/{key}_background.pkl")
        explainer = shap.Explainer(model, background)

    MODELS[key] = {
        "model": model,
        "scaler": scaler,
        "model_type": model_type,
        "features": features,
        "explainer": explainer,
        "config": cfg,
    }

with open("models/metrics.json") as f:
    METRICS = json.load(f)


def get_shap_values(disease_key, X_scaled):
    """Returns per-feature SHAP contributions for a single scaled sample."""
    entry = MODELS[disease_key]
    model_type = entry["model_type"]
    explainer = entry["explainer"]

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        shap_values = explainer.shap_values(X_scaled)
        sv = shap_values[1][0] if isinstance(shap_values, list) else (
            shap_values[0][:, 1] if shap_values.ndim == 3 else shap_values[0]
        )
    else:
        sv = explainer(X_scaled).values[0]

    return sv


@app.route("/")
def home():
    diseases_info = {
        key: {"label": cfg["label"], "fields": cfg["fields"], "field_order": list(cfg["fields"].keys())}
        for key, cfg in DISEASES.items()
    }
    return render_template("index.html", diseases=diseases_info)


@app.route("/api/diseases", methods=["GET"])
def list_diseases():
    return jsonify({
        key: {"label": cfg["label"], "fields": cfg["fields"], "field_order": list(cfg["fields"].keys())}
        for key, cfg in DISEASES.items()
    })


@app.route("/api/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in MODELS:
        return jsonify({"error": f"Unknown disease type '{disease}'"}), 404

    entry = MODELS[disease]
    features = entry["features"]
    data = request.get_json()

    try:
        row = [float(data[f]) for f in features]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    X = pd.DataFrame([row], columns=features)
    X_scaled = entry["scaler"].transform(X)

    model = entry["model"]
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    risk_score = round(float(proba[1]) * 100, 1)

    sv = get_shap_values(disease, X_scaled)
    explanation = sorted(
        [{"feature": f, "impact": round(float(v), 3)} for f, v in zip(features, sv)],
        key=lambda x: -abs(x["impact"])
    )

    cfg = entry["config"]
    return jsonify({
        "disease": cfg["label"],
        "prediction": cfg["positive_label"] if pred == 1 else cfg["negative_label"],
        "risk_score_percent": risk_score,
        "top_factors": explanation[:3],
        "all_factors": explanation,
        "model_used": entry["model_type"],
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(METRICS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "diseases_loaded": list(MODELS.keys())})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
