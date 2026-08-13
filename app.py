from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap

app = Flask(__name__)
CORS(app)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
...

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = shap.TreeExplainer(model)
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        row = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    X = pd.DataFrame([row], columns=FEATURES)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    risk_score = round(float(proba[1]) * 100, 1)

    shap_values = explainer.shap_values(X_scaled)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0][:, 1]

    explanation = sorted(
        [{"feature": f, "impact": round(float(v), 3)} for f, v in zip(FEATURES, sv)],
        key=lambda x: -abs(x["impact"])
    )

    return jsonify({
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "risk_score_percent": risk_score,
        "top_factors": explanation[:3],
        "all_factors": explanation,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
