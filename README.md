# Diabetes Risk Prediction with Explainable AI

A working starter kit: trained model + SHAP explainability + Flask API + simple frontend.
Tested end-to-end — accuracy ~73%, F1 ~0.58 on the Pima Indians Diabetes dataset.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model

```bash
python train.py
```

This trains a Random Forest on `diabetes.csv`, prints accuracy/precision/recall/F1,
and saves `diabetes_model.pkl` + `scaler.pkl`.

## 2. See explainability in action (optional, standalone demo)

```bash
python explain_example.py
```

Shows how SHAP explains a single prediction — which features pushed the risk up or down.

## 3. Run the API

```bash
python app.py
```

Runs on `http://127.0.0.1:5000`. Test it:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

## 4. Open the demo frontend

Just open `index.html` in your browser (Flask must be running).

## Files

- `diabetes.csv` — dataset (768 patients, 8 features, 1 target)
- `train.py` — preprocessing + training + evaluation
- `app.py` — Flask API that predicts + explains
- `explain_example.py` — standalone SHAP demo (no Flask needed)
- `index.html` — simple form to test predictions in a browser
- `diabetes_model.pkl`, `scaler.pkl` — the trained model (already generated, ready to use)

## Next steps to extend this project

- Swap in a second disease dataset (e.g., UCI Heart Disease) and support multiple prediction types
- Try XGBoost instead of Random Forest and compare metrics
- Add a confusion matrix / ROC curve visualization to the frontend
- Deploy on Render/Railway (free tier) so it's a live link, not just localhost
