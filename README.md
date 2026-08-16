# Disease Risk Predictor — Diabetes & Heart Disease (Explainable AI)

An AI/ML web app that predicts disease risk for **two conditions** — Diabetes and
Heart Disease — using models chosen through proper cross-validation comparison,
with SHAP explainability so every prediction shows *why*, not just *what*.

## 🔗 Live Demo
https://diabetes-risk-predictor-ci2z.onrender.com
*(Free tier — first load may take 20-30s if the server was asleep.)*

## What makes this a genuine ML project (not just an API wrapper)

- **Two independent models, chosen by evidence, not guesswork.** For each disease,
  three algorithms (Logistic Regression, Random Forest, XGBoost) are compared
  using 5-fold cross-validation. The best performer is kept — and it's a
  *different* algorithm for each disease (Logistic Regression won for Diabetes,
  Random Forest won for Heart Disease), which is itself a useful thing to be able
  to explain.
- **Explainability via SHAP**, not just an accuracy number. Every prediction
  returns the top features that pushed the risk up or down, and by how much.
- **Full evaluation, not just accuracy.** Precision, recall, F1, and ROC-AUC are
  all tracked — see `models/metrics.json` after training.

## Results

| Disease | Best Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Diabetes | Logistic Regression | 0.708 | 0.60 | 0.50 | 0.545 | 0.813 |
| Heart Disease | Random Forest | 0.836 | 0.78 | 0.97 | 0.865 | 0.916 |

(Pima Indians Diabetes dataset, 768 patients · UCI Heart Disease dataset, 303 patients)

## Project Structure

```
config.py            # Central config: dataset paths, features, form field definitions
train_all.py          # Trains & compares 3 models per disease, saves the best
visualize_all.py       # Generates confusion matrix, ROC curve, SHAP summary plots
app.py                # Flask API + serves the frontend
templates/index.html  # Frontend — disease tabs, dynamic form, risk bar
models/                # Saved models, scalers, SHAP backgrounds, metrics.json
plots/                  # Generated evaluation plots (per disease)
diabetes.csv, heart.csv # Datasets
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train both models

```bash
python train_all.py
```

Trains and cross-validates 3 models per disease, saves the best one, and writes
`models/metrics.json` with full comparison results.

## 2. Generate evaluation plots

```bash
python visualize_all.py
```

Produces `plots/<disease>_confusion_matrix.png`, `plots/<disease>_roc_curve.png`,
and `plots/<disease>_shap_summary.png` for both diseases.

## 3. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000` — switch between Diabetes and Heart Disease tabs,
fill the form, and get a risk score with an explanation.

## API

- `GET /api/diseases` — list available diseases and their form fields
- `POST /api/predict/<disease>` — `disease` is `diabetes` or `heart`; body is a
  JSON object of feature values; returns prediction, risk %, and SHAP factors
- `GET /api/metrics` — full cross-validation + test metrics for both models
- `GET /api/health` — health check

## Deployment (Render)

This repo includes a `Procfile` (`web: gunicorn app:app`) and `runtime.txt`
(pins Python 3.11.9) for one-click deployment on Render's free tier. Build
command: `pip install -r requirements.txt`.

## Next steps to extend this project further

- Add a third disease (e.g., liver disease) — just add one entry to `config.py`
- Hyperparameter tuning with `GridSearchCV` on the winning model per disease
- SHAP waterfall/force plot for a single prediction (not just the summary plot)
- Swap Render's free tier for a paid one to avoid cold-start delay
