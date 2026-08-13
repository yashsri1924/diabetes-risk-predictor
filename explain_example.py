import pandas as pd
import numpy as np
import joblib
import shap

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("diabetes.csv", names=columns)
X = df.drop("Outcome", axis=1)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_scaled)

print("Prediction:", model.predict(sample_scaled)[0])
print("Prediction probability:", model.predict_proba(sample_scaled)[0])
print("\nSHAP values shape:", np.array(shap_values).shape)

# Show which features pushed the prediction up or down
sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0][:, 1]
for name, val in sorted(zip(X.columns, sv), key=lambda x: -abs(x[1])):
    direction = "increases" if val > 0 else "decreases"
    print(f"  {name:28s} {direction} risk (SHAP={val:.3f})")
