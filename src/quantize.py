import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("Loading trained model...")
model = joblib.load("model/model.pkl")

# Load dataset for evaluation
print("Loading dataset for R² evaluation...")
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Evaluate R² before quantization
y_pred_before = model.predict(X_test)
r2_before = r2_score(y_test, y_pred_before)
print(f"R² before quantization: {r2_before:.4f}")

# Quantize coefficients (reduce precision to float16)
print("Quantizing model coefficients...")
model.coef_ = model.coef_.astype(np.float16)
model.intercept_ = np.float16(model.intercept_)

# Evaluate R² after quantization
y_pred_after = model.predict(X_test)
r2_after = r2_score(y_test, y_pred_after)
print(f"R² after quantization: {r2_after:.4f}")

# Save quantized model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_quantized.pkl")
print("Quantized model saved at model/model_quantized.pkl")
