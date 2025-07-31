import joblib
import numpy as np

print("Loading quantized model...")
model = joblib.load("model/model_quantized.pkl")

# Sample inference
sample_input = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]])
prediction = model.predict(sample_input)

print(f"Predicted house price: {prediction[0]:.2f}")
