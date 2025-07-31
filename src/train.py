import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
joblib.dump(data, "data/data.pkl")
print("Data has been collected and stored as Data dump file")
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained. MSE: {mse:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model saved at model/model.pkl")
