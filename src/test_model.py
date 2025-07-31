import joblib
import os
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def test_model_exists():
    assert os.path.exists("model/model.pkl"), "Model file not found!"

def test_model_performance():
    model = joblib.load("model/model.pkl")
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 1.0, f"MSE too high: {mse}"
