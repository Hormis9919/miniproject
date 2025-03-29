import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load saved models from "models/" folder
rf_model = joblib.load("models/random_forest.pkl")
et_model = joblib.load("models/extra_trees.pkl")

# ✅ Load optimized weights from "weights/" folder
weights = joblib.load("weights/model_weights.pkl")
rf_weight = weights["rf_weight"]
et_weight = weights["et_weight"]

# ✅ Load test dataset
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv").values.ravel()  # Convert y_test to 1D array

print(f"✅ Models, weights, and test dataset loaded successfully!")
print(f"📌 RF Weight: {rf_weight:.2f}, ET Weight: {et_weight:.2f}")

# ✅ Define evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)  # Accuracy metric
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Percentage error

    print(f"\n🔹 {model_name} Evaluation 🔹")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score (Accuracy): {r2:.4%}")
    print(f"MAPE: {mape:.2f}%")

# ✅ Generate predictions for the test data
rf_pred = rf_model.predict(X_test)
et_pred = et_model.predict(X_test)

# ✅ Apply soft voting (weighted averaging)
y_pred_weighted = (rf_weight * rf_pred) + (et_weight * et_pred)

# ✅ Evaluate the final weighted model
evaluate_model(y_test, y_pred_weighted, "Optimized Weighted Averaged Model")
