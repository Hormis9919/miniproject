import joblib
import numpy as np
import pandas as pd

# Load saved models from "models/" folder
rf_model = joblib.load("models/random_forest.pkl")
et_model = joblib.load("models/extra_trees.pkl")

# Load optimized weights from "weights/" folder
weights = joblib.load("weights/model_weights.pkl")
rf_weight = weights["rf_weight"]
et_weight = weights["et_weight"]

# Load test dataset (Only for column reference)
X_test = pd.read_csv("dataset/X_test.csv")

print(f"Models, weights, and test dataset loaded successfully! RF Weight: {rf_weight:.2f}, ET Weight: {et_weight:.2f}")

#Function to make predictions using soft voting
def predict_price(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches training data
    input_df = input_df.reindex(columns=X_test.columns, fill_value=0)

    # Model predictions
    rf_pred = rf_model.predict(input_df)[0]
    et_pred = et_model.predict(input_df)[0]
    avg_pred = (rf_weight * rf_pred) + (et_weight * et_pred)

    return {
        "random_forest_prediction": round(rf_pred, 2),
        "extra_trees_prediction": round(et_pred, 2),
        "final_prediction": round(avg_pred, 2)
    }
