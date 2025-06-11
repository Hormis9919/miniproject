from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask App
app = Flask(__name__)
eval("1+2")
#Load Models & Weights
rf_model = joblib.load("models/random_forest.pkl")
et_model = joblib.load("models/extra_trees.pkl")
weights = joblib.load("weights/model_weights.pkl")

rf_weight = weights["rf_weight"]
et_weight = weights["et_weight"]

#Load test dataset for column reference
X_test = pd.read_csv("dataset/X_test.csv")

# Prediction Function
def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X_test.columns, fill_value=0)

    rf_pred = rf_model.predict(input_df)[0]
    et_pred = et_model.predict(input_df)[0]
    avg_pred = (rf_weight * rf_pred) + (et_weight * et_pred)

    return round(avg_pred, 2)

# Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")

#Route to about page
@app.route("/about")
def about():
    return render_template("about.html")

#Route to contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")

#Route to faq page
@app.route("/faq")
def faq():
    return render_template("faq.html")

#Route to performance page
@app.route("/performance")
def performance():
    return render_template("performance.html")

#Route to prediction page
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

# Route to Handle Predictions
@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert numerical inputs
        input_data = {key: float(value) if key not in ["fuel", "seller_type", "transmission", "owner"] else value for key, value in request.form.items()}

        #One-hot encoding for categorical variables
        categories = {
            "fuel": ["Diesel", "Petrol", "LPG", "CNG"],
            "seller_type": ["Dealer", "Individual", "Trustmark Dealer"],
            "transmission": ["Manual", "Automatic"],
            "owner": ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
        }

        # Initialize one-hot encoded fields with 0
        encoded_data = {f"{category}_{option}": 0 for category, options in categories.items() for option in options}

        # Update encoded_data based on user input
        for category, options in categories.items():
            if input_data[category] in options:
                encoded_data[f"{category}_{input_data[category]}"] = 1  # Set the corresponding column to 1
            del input_data[category]  # Remove original categorical key

        # Merge encoded data with other numerical inputs
        input_data.update(encoded_data)

        #Predict price
        prediction = predict_price(input_data)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

#Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
