import pandas as pd

# Load the dataset
df = pd.read_csv("Car details v3.csv")

# Handle missing values
df.dropna(inplace=True)


# Convert categorical variables to numerical using One-Hot Encoding
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']  # Adjust if needed
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now, we can define X and y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Convert 'mileage' column to numeric (remove 'kmpl')
# Convert 'mileage' column to numeric, handling both 'kmpl' and 'km/kg'
df['mileage'] = df['mileage'].str.replace(' kmpl', '', regex=True)
df['mileage'] = df['mileage'].str.replace(' km/kg', '', regex=True)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')  # Convert to float


# Convert 'engine' column to numeric (remove 'CC')
df['engine'] = df['engine'].str.replace(' CC', '', regex=True).astype(float)

# Convert 'max_power' column to numeric (remove 'bhp')
df['max_power'] = df['max_power'].str.replace(' bhp', '', regex=True).astype(float)

# Extract the first numeric value from 'torque' column (ignoring '@' values)
df['torque'] = df['torque'].str.split(' ').str[0]  # Keep only the first number
df['torque'] = pd.to_numeric(df['torque'], errors='coerce')  # Convert to float


# Define features (X) and target variable (y)
X = df.drop(columns=['selling_price', 'name'])  # Drop 'selling_price' (target) and 'name' (not useful)
y = df['selling_price']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Hyper parameter training
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Optimize Random Forest using RandomizedSearchCV
rf_cv = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1, verbose=2)
rf_cv.fit(X_train, y_train)

print("Best Parameters for Random Forest:", rf_cv.best_params_)

# Train Random Forest with the best parameters
rf_model = RandomForestRegressor(**rf_cv.best_params_, random_state=42)
rf_model.fit(X_train, y_train)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Extra Trees Regressor
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_et = et_model.predict(X_test)
from sklearn.metrics import r2_score

# Function to evaluate models with additional metrics
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)  # Accuracy metric
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Percentage error
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Accuracy: {r2 * 100:.2f}%, MAPE: {mape:.2f}%\n")

# Evaluate Random Forest
evaluate_model(y_test, rf_model.predict(X_test), "Random Forest")

# Evaluate Extra Trees
evaluate_model(y_test, et_model.predict(X_test), "Extra Trees")

# Average model predictions
y_pred_avg = (rf_model.predict(X_test) + et_model.predict(X_test)) / 2
evaluate_model(y_test, y_pred_avg, "Averaged Model (RF + Extra Trees)")

# ----------------- User Input for Prediction ----------------- #
def predict_price():
    print("\n🔹 Enter car details to predict its selling price 🔹\n")

    # Get user input
    year = int(input("Enter car year (e.g., 2015): "))
    km_driven = int(input("Enter kilometers driven: "))
    mileage = float(input("Enter mileage (kmpl): "))
    engine = float(input("Enter engine capacity (CC): "))
    max_power = float(input("Enter max power (bhp): "))
    torque = float(input("Enter torque (Nm): "))
    seats = int(input("Enter number of seats: "))

    # One-hot encoded categorical features
    fuel_type = input("Enter fuel type (Diesel/Petrol/LPG): ").strip().lower()
    seller_type = input("Enter seller type (Dealer/Individual/Trustmark Dealer): ").strip().lower()
    transmission = input("Enter transmission type (Manual/Automatic): ").strip().lower()
    owner_type = input("Enter owner type (First/Second/Third/Fourth & Above): ").strip().lower()

    # Create a dictionary for input features
    input_data = {
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'torque': torque,
        'seats': seats,
        'fuel_Diesel': 1 if fuel_type == 'diesel' else 0,
        'fuel_LPG': 1 if fuel_type == 'lpg' else 0,
        'fuel_Petrol': 1 if fuel_type == 'petrol' else 0,
        'seller_type_Individual': 1 if seller_type == 'individual' else 0,
        'seller_type_Trustmark Dealer': 1 if seller_type == 'trustmark dealer' else 0,
        'transmission_Manual': 1 if transmission == 'manual' else 0,
        'owner_Fourth & Above Owner': 1 if owner_type == 'fourth & above' else 0,
        'owner_Second Owner': 1 if owner_type == 'second' else 0,
        'owner_Test Drive Car': 0,  # This may not be needed
        'owner_Third Owner': 1 if owner_type == 'third' else 0
    }

    # Convert to DataFrame and align with training data columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Fill missing columns with 0

    # Make predictions using the best model
    rf_pred = rf_model.predict(input_df)[0]
    et_pred = et_model.predict(input_df)[0]
    avg_pred = (rf_pred + et_pred) / 2  # Averaging both models

    print("\n🔹 Predicted Selling Prices 🔹")
    print(f"-> Random Forest Prediction: ₹{rf_pred:,.2f}")
    print(f"-> Extra Trees Prediction: ₹{et_pred:,.2f}")
    print(f"-> Averaged Model Prediction: ₹{avg_pred:,.2f} (Recommended)")

# Call the function after training
predict_price()
