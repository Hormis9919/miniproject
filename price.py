import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/Car details v3.csv")

# Handle missing values
df.dropna(inplace=True)

#outlier removal(IQR)
def remove_outliers_iqr(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0*IQR
    upper_bound = Q3 + 3.0*IQR
    return df[(df[column]>=lower_bound) & (df[column]<=upper_bound)]
    
#outlier capping
def cap_outliers(df,column, lower_percentile = 0.01, upper_percentile = 0.99):
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

#applying outlier handling methods to columns
df = remove_outliers_iqr(df,"km_driven")
df = cap_outliers(df,"year")
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
rf_cv = RandomizedSearchCV(RandomForestRegressor(),param_distributions=param_grid,n_iter= cv=5, n_jobs=-1, verbose=2)
rf_cv.fit(X_train, y_train)

print("Best Parameters for Random Forest:", rf_cv.best_params_)

#Optimize Extra Trees using` RandomizedSearchCV

#Train Random Forest Regressor
rf_model = RandomForestRegressor(**rf_cv.best_params_, random_state=42)
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

import joblib

# Save trained models
#joblib.dump(rf_model, "models/random_forest.pkl")
#joblib.dump(et_model, "models/extra_trees.pkl")

#print("Models saved successfully as 'random_forest.pkl' and 'extra_trees.pkl'!")



# Evaluate Random Forest
evaluate_model(y_test, rf_model.predict(X_test), "Random Forest")

# Evaluate Extra Trees
evaluate_model(y_test, et_model.predict(X_test), "Extra Trees")

# Average model predictions using weighted  soft voting
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

# Define loss function (Minimize RMSE)
def loss_function(weights):
    weighted_pred = (weights[0] * rf_model.predict(X_test)) + (weights[1] * et_model.predict(X_test))
    return np.sqrt(mean_squared_error(y_test, weighted_pred))  # RMSE is used as metric

# Constraints: Weights should sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1})
bounds = [(0, 1), (0, 1)]  # Weights should be between 0 and 1

# Optimize weights
initial_weights = [0.5, 0.5]  # Start with equal weights
optimized_weights = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)
rf_weight, et_weight = optimized_weights.x
weights = {"rf_weight": rf_weight, "et_weight": et_weight}
#joblib.dump(weights, "weights/model_weights.pkl")

print(f"\n🔹 Optimized Weights - Random Forest: {rf_weight:.2f}, Extra Trees: {et_weight:.2f}")

# Apply optimized weights to get final predictions
y_pred_weighted = (rf_weight * rf_model.predict(X_test)) + (et_weight * et_model.predict(X_test))
evaluate_model(y_test, y_pred_weighted, "Optimized Weighted Averaged Model")
