#LOAN PREDICTION DATASET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
loan_data = pd.read_csv("C:/Users/NISHITA/Desktop/COURSE/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")

# Check and handle missing values if necessary

# Perform one-hot encoding for categorical variables
loan_data_encoded = pd.get_dummies(loan_data, drop_first=True)

# Split data into features (X) and target variable (y)
X = loan_data_encoded.drop(columns=["Loan_Status_Y"])  # Features
y = loan_data_encoded["Loan_Status_Y"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
# Convert predictions to binary values using a threshold (e.g., 0.5)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = (y_pred_binary == y_test).mean()
print("Accuracy:", accuracy)