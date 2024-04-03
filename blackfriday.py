import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Read the dataset
df = pd.read_csv("C:/Users/NISHITA/Desktop/COURSE/Black Friday Sales/train.csv")

# Handling missing values
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Age', 'City_Category'], drop_first=True)

# Convert 'Stay_In_Current_City_Years' to numerical
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', 4)
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)

# Prepare data for linear regression
X = df.drop(columns=["Purchase", "Product_ID"])  # Drop the 'Product_ID' column
y = df["Purchase"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=1, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
# Train the Ridge regression model with hyperparameter tuning
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Train the Ridge regression model with the best hyperparameters
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)

# Predictions on test set
y_pred = ridge.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
'''
# Example: Using Random Forest Regressor instead of Ridge Regression
from sklearn.ensemble import RandomForestRegressor

# Define Random Forest Regressor model
rf = RandomForestRegressor()

# Define hyperparameters to search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_scaled, y_train)

# Best hyperparameters
best_params_rf = grid_search_rf.best_params_

# Train the Random Forest model with the best hyperparameters
rf_best = RandomForestRegressor(**best_params_rf)
rf_best.fit(X_train_scaled, y_train)

# Predictions on test set
y_pred_rf = rf_best.predict(X_test_scaled)

# Calculate Mean Squared Error for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Mean Squared Error (Random Forest):", mse_rf)
