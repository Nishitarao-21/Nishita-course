
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import mean_squared_error

# give or make an example data
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

# find the mean squared error the following function is used for that
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error:", mse)

# Read function to read the dataset
df = pd.read_csv("C:/Users/NISHITA/Desktop/COURSE/walmart-recruiting-store-sales-forecasting/train/train.csv")
data = df.to_numpy()

# Hand or find the  missing values
print(df.isnull().sum())

# using Plot boxplot before handling missing values
def plot_boxplot(data, column):
    sns.boxplot(y=column, data=data)
    plt.title(f"Boxplot of {column}")
    plt.show()

graphs = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
for column in graphs:
    plot_boxplot(df, column)

# Converting the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# One-hot encode 'Date' column
encoded_df = pd.get_dummies(df, columns=['Date'])

# Preparing data for linear regression purpose
X = encoded_df.drop('Weekly_Sales', axis=1)
y = encoded_df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg.fit(X_train_scaled, y_train)

y_pred_scaled = reg.predict(X_test_scaled)

# the following method is use find the mean error of the data given
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print("Mean Squared Error (scaled features):", mse_scaled)
