import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# lotate the location of your dataset
df = pd.read_csv("C:/Users/NISHITA/Desktop/COURSE/Wine Quality/winequalityN.csv")

import numpy as np
from sklearn.metrics import mean_squared_error


true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

#mean squared error method
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error:", mse)



X = df.drop(columns=["alcohol"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = pd.get_dummies(X_train, columns=['type'])
X_test = pd.get_dummies(X_test, columns=['type'])

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

#the function on putting the prediction on
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


