#BANKRUPTCY PREDICTION DATASET!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


train_df = pd.read_csv("C:/Users/NISHITA/Desktop/Rawieee/data.csv")


X = train_df.drop(columns=['Bankrupt?'])  # Features
y = train_df['Bankrupt?']  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred_train = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", train_accuracy)

feature_importances = model.feature_importances_


feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


print("Important Features:")
print(feature_importance_df)