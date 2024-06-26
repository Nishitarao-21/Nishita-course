
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

reg=LinearRegression()
df=pd.read_csv("C:/Users/NISHITA/Desktop/COURSE/archive (2)/HousingData.csv")


print(df.isnull().sum())


# Replace chained assignment with inplace method
df.loc[:, 'CRIM'] = df['CRIM'].fillna(df['CRIM'].mean())
df.loc[:, 'ZN'] = df['ZN'].fillna(df['ZN'].mean())
df.loc[:, 'INDUS'] = df['INDUS'].fillna(df['INDUS'].mean())
df.loc[:, 'CHAS'] = df['CHAS'].fillna(df['CHAS'].mean())
df.loc[:, 'AGE'] = df['AGE'].fillna(df['AGE'].mean())
df.loc[:, 'LSTAT'] = df['LSTAT'].fillna(df['LSTAT'].mean())

x=df.drop('MEDV',axis=1)
y=df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

train=reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

score=mean_squared_error(y_test,y_pred)

print(score)