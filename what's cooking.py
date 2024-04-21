#WHATS COOKING DATASET!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Load the dataset
train_df = pd.read_json("C:/Users/NISHITA/Desktop/Rawieee/train.json")
test_df = pd.read_json("C:/Users/NISHITA/Desktop/Rawieee/train.json")

# Split the data into features (X) and labels (y)
X_train = train_df['ingredients'].astype(str)
y_train = train_df['cuisine']

X_test = test_df['ingredients'].astype(str)
y_test = test_df['cuisine']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Initialize the classifier
classifier = LogisticRegression()

# Create a pipeline
pipeline = Pipeline([('tfidf', tfidf_vectorizer),
                     ('clf', classifier)])

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



