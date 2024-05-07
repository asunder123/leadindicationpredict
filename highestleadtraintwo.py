import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Read data from the CSV file
data = pd.read_csv('Data.csv')

# Split data into features and target
X = data[['Age', 'Income', 'Conversation']]
y = data['Lead']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a ColumnTransformer to transform different types of columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age', 'Income']),
        ('text', TfidfVectorizer(), 'Conversation')
    ]
)

# Create a pipeline that first transforms the data and then fits a classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Save the trained model to a file
joblib.dump(pipeline, 'insurance_lead_model.joblib')
