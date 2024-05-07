import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Generate synthetic data (replace with your actual data)
data = {
    'Age': [30, 45, 22, 50, 28, 35, 40, 55, 60, 25, 27, 59],
    'Income': [50000, 75000, 40000, 90000, 55000, 60000, 70000, 100000, 120000, 35000, 40000,65000],
    'Conversation': [
        "Considering life insurance for my new business loan.",
        "Life insurance is important for my family's future.",
        "Not interested in life insurance at the moment.",
        "Already have a comprehensive life insurance plan.",
        "Looking into life insurance options for my parents.",
        "Exploring life insurance as a new parent.",
        "Life insurance seems too expensive for my budget.",
        "I have life insurance through my employer.",
        "My spouse handles our life insurance decisions.",
        "I'm not sure if I need life insurance at this stage in life.",
        "Not interested in this.",
        "Interested in this life insurance policy."
    ],
    'Lead': [1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,1]  # 1 for potential lead, 0 for not
}

df = pd.DataFrame(data)

# Split data into features and target
X = df[['Age', 'Income', 'Conversation']]
y = df['Lead']

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
