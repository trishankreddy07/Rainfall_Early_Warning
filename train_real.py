import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load the CSV Dataset
print("Loading dataset...")
df = pd.read_csv('weatherAUS.csv')

# 2. Select Relevant Features & Target
# Using key features that match weather station/satellite inputs
features = ['Humidity9am', 'Pressure9am', 'MinTemp', 'MaxTemp', 'Rainfall']
target = 'RainTomorrow'

# Drop rows where the target value itself is missing
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 3. Clean Categorical Target Variable
# Convert "Yes"/"No" text into numeric labels (1 and 0)
y = y.map({'No': 0, 'Yes': 1})

# Check missing value statistics before imputing
print("\nMissing values per column before preprocessing:")
print(X.isnull().sum())

# 4. Create an ML Pipeline with Missing Value Imputation
# Pipeline Step A: Impute missing numerical values using column MEDIAN
# Pipeline Step B: Train the Random Forest Classifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fills NaN with column median
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Fit the Pipeline (Imputer learns medians and Model learns patterns)
print("\nTraining model on processed data...")
pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))

# 8. Save the Pipeline Object (Imputer + Model together)
# Saving the entire pipeline ensures new data in Flask gets imputed automatically!
with open('rainfall_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Saved trained pipeline as 'rainfall_pipeline.pkl'")