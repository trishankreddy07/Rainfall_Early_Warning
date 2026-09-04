import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Generate Synthetic Training Data (Simulating weather parameters)
np.random.seed(42)
n_samples = 1000

humidity = np.random.uniform(40, 100, n_samples)
pressure = np.random.uniform(980, 1030, n_samples)
radar_reflectivity = np.random.uniform(10, 65, n_samples)

# High humidity + low pressure + high radar reflectivity = High Risk (1)
risk_score = (humidity * 0.4) + (radar_reflectivity * 0.6) - (pressure * 0.05)
target = (risk_score > 30).astype(int)  # 1 for High Risk / Heavy Rain, 0 for Low Risk

df = pd.DataFrame({
    'humidity': humidity,
    'pressure': pressure,
    'radar': radar_reflectivity,
    'risk': target
})

# 2. Split Features and Target
X = df[['humidity', 'pressure', 'radar']]
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check performance
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 4. Save the Model to Disk using Pickle
with open('rainfall_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as 'rainfall_model.pkl'")