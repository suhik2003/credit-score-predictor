import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Simulated dataset
data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23, 40],
    'income': [50000, 120000, 80000, 150000, 30000, 100000],
    'loan_amount': [20000, 50000, 30000, 70000, 15000, 40000],
    'credit_history': [1, 0.5, 0.8, 0.3, 1, 0.7],
    'score': ['Good', 'Poor', 'Average', 'Poor', 'Good', 'Average']
})

X = data.drop('score', axis=1)
y = data['score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/credit_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
