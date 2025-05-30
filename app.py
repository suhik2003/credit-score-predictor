from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('model/credit_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['income']),
            float(request.form['loan_amount']),
            float(request.form['credit_history'])
        ]

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        # Create a simple bar chart
        plt.bar(['Good', 'Average', 'Poor'], model.predict_proba(scaled_features)[0], color=['green', 'orange', 'red'])
        plt.ylabel('Probability')
        plt.title('Credit Score Prediction')
        chart_path = os.path.join('static', 'chart.png')
        plt.savefig(chart_path)
        plt.close()

        return render_template('result.html', prediction=prediction, chart_path=chart_path)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
