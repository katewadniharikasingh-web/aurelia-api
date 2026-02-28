from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow Lovable app to call this API

# Load model and scaler
model  = pickle.load(open('aurelia_model.pkl',  'rb'))
scaler = pickle.load(open('aurelia_scaler.pkl', 'rb'))

FEATURES = [
    'pain_score',
    'dyspareunia',
    'bowel_pain',
    'infertility',
    'progressive_pain',
    'menstrual_irregularity',
    'hormone_abnormality',
    'delta_T',
    'hrv'
]

@app.route('/')
def health():
    return jsonify({"status": "Aurelia API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate all features present
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build input array
        input_data = [[data[f] for f in FEATURES]]

        # Scale + predict
        input_scaled = scaler.transform(input_data)
        risk_score   = float(model.predict_proba(input_scaled)[0][1])

        # Risk level
        if risk_score >= 0.65:
            risk_level = "High Risk"
            message    = "Your symptom pattern suggests high risk. Please consult a gynecologist as soon as possible."
            color      = "red"
        elif risk_score >= 0.35:
            risk_level = "Moderate Risk"
            message    = "Some risk indicators detected. Consider speaking with a healthcare provider."
            color      = "orange"
        else:
            risk_level = "Low Risk"
            message    = "Low risk indicators detected. Continue monitoring your symptoms."
            color      = "green"

        return jsonify({
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "message":    message,
            "color":      color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
