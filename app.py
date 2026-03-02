from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model + scaler
model  = pickle.load(open('aurelia_model.pkl',  'rb'))
scaler = pickle.load(open('aurelia_scaler.pkl', 'rb'))

# Expected feature order for model
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

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        # ---- Default baseline values ----
        default_values = {
            'pain_score': 0,
            'dyspareunia': 0,
            'bowel_pain': 0,
            'infertility': 0,
            'progressive_pain': 0,
            'menstrual_irregularity': 0,
            'hormone_abnormality': 0,
            'delta_T': 0,
            'hrv': 0
        }

        # Merge incoming fields
        for key in default_values:
            if key in data:
                default_values[key] = float(data[key])

        # If ESP32 sends delta_temp instead of delta_T
        if 'delta_temp' in data:
            default_values['delta_T'] = float(data['delta_temp'])

        # If ESP32 sends bpm instead of hrv
        if 'bpm' in data:
            # Temporary simple mapping (can replace with real HRV later)
            default_values['hrv'] = float(data['bpm']) / 100.0

        # Build feature array in correct order
        input_data = np.array([[default_values[f] for f in FEATURES]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict probability
        risk_score = float(model.predict_proba(input_scaled)[0][1])

        # Risk categorization
        if risk_score >= 0.65:
            risk_level = "High Risk"
            message = "Your symptom pattern suggests high risk. Please consult a gynecologist."
            color = "red"
        elif risk_score >= 0.35:
            risk_level = "Moderate Risk"
            message = "Some risk indicators detected. Consider speaking with a healthcare provider."
            color = "orange"
        else:
            risk_level = "Low Risk"
            message = "Low risk indicators detected. Continue monitoring."
            color = "green"

        return jsonify({
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "message": message,
            "color": color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
