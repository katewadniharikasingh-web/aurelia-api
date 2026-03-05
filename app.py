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

# Expected feature order
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

        print("Received:", data)

        # Default values
        default_values = {
            'pain_score': 0,
            'dyspareunia': 0,
            'bowel_pain': 0,
            'infertility': 0,
            'progressive_pain': 0,
            'menstrual_irregularity': 0,
            'hormone_abnormality': 0,
            'delta_T': 0,
            'hrv': 40
        }

        # Merge incoming JSON
        for key in default_values:
            if key in data:
                default_values[key] = float(data[key])

        # ESP32 compatibility
        if 'delta_temp' in data:
            default_values['delta_T'] = float(data['delta_temp'])

        if 'bpm' in data:
            default_values['hrv'] = float(data['bpm']) / 100.0

        # Build feature array
        input_data = np.array([[default_values[f] for f in FEATURES]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # ML prediction
        risk_score = float(model.predict_proba(input_scaled)[0][1])

        # ---------------------------
        # Symptom scoring
        # ---------------------------

        symptom_sum = (
            default_values['pain_score'] +
            default_values['dyspareunia'] +
            default_values['bowel_pain'] +
            default_values['infertility'] +
            default_values['progressive_pain'] +
            default_values['menstrual_irregularity'] +
            default_values['hormone_abnormality']
        )

        # ---------------------------
        # Hybrid Decision System
        # ---------------------------

        if symptom_sum >= 5 or risk_score >= 0.65:

            risk_level = "High Risk"
            message = "Your symptom pattern suggests high risk. Please consult a gynecologist."
            color = "red"

        elif symptom_sum >= 3 or risk_score >= 0.35:

            risk_level = "Moderate Risk"
            message = "Some risk indicators detected. Consider speaking with a healthcare provider."
            color = "orange"

        else:

            risk_level = "Low Risk"
            message = "Low risk indicators detected. Continue monitoring."
            color = "green"

        # Return result
        return jsonify({

            "risk_score": round(risk_score * 100, 1),
            "risk_level": risk_level,
            "message": message,
            "color": color,

            # Sensor values returned to frontend
            "delta_T": round(default_values['delta_T'], 2),
            "hrv": round(default_values['hrv'], 2)

        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
