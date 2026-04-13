from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

# ── Load ML components ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE, 'encoder.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# Columns in the EXACT order expected by the scaler
EXPECTED_COLS = [
    "property_type", "price_per_sqft", "area", "areaWithType",
    "bedRoom", "bathroom", "balcony", "additionalRoom",
    "floorNum", "facing", "agePossession"
]

# Extract encoder categories for the frontend
categories = {}
for feat in ["property_type", "areaWithType", "additionalRoom", "facing", "agePossession"]:
    if feat in encoders:
        categories[feat] = list(encoders[feat].classes_)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', categories=categories)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)

        property_type  = body['property_type']
        areaWithType   = body['areaWithType']
        additionalRoom = body.get('additionalRoom', 'not available')
        facing         = body['facing']
        agePossession  = body['agePossession']
        bedRoom        = float(body['bedRoom'])
        bathroom       = float(body['bathroom'])
        balcony        = float(body['balcony'])
        floorNum       = float(body['floorNum'])
        area           = float(body.get('area', 1000))
        price_per_sqft = float(body.get('price_per_sqft', 8000))

        data = {
            "property_type": property_type,
            "price_per_sqft": price_per_sqft,
            "area": area,
            "areaWithType": areaWithType,
            "bedRoom": bedRoom,
            "bathroom": bathroom,
            "balcony": balcony,
            "additionalRoom": additionalRoom,
            "floorNum": floorNum,
            "facing": facing,
            "agePossession": agePossession
        }

        df = pd.DataFrame([data], columns=EXPECTED_COLS)

        # Encode categorical columns
        for col, enc in encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])

        # Scale
        scaled = scaler.transform(df)

        # Predict (output assumed to be in Crores)
        pred_cr = float(model.predict(scaled)[0])

        return jsonify({
            'success': True,
            'predicted_price_cr': round(pred_cr, 4)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
