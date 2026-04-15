from flask import Flask, request, jsonify, send_file
import numpy as np
import joblib
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import certifi
import os

# ==============================
# Initialize App
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# MongoDB Connection
# ==============================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://madhan:diet@cluster0.jascrcq.mongodb.net/?appName=Cluster0"
)

client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where()
)

db = client["water_quality_db"]
collection = db["predictions"]

# ==============================
# Load Model & Scaler
# ==============================
try:
    model = joblib.load("models/water_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Model & Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ==============================
# Routes
# ==============================

@app.route('/')
def home():
    return send_file("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received:", data)

        # Extract features
        features = [
            data['ph'], data['Hardness'], data['Solids'],
            data['Chloramines'], data['Sulfate'], data['Conductivity'],
            data['Organic_carbon'], data['Trihalomethanes'], data['Turbidity']
        ]

        # Convert + Scale
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        result = "Safe" if prediction == 1 else "Not Safe"

        # ==============================
        # Store in MongoDB (UTC - BEST PRACTICE)
        # ==============================
        record = {
            "input": data,
            "prediction": result,
            "confidence": float(prob),
            "timestamp": datetime.utcnow()   # ✅ FIXED
        }

        inserted = collection.insert_one(record)
        print("Inserted ID:", inserted.inserted_id)

        return jsonify({
            "prediction": result,
            "confidence": round(float(prob), 4)
        })

    except Exception as e:
        print("Prediction ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    try:
        records = list(collection.find().sort("timestamp", -1).limit(10))

        for r in records:
            r["_id"] = str(r["_id"])
            if isinstance(r["timestamp"], datetime):
                r["timestamp"] = r["timestamp"].isoformat()

        return jsonify(records)

    except Exception as e:
        print("History ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ==============================
# Run Server (IMPORTANT FIX)
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # ✅ RENDER FIX
    app.run(host="0.0.0.0", port=port)