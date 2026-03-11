from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import os
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")

# -------------------------------
# Create Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load your trained model
# -------------------------------
model_path = r"../models/initial_model.pkl"
print(f"Loading model from {model_path} ...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# -------------------------------
# Health check endpoint
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "keys": {
            "OPENAI_API_KEY": OPENAI_API_KEY is not None,
            "CLERK_JWKS_URL": CLERK_JWKS_URL is not None,
            "CLERK_SECRET_KEY": CLERK_SECRET_KEY is not None,
            "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY": NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is not None
        },
        "model_loaded": model is not None
    })

# -------------------------------
# Model info endpoint
# -------------------------------
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_type": type(model).__name__,
        "features_expected": ["lag_1", "lag_2", "lag_52"],  # match model training
        "version": "1.0"
    })

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # -------------------------------
        # Extract required features (from training)
        # -------------------------------
        lag_1 = data.get("lag_1")
        lag_2 = data.get("lag_2")
        lag_52 = data.get("lag_52")

        # Check for missing features
        missing = [f for f in ["lag_1", "lag_2", "lag_52"] if data.get(f) is None]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Build DataFrame for prediction
        features = pd.DataFrame({
            "lag_1": [lag_1],
            "lag_2": [lag_2],
            "lag_52": [lag_52]
        })

        # Make prediction
        prediction = model.predict(features)

        return jsonify({
            "prediction": float(prediction[0]),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# -------------------------------
# Root endpoint
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "API is running. Use /health, /info, or /predict"

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    