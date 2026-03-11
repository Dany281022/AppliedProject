# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
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
# Create FastAPI app
# -------------------------------
app = FastAPI(
    title="ML Prediction API",
    description="API for TeamDany Milestone 3",
    version="1.0"
)

# -------------------------------
# Load trained model safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier api/
model_path = os.path.join(BASE_DIR, "../models/initial_model.pkl")
print(f"Loading model from {model_path} ...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# -------------------------------
# Request & Response schemas
# -------------------------------
class PredictionRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_52: float

class PredictionResponse(BaseModel):
    prediction: float
    status: str

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "keys": {
            "OPENAI_API_KEY": OPENAI_API_KEY is not None,
            "CLERK_JWKS_URL": CLERK_JWKS_URL is not None,
            "CLERK_SECRET_KEY": CLERK_SECRET_KEY is not None,
            "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY": NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is not None
        },
        "model_loaded": model is not None
    }

# -------------------------------
# Model info endpoint
# -------------------------------
@app.get("/info")
def info():
    """Return model type and expected features"""
    return {
        "model_type": type(model).__name__,
        "features_expected": ["lag_1", "lag_2", "lag_52"],
        "version": "1.0"
    }

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make a prediction"""
    try:
        features = pd.DataFrame({
            "lag_1": [request.lag_1],
            "lag_2": [request.lag_2],
            "lag_52": [request.lag_52]
        })
        prediction = model.predict(features)
        return PredictionResponse(prediction=float(prediction[0]), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))