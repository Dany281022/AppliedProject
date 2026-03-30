from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time
import numpy as np

app = FastAPI(
    title="ML Prediction API",
    description="Weekly Sales Prediction API - TeamDany Milestone 4",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
print(f"Loading model from {model_path} ...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# Pre-warm the model at startup to avoid cold-start latency on first request
FEATURES = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_12', 'lag_26', 'lag_52',
            'ma_4', 'ma_12', 'std_4', 'weekofyear', 'month', 'year']

_dummy = pd.DataFrame([{f: 0.0 for f in FEATURES}])
_dummy['weekofyear'] = 1
_dummy['month'] = 1
_dummy['year'] = 2024
model.predict(_dummy)
print("Model warm-up complete!")

class PredictionRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_4: float
    lag_8: float
    lag_12: float
    lag_26: float
    lag_52: float
    ma_4: float
    ma_12: float
    std_4: float
    weekofyear: int
    month: int
    year: int

class PredictionResponse(BaseModel):
    prediction: float
    confidence: str
    status: str
    response_time_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0"
    }

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "features_expected": model.feature_names_in_.tolist(),
        "feature_importances": model.feature_importances_.tolist(),
        "num_features": len(model.feature_names_in_),
        "version": "1.0",
        "confidence_note": "Confidence intervals are not standard for regression. A +/- 10% prediction range is returned instead.",
        "performance": {
            "rmse": 2034159.61,
            "mae": 1472778.63,
            "r2": 0.3025
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        start = time.time()

        features = pd.DataFrame([{
            "lag_1": request.lag_1,
            "lag_2": request.lag_2,
            "lag_4": request.lag_4,
            "lag_8": request.lag_8,
            "lag_12": request.lag_12,
            "lag_26": request.lag_26,
            "lag_52": request.lag_52,
            "ma_4": request.ma_4,
            "ma_12": request.ma_12,
            "std_4": request.std_4,
            "weekofyear": request.weekofyear,
            "month": request.month,
            "year": request.year
        }])

        prediction = model.predict(features)
        elapsed = round((time.time() - start) * 1000, 2)

        pred_value = float(prediction[0])
        lower = pred_value * 0.90
        upper = pred_value * 1.10
        confidence_str = f"USD {lower:,.0f} - USD {upper:,.0f} (+-10% interval)"

        return PredictionResponse(
            prediction=pred_value,
            confidence=confidence_str,
            status="success",
            response_time_ms=elapsed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)