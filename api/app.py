from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

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

class PredictionRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_52: float

class PredictionResponse(BaseModel):
    prediction: float
    status: str

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "version": "1.0"}

@app.get("/info")
def info():
    return {"model_type": type(model).__name__, "features_expected": ["lag_1", "lag_2", "lag_52"], "version": "1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = pd.DataFrame({"lag_1": [request.lag_1], "lag_2": [request.lag_2], "lag_52": [request.lag_52]})
        prediction = model.predict(features)
        return PredictionResponse(prediction=float(prediction[0]), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)