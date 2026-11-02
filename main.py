from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import HeartFeatures, HealthResponse, InfoResponse, PredictionResponse

MODEL_PATH = Path(__file__).parent / "model" / "heart_model.joblib"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = joblib.load(MODEL_PATH)  # load once at startup
    yield
    state.clear()


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


def _bundle():
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["bundle"]


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok" if "bundle" in state else "model_not_loaded"}


@app.get("/info", response_model=InfoResponse)
def info():
    b = _bundle()
    final_step = b["model"].steps[-1][1]
    return {
        "model_type": type(final_step).__name__,
        "model_name": b["name"],
        "n_features": len(b["features"]),
        "features": b["features"],
        "target": "heart_disease (True = presence, False = absence)",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    b = _bundle()
    row = pd.DataFrame([payload.model_dump()])[b["features"]]  # enforce training column order
    proba = float(b["model"].predict_proba(row)[0, 1])
    return {"heart_disease": proba >= 0.5, "probability": round(proba, 4)}