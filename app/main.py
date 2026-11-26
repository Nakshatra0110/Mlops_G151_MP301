"""MP301 prediction API. Request rules live in the Pydantic schema."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mp301.config_ml import load_params  # noqa: E402  (path set up above)
from src.mp301.model import FallbackPredictor  # noqa: E402

try:
    PARAMS = load_params()
except Exception:  # tests run with a minimal checkout
    PARAMS = {
        "model": {"fallback_waste_kg": 18.0},
        "api": {"max_cooked_kg": 1500, "min_plate_size_cm": 10},
    }

API_CFG = PARAMS.get("api", {})
MODEL_PATH = ROOT / PARAMS.get("paths", {}).get("model_out", "models/model.joblib")


class JsonFormatter(logging.Formatter):
    """One JSON object per line on stdout; no customer details are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key in ("prediction", "model", "status"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("mp301.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.propagate = False


class PredictRequest(BaseModel):
    students_registered: int = Field(..., ge=0)
    attendance_pct: float = Field(..., ge=0, le=100)
    menu_rating: float = Field(..., ge=0, le=5)
    temp_c: float = Field(..., ge=-50, le=60)
    cooked_kg: float = Field(..., ge=0, le=API_CFG.get("max_cooked_kg", 1500))
    plate_size_cm: float = Field(..., ge=API_CFG.get("min_plate_size_cm", 10), le=100)
    meal: str = Field(..., min_length=1)
    hostel: str = Field(..., min_length=1)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if MODEL_PATH.exists():
            _MODEL = joblib.load(MODEL_PATH)
        else:
            _MODEL = FallbackPredictor(PARAMS.get("model", {}).get("fallback_waste_kg", 18.0))
    return _MODEL


app = FastAPI(title="MP301 Hostel Mess Food Waste API", version="1.0")


@app.get("/health")
def health() -> dict:
    logger.info("health_check")
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    frame = pd.DataFrame([request.model_dump()])
    prediction = float(np.asarray(get_model().predict(frame), dtype=float).ravel()[0])
    logger.info("prediction", extra={"prediction": round(prediction, 4)})
    return {"prediction": prediction}
