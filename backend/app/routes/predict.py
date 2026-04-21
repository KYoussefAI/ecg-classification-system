"""
Prediction route.
POST /api/predict
  • Accepts ECG signal as JSON array OR file (.npy)
  • Runs inference via ModelService
  • Persists result to DB if user is authenticated
"""

import json
import logging
import numpy as np

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import get_db
from app.schemas.schemas import PredictionRequest, PredictionResponse
from app.services.model_service import ModelService
from app.services.auth_service import decode_token

logger = logging.getLogger(__name__)
router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


def _parse_signal(raw: list) -> np.ndarray:
    """Convert JSON signal payload to numpy array of shape (1000, 12)."""
    arr = np.array(raw, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size != 12000:
            raise ValueError(f"Flat signal must have 12000 values, got {arr.size}")
        arr = arr.reshape(1000, 12)

    elif arr.ndim == 2:
        if arr.shape == (12, 1000):
            arr = arr.T
        elif arr.shape != (1000, 12):
            raise ValueError(f"2-D signal must be (1000,12) or (12,1000), got {arr.shape}")

    else:
        raise ValueError(f"Signal must be 1-D or 2-D, got {arr.ndim}-D")

    return arr


@router.post("/", response_model=PredictionResponse)
async def predict(
    body: PredictionRequest,
    db=Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    # ---- parse signal ----
    try:
        signal = _parse_signal(body.signal_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


    # ---- run model ----
    try:
        svc = ModelService.get_instance()
        result = svc.predict(signal)
    except Exception as e:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # ---- optional: identify user ----
    user_id = None
    if credentials and credentials.credentials:
        payload = decode_token(credentials.credentials)
        if payload:
            row = await db.execute(
                "SELECT id FROM users WHERE username=?", (payload["sub"],)
            )
            user_row = await row.fetchone()
            if user_row:
                user_id = user_row["id"]

    patient_name = body.patient_name
    age = body.age
    sex = body.sex

    # ---- save prediction ----
    cursor = await db.execute(
        """
        INSERT INTO predictions
          (user_id, patient_name, age, sex, signal_shape, predictions, top_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            patient_name,
            age,
            sex,
            str(signal.shape),
            json.dumps(result["predictions"]),
            result["top_class"],
            result["confidence"],
        )
    )
    await db.commit()
    record_id = cursor.lastrowid

    row = await db.execute("SELECT created_at FROM predictions WHERE id=?", (record_id,))
    saved = await row.fetchone()

    return PredictionResponse(
        id=record_id,
        predictions=result["predictions"],
        top_class=result["top_class"],
        confidence=result["confidence"],
        positive_classes=result["positive_classes"],
        patient_name=patient_name,
        age=age,
        sex=sex,
        created_at=saved["created_at"] if saved else None,
    )