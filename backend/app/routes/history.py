"""History route — GET /api/history"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.schemas import HistoryEntry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=list[HistoryEntry])
async def get_history(
    limit: int = 50,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    row = await db.execute("SELECT id FROM users WHERE username=?", (user["sub"],))
    user_row = await row.fetchone()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    cursor = await db.execute(
        """
        SELECT id, patient_name, age, sex, top_class, confidence, predictions, created_at
        FROM predictions
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_row["id"], limit)
    )
    rows = await cursor.fetchall()

    results = []
    for r in rows:
        preds = json.loads(r["predictions"]) if r["predictions"] else []
        positive = [p["class"] for p in preds if p.get("positive")]
        results.append(HistoryEntry(
            id=r["id"],
            patient_name=r["patient_name"],
            age=r["age"],
            sex=r["sex"],
            top_class=r["top_class"],
            confidence=r["confidence"],
            positive_classes=positive,
            created_at=r["created_at"],
        ))
    return results


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    row = await db.execute("SELECT id FROM users WHERE username=?", (user["sub"],))
    user_row = await row.fetchone()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    await db.execute(
        "DELETE FROM predictions WHERE id=? AND user_id=?",
        (prediction_id, user_row["id"])
    )
    await db.commit()
    return {"message": "Deleted"}
