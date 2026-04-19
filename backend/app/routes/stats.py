"""Stats route — GET /api/stats"""

import json
import logging
from collections import defaultdict
from fastapi import APIRouter, Depends

from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.schemas import StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=StatsResponse)
async def get_stats(user=Depends(get_current_user), db=Depends(get_db)):
    row = await db.execute("SELECT id FROM users WHERE username=?", (user["sub"],))
    user_row = await row.fetchone()
    user_id = user_row["id"] if user_row else None

    # All predictions for this user
    cursor = await db.execute(
        "SELECT top_class, confidence, predictions, created_at FROM predictions WHERE user_id=?",
        (user_id,)
    )
    rows = await cursor.fetchall()

    total          = len(rows)
    class_dist     = defaultdict(int)
    conf_sum       = 0.0
    trend_counts   = defaultdict(int)

    for r in rows:
        class_dist[r["top_class"]] += 1
        conf_sum += r["confidence"] or 0.0
        date_str = str(r["created_at"])[:10]  # YYYY-MM-DD
        trend_counts[date_str] += 1

    avg_conf = round(conf_sum / total, 4) if total else 0.0

    # Sort trend by date
    trend = sorted(
        [{"date": d, "count": c} for d, c in trend_counts.items()],
        key=lambda x: x["date"]
    )[-30:]  # last 30 days

    return StatsResponse(
        total_predictions=total,
        class_distribution=dict(class_dist),
        avg_confidence=avg_conf,
        recent_trend=trend,
    )
