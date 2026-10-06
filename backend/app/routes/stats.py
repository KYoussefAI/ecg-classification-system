import json
from collections import Counter
from fastapi import APIRouter, Depends
from app.database import get_db
from app.dependencies import get_current_user

router = APIRouter()


@router.get("/")
async def get_stats(user=Depends(get_current_user), db=Depends(get_db)):
    cursor = await db.execute(
        "SELECT result FROM screening_results WHERE user_id=(SELECT id FROM users WHERE username=?)",
        (user["sub"],),
    )
    results = [json.loads(r["result"]) for r in await cursor.fetchall()]
    return {
        "total_predictions": len(results),
        "class_distribution": dict(
            Counter(c for r in results for c in r["positive_classes"])
        ),
    }
