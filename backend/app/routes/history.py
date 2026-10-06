import json
from fastapi import APIRouter, Depends, HTTPException, Query
from app.database import get_db
from app.dependencies import get_current_user

router = APIRouter()


@router.get("/")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    cursor = await db.execute(
        """SELECT r.* FROM screening_results r JOIN users u ON r.user_id=u.id
                                 WHERE u.username=? ORDER BY r.id DESC LIMIT ?""",
        (user["sub"], limit),
    )
    return [
        {**json.loads(r["result"]), "id": r["id"], "created_at": r["created_at"]}
        for r in await cursor.fetchall()
    ]


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: int, user=Depends(get_current_user), db=Depends(get_db)
):
    cursor = await db.execute(
        "DELETE FROM screening_results WHERE id=? AND user_id=(SELECT id FROM users WHERE username=?)",
        (prediction_id, user["sub"]),
    )
    await db.commit()
    if not cursor.rowcount:
        raise HTTPException(404, "Result not found")
    return {"message": "Deleted"}
