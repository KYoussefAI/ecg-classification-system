import json
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database import get_db
from app.schemas.schemas import PredictionRequest, PredictionResponse
from app.services.auth_service import decode_token
from app.services.input_service import MAX_UPLOAD_BYTES, parse_file
from app.services.model_service import ModelService, ModelUnavailable

router = APIRouter()
bearer = HTTPBearer(auto_error=False)


async def execute_prediction(body, db, credentials):
    user_id = None
    if credentials:
        payload = decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(401, "Invalid or expired token")
        row = await db.execute(
            "SELECT id FROM users WHERE username=?", (payload["sub"],)
        )
        user = await row.fetchone()
        if not user:
            raise HTTPException(401, "Unknown account")
        user_id = user["id"]
    if body.save_history and user_id is None:
        raise HTTPException(401, "Sign in to explicitly save a de-identified result")
    try:
        result = await run_in_threadpool(
            ModelService.get_instance().predict,
            body.signal_data,
            leads=body.leads,
            sample_rate=body.sample_rate,
            units=body.units,
        )
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc
    result.update({"id": None, "case_id": body.case_id, "created_at": None})
    if body.synthetic:
        result["warnings"].insert(
            0, "Synthetic input: these outputs have no patient interpretation."
        )
    if body.save_history:
        cursor = await db.execute(
            "INSERT INTO screening_results (user_id, case_id, model_version, result) VALUES (?, ?, ?, ?)",
            (user_id, body.case_id, result["model_version"], json.dumps(result)),
        )
        await db.commit()
        result["id"] = cursor.lastrowid
        row = await db.execute(
            "SELECT created_at FROM screening_results WHERE id=?", (cursor.lastrowid,)
        )
        result["created_at"] = (await row.fetchone())["created_at"]
    return result


@router.post("/", response_model=PredictionResponse)
async def predict(
    body: PredictionRequest,
    db=Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    return await execute_prediction(body, db, credentials)


async def read_upload(file, sample_rate, units):
    try:
        return parse_file(
            await file.read(MAX_UPLOAD_BYTES + 1), file.filename, sample_rate, units
        )
    except (ValueError, KeyError, TypeError, EOFError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@router.post("/validate-file")
async def validate_file(
    file: UploadFile = File(...),
    sample_rate: float = Form(100),
    units: str = Form("mV"),
):
    return await read_upload(file, sample_rate, units)


@router.post("/file", response_model=PredictionResponse)
async def predict_file(
    file: UploadFile = File(...),
    sample_rate: float = Form(100),
    units: str = Form("mV"),
    case_id: str | None = Form(None, max_length=100),
    save_history: bool = Form(False),
    db=Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    parsed = await read_upload(file, sample_rate, units)
    body = PredictionRequest(
        **{
            k: parsed[k]
            for k in ("signal_data", "leads", "sample_rate", "units", "synthetic")
        },
        case_id=case_id,
        save_history=save_history,
    )
    return await execute_prediction(body, db, credentials)
