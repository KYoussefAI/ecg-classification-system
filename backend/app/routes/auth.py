"""Auth routes: /api/auth/register  /api/auth/login"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends

from app.database import get_db
from app.schemas.schemas import RegisterRequest, LoginRequest, TokenResponse
from app.services.auth_service import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db=Depends(get_db)):
    # Check username / email uniqueness
    row = await db.execute(
        "SELECT id FROM users WHERE username=? OR email=?",
        (body.username, body.email)
    )
    if await row.fetchone():
        raise HTTPException(status_code=400, detail="Username or email already registered")

    hashed = hash_password(body.password)
    await db.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (body.username, body.email, hashed)
    )
    await db.commit()

    token = create_access_token({"sub": body.username, "email": body.email})
    return TokenResponse(access_token=token, username=body.username, email=body.email)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db=Depends(get_db)):
    row = await db.execute(
        "SELECT username, email, password FROM users WHERE username=?",
        (body.username,)
    )
    user = await row.fetchone()

    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user["username"], "email": user["email"]})
    return TokenResponse(access_token=token, username=user["username"], email=user["email"])
