"""
Authentication helpers: password hashing and JWT token management.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production-super-secret-key")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ------------------------------------------------------------------
# Password helpers
# ------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ------------------------------------------------------------------
# JWT helpers
# ------------------------------------------------------------------

def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None
