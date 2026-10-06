"""
Authentication helpers: password hashing and JWT token management.
"""

import os
import logging
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

logger = logging.getLogger(__name__)


def validate_secret(secret, environment):
    invalid = (
        not secret
        or len(secret) < 32
        or len(set(secret)) < 8
        or any(
            word in secret.lower()
            for word in ("change-me", "placeholder", "your-secret", "replace-me")
        )
    )
    if environment != "development" and invalid:
        raise RuntimeError(
            "Production requires a generated SECRET_KEY of at least 32 characters"
        )
    if invalid:
        logger.warning(
            "Using an ephemeral development signing key; sessions expire on restart"
        )
        return secrets.token_urlsafe(48)
    return secret


SECRET_KEY = validate_secret(
    os.getenv("SECRET_KEY"), os.getenv("APP_ENV", "development")
)
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ------------------------------------------------------------------
# Password helpers
# ------------------------------------------------------------------


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if len(plain.encode()) > 72:
        return False
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ------------------------------------------------------------------
# JWT helpers
# ------------------------------------------------------------------


def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )
        return payload if isinstance(payload.get("sub"), str) else None
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None
