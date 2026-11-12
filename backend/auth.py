"""Authentication and authorization utilities for ResQTrack.

Provides JWT generation and verification, password verification,
FastAPI dependencies for role-based authorization, and rate-limiting.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from backend.database import (
    connect,
    get_user_by_email,
    get_user_by_id,
    row_to_dict,
    verify_password,
)

# JWT Secret - configurable via environment variable
JWT_SECRET = os.environ.get(
    "RESQTRACK_JWT_SECRET",
    "resqtrack-emergency-secret-key-32bytes-long-2026-production",
)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

security_bearer = HTTPBearer(auto_error=False)

# Simple sliding window rate limiter: key -> list of timestamps
_RATE_LIMITS: Dict[str, list[float]] = defaultdict(list)


def check_rate_limit(key: str, max_requests: int = 15, window_seconds: int = 60) -> None:
    now = time.time()
    timestamps = [t for t in _RATE_LIMITS[key] if now - t < window_seconds]
    timestamps.append(now)
    _RATE_LIMITS[key] = timestamps
    if len(timestamps) > max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and try again.",
        )


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2)
    phone: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)


class UserProvisionRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    role: str = "RESPONDER"  # CITIZEN, ADMIN, RESPONDER
    phone: Optional[str] = None
    responder_id: Optional[str] = None


# ============================================================
# TOKEN OPERATIONS
# ============================================================

def create_access_token(payload: dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = payload.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


# ============================================================
# FASTAPI DEPENDENCIES
# ============================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    connection = connect()
    try:
        user = get_user_by_id(connection, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account no longer exists.",
            )
        if user.get("status") != "ACTIVE":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is suspended or inactive.",
            )
        user.pop("password_hash", None)
        return user
    finally:
        connection.close()


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> Optional[dict[str, Any]]:
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        connection = connect()
        try:
            user = get_user_by_id(connection, user_id)
            if user and user.get("status") == "ACTIVE":
                user.pop("password_hash", None)
                return user
            return None
        finally:
            connection.close()
    except Exception:
        return None


def require_role(*allowed_roles: str) -> Callable:
    async def role_checker(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        user_role = current_user.get("role", "").upper()
        allowed = [r.upper() for r in allowed_roles]
        if user_role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {', '.join(allowed_roles)}.",
            )
        return current_user
    return role_checker


# Shortcuts
require_admin = require_role("ADMIN")
require_responder = require_role("RESPONDER", "ADMIN")
require_citizen = require_role("CITIZEN", "ADMIN")
