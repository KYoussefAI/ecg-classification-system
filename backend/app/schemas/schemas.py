"""
Pydantic schemas for request validation and response serialisation.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email:    EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    email:        str


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    The frontend sends ECG signal data as a flat list or nested list.

    signal_data: list of 1000 timesteps × 12 leads  →  shape (1000, 12)
                 Either a flat list of 12000 floats or a list of 1000 lists of 12 floats.
    patient_name: optional label stored with the record
    age:          patient age (stored, not used by model)
    sex:          patient sex  (stored, not used by model)
    """
    signal_data:  list[Any]  = Field(..., description="ECG signal (1000×12)")
    patient_name: str | None = Field(None, max_length=100)
    age:          int | None = Field(None, ge=0, le=130)
    sex:          str | None = Field(None, pattern="^(M|F|male|female|Male|Female)?$")


class ClassPrediction(BaseModel):
    class_name:  str   = Field(..., alias="class")
    description: str
    probability: float
    positive:    bool

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    id:               int | None = None
    predictions:      list[dict]
    top_class:        str
    confidence:       float
    positive_classes: list[str]
    patient_name:     str | None = None
    age:              int | None = None
    sex:              str | None = None
    created_at:       str | None = None


# ── History ───────────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    id:               int
    patient_name:     str | None
    age:              int | None
    sex:              str | None
    top_class:        str
    confidence:       float
    positive_classes: list[str]
    created_at:       str


# ── Stats ─────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_predictions: int
    class_distribution: dict[str, int]
    avg_confidence:     float
    recent_trend:       list[dict]  # [{date, count}]
