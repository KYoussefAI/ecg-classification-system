from typing import Any
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_bytes(cls, value):
        if len(value.encode()) > 72:
            raise ValueError("Password may contain at most 72 UTF-8 bytes")
        return value


class LoginRequest(BaseModel):
    username: str = Field(max_length=50)
    password: str = Field(max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str


class PredictionRequest(BaseModel):
    signal_data: list[Any] = Field(max_length=1000)
    case_id: str | None = Field(None, max_length=100)
    leads: list[str] | None = None
    sample_rate: float = 100
    units: str = "mV"
    save_history: bool = False
    synthetic: bool = False


class ClassPrediction(BaseModel):
    class_name: str = Field(alias="class")
    description: str
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    positive: bool


class PredictionResponse(BaseModel):
    id: int | None = None
    case_id: str | None = None
    created_at: str | None = None
    model_version: str
    predictions: list[ClassPrediction]
    positive_classes: list[str]
    signal_quality: dict
    preprocessing: dict
    warnings: list[str]
    disclaimer: str
