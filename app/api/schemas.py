"""Validate messages and caller-owned conversation context."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.contracts.meal_memory import MealSnapshot


class HealthResponse(BaseModel):
    status: str


class ProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    birth_date: str | None = None
    gender: Literal["male", "female"] | None = None
    weight: float | None = Field(default=None, gt=0, strict=True)
    height: float | None = Field(default=None, gt=0, strict=True)
    activity_level: (
        Literal["sedentary", "light", "moderate", "active", "very_active"] | None
    ) = None
    goal: Literal["weight_loss", "maintenance", "weight_gain", "muscle_gain"] | None = (
        None
    )
    dietary_restrictions: list[str] = Field(default_factory=list, max_length=50)
    forbidden_foods: list[str] = Field(default_factory=list, max_length=50)
    preferred_foods: list[str] = Field(default_factory=list, max_length=50)
    allergies: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value or parsed >= date.today():
            raise ValueError("birth_date must be a past date in YYYY-MM-DD format")
        return value


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(strict=True, min_length=1, max_length=10000)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("history content must contain non-whitespace characters")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(strict=True, min_length=1, max_length=4000)
    profile: ProfileContext | None = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    meal_context: list[MealSnapshot] = Field(default_factory=list, max_length=4)

    @field_validator("meal_context")
    @classmethod
    def reject_duplicate_meal_ids(cls, value: list[MealSnapshot]) -> list[MealSnapshot]:
        identifiers = [meal.meal_id for meal in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("meal_context must contain unique meal IDs")
        return value

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value


class ChatResponse(BaseModel):
    response: str
    intent: str
    status: str
    language: str
    profile: dict[str, Any]
    revision_count: int
    meals: list[MealSnapshot] = Field(default_factory=list, max_length=4)
