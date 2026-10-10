"""Validated meal selection and changes proposed by the language model."""

from pydantic import Field

from app.ai.contracts.meal_memory import MealFood, SnapshotModel


class MealSelection(SnapshotModel):
    meal_id: str | None = Field(...)


class MealChange(SnapshotModel):
    food_index: int | None = Field(..., ge=0, strict=True)
    food: MealFood | None = Field(...)


class MealChanges(SnapshotModel):
    changes: list[MealChange] = Field(..., max_length=10)
