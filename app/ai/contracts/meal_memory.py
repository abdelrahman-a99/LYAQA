"""HTTP snapshots of final meal candidates and calculated nutrition."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MealFood(SnapshotModel):
    food_name_original: str
    food_name_en: str = Field(min_length=1)
    quantity: float = Field(gt=0, strict=True)
    unit: str = Field(min_length=1)


class MealNutrition(SnapshotModel):
    total_calories: float = Field(ge=0, strict=True)
    total_protein_g: float = Field(ge=0, strict=True)
    total_carbs_g: float = Field(ge=0, strict=True)
    total_fat_g: float = Field(ge=0, strict=True)
    foods: list[dict] = Field(default_factory=list)
    unmatched_foods: list[dict] = Field(default_factory=list)


class MealSnapshot(SnapshotModel):
    meal_id: UUID
    meal_type: str | None = None
    target_calories: float = Field(gt=0, strict=True)
    min_protein_g: float | None = Field(default=None, ge=0, strict=True)
    foods: list[MealFood] = Field(min_length=1)
    nutrition: MealNutrition
    valid: bool = Field(strict=True)
    violations: list[str] = Field(default_factory=list)
    partial_failures: list[str] = Field(default_factory=list)
