"""Strict structured outputs for NLP parsing."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

INTENT_TYPES = Literal[
    "calculate_calories",  # log eaten meal
    "calculate_macros",  # macro breakdown request
    "food_nutrition",  # nutrition query about a single food
    "meal_recommendation",  # single meal suggestion with constraints
    "meal_edit",  # change a previously suggested meal
    "daily_plan",  # full day/multi-day plan
    "food_substitution",  # substitutes for X
    "fitness_nutrition_question",  # workout/sports nutrition (RAG trigger)
    "update_profile",  # profile metrics & onboarding
    "general_chat",  # greetings / chit-chat
]


class FoodItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    food_name_original: str = Field(
        ..., description="Original food name as written by the user"
    )
    food_name_en: str = Field(
        ..., description="Standard English USDA-compatible food name"
    )
    quantity: float = Field(..., description="Numeric quantity e.g. 0.5, 1.0, 200.0")
    unit: str = Field(
        ...,
        description="Unit: gram, cup, piece, slice, can, bottle, tablespoon, bowl, loaf, milliliter",
    )


class UserProfileExtract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    birth_date: Optional[str] = Field(..., description="YYYY-MM-DD or null")
    gender: Optional[Literal["male", "female"]] = Field(
        ..., description="male, female, or null"
    )
    weight: Optional[float] = Field(..., description="Weight in kg or null")
    height: Optional[float] = Field(..., description="Height in cm or null")
    activity_level: Optional[
        Literal["sedentary", "light", "moderate", "active", "very_active"]
    ] = Field(..., description="Activity level or null")
    goal: Optional[
        Literal["weight_loss", "maintenance", "weight_gain", "muscle_gain"]
    ] = Field(..., description="User goal or null")
    dietary_restrictions: List[str] = Field(
        ..., description="e.g. ['keto', 'vegan'] or []"
    )
    forbidden_foods: List[str] = Field(
        ..., description="Foods user avoids e.g. ['eggs'] or []"
    )
    preferred_foods: List[str] = Field(..., description="Foods user prefers or []")
    allergies: List[str] = Field(
        ..., description="Medical allergies e.g. ['dairy'] or []"
    )


class NLPOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: INTENT_TYPES = Field(
        ..., description="User intent according to project specifications"
    )
    language_detected: Literal["arabic", "english", "mixed"] = Field(
        ..., description="Detected input language"
    )
    raw_text: str = Field(..., description="Original user input, echoed verbatim")

    # ── Extended fields for meal-planning / recommendation ──
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack", "all_day"]] = (
        Field(
            ...,
            description="Meal requested: breakfast, lunch, dinner, snack, all_day, or null",
        )
    )
    target_calories: Optional[float] = Field(
        ...,
        description="Explicit target calories (e.g. 400.0 for '400-calorie breakfast') or null",
    )
    requested_substitution: Optional[str] = Field(
        ...,
        description="Food to substitute when intent is food_substitution (e.g. 'eggs') or null",
    )

    foods: List[FoodItem] = Field(..., description="Extracted food items ([] if none)")

    # No `default=None` here — Groq's strict mode requires every field to be `required` in the schema.
    # "Optional" is expressed by the TYPE being nullable (Optional[...]), not by a Python-side default value.
    profile_data: Optional[UserProfileExtract] = Field(
        ...,
        description="Full profile object whenever ANY profile info is present, otherwise null",
    )
