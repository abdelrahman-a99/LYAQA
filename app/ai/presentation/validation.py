"""Existing meal and daily nutritional validation."""

from __future__ import annotations

from typing import List, Optional

from app.ai.nutrition.constants import (
    DAILY_CALORIE_TOLERANCE,
    DAILY_MACRO_TOLERANCE,
    MEAL_CALORIE_TOLERANCE,
)
from app.ai.nutrition.validation import (
    validate_daily_nutrition,
    validate_meal_nutrition,
)


def validate_daily_nutrition_report(
    meals: List[dict],
    profile: dict,
    calorie_tolerance: float = DAILY_CALORIE_TOLERANCE,
    fat_tolerance: float = DAILY_MACRO_TOLERANCE,
    carbs_tolerance: float = DAILY_MACRO_TOLERANCE,
) -> dict:
    """
    Thin wrapper around validate_daily_nutrition() that returns the fixed
    structured format required by Task 4:

        {"valid": bool, "violations": [str, ...]}

    Task 4 must not need to parse or reformat this output.  Each string in
    `violations` is a clear, actionable description of what failed.

    The underlying bound logic (protein minimum, fat/carbs bounds, calorie
    range, restriction checks) is unchanged — only the output shape differs.

    Args:
        meals: list of `calculate_meal_total` outputs.
        profile: Task 1 profile dict (all 6 mandatory fields non-None).
        calorie_tolerance: ±% around target calories (default DAILY_CALORIE_TOLERANCE).
        fat_tolerance: ±% around target fat grams (default DAILY_MACRO_TOLERANCE).
        carbs_tolerance: ±% around target carbs grams (default DAILY_MACRO_TOLERANCE).

    Returns:
        {"valid": bool, "violations": [str, ...]}
    """
    result = validate_daily_nutrition(
        meals, profile, calorie_tolerance, fat_tolerance, carbs_tolerance
    )
    return {
        "valid": result["valid"],
        "violations": result["issues"],
    }


def validate_meal_nutrition_report(
    meal: dict,
    profile: dict,
    target_calories: Optional[float] = None,
    calorie_tolerance: float = MEAL_CALORIE_TOLERANCE,
    min_protein_g: Optional[float] = None,
) -> dict:
    """
    Thin wrapper around validate_meal_nutrition() that returns the fixed
    structured format required by Task 4:

        {"valid": bool, "violations": [str, ...]}

    Task 4 must not need to parse or reformat this output.  Each string in
    `violations` is a clear, actionable description of what failed.

    The underlying bound logic (calorie range, protein floor, restriction
    checks) is unchanged — only the output shape differs.

    Args:
        meal: output of `calculate_meal_total`.
        profile: profile dict — may be partial or {}.
        target_calories: optional per-meal calorie budget.
                         If None, calorie check is skipped.
        calorie_tolerance: ±% around target_calories (default MEAL_CALORIE_TOLERANCE).
        min_protein_g: optional minimum protein floor for this meal (g).
                       If None, protein check is skipped.

    Returns:
        {"valid": bool, "violations": [str, ...]}
    """
    result = validate_meal_nutrition(
        meal, profile, target_calories, calorie_tolerance, min_protein_g
    )
    return {
        "valid": result["valid"],
        "violations": result["issues"],
    }
