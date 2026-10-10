"""Existing meal and daily nutritional validation."""

from __future__ import annotations

from typing import List, Optional

from app.ai.nutrition.constants import (
    DAILY_CALORIE_TOLERANCE,
    DAILY_MACRO_TOLERANCE,
    MEAL_CALORIE_TOLERANCE,
)
from app.ai.nutrition.profile import (
    _validate_profile_fields,
    calculate_macros,
)
from app.ai.nutrition.restrictions import (
    _check_food_against_restrictions,
    _collect_user_restrictions,
)


def validate_daily_nutrition(
    meals: List[dict],
    profile: dict,
    calorie_tolerance: float = DAILY_CALORIE_TOLERANCE,
    fat_tolerance: float = DAILY_MACRO_TOLERANCE,
    carbs_tolerance: float = DAILY_MACRO_TOLERANCE,
) -> dict:
    """
    Validate a full day of meals against the user's targets and restrictions.

    Args:
        meals: list of `calculate_meal_total` outputs.
        profile: Task 1 profile dict (all 6 mandatory fields non-None).
                 Uses goal, weight, forbidden_foods, allergies,
                 dietary_restrictions.
        calorie_tolerance: ±% around target calories (default DAILY_CALORIE_TOLERANCE).
        fat_tolerance: ±% around target fat grams (default DAILY_MACRO_TOLERANCE).
        carbs_tolerance: ±% around target carbs grams (default DAILY_MACRO_TOLERANCE).

    Returns:
        {
          "valid":                bool,
          "daily_total":          {total_calories, total_protein_g, ...},
          "targets":              {target_calories, protein_g, carbs_g, fat_g},
          "checks":               {calories, protein, fat, carbs, restrictions},
          "restriction_violations": [...],
          "issues":               [str, ...]
        }

    Raises:
        ValueError: if any mandatory profile field is None.
    """
    _validate_profile_fields(profile)
    targets = calculate_macros(profile)
    target_calories = targets["calories"]

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    all_foods: List[dict] = []
    issues: List[str] = []

    # ── Aggregate ─────────────────────────────────────────────────────────
    for meal in meals:
        if not meal.get("success", False):
            unmatched = meal.get("unmatched_foods", [])
            if unmatched:
                for u in unmatched:
                    name = u.get("food_name_original", "unknown")
                    reason = u.get("reason", "unknown_error")
                    issues.append(
                        f"Could not compute nutrition for '{name}' ({reason})"
                    )
            else:
                issues.append("One of the meals could not be calculated")
            continue

        total_calories += meal["total_calories"]
        total_protein += meal["total_protein_g"]
        total_carbs += meal["total_carbs_g"]
        total_fat += meal["total_fat_g"]
        all_foods.extend(meal["foods"])

    # ── Calorie check ─────────────────────────────────────────────────────
    cal_min = target_calories * (1.0 - calorie_tolerance)
    cal_max = target_calories * (1.0 + calorie_tolerance)
    calories_valid = cal_min <= total_calories <= cal_max
    if not calories_valid:
        issues.append(
            f"Daily calories {round(total_calories, 2)} are outside "
            f"the target range ({round(cal_min, 2)}–{round(cal_max, 2)} kcal)"
        )

    # ── Protein check (minimum only) ──────────────────────────────────────
    protein_min = targets["protein_g"]
    protein_valid = total_protein >= protein_min
    if not protein_valid:
        issues.append(
            f"Daily protein {round(total_protein, 2)}g is below "
            f"the minimum target ({protein_min}g)"
        )

    # ── Fat check ─────────────────────────────────────────────────────────
    fat_min = targets["fat_g"] * (1.0 - fat_tolerance)
    fat_max = targets["fat_g"] * (1.0 + fat_tolerance)
    fat_valid = fat_min <= total_fat <= fat_max
    if not fat_valid:
        issues.append(
            f"Daily fat {round(total_fat, 2)}g is outside "
            f"the target range ({round(fat_min, 2)}–{round(fat_max, 2)} g)"
        )

    # ── Carbs check ───────────────────────────────────────────────────────
    carbs_min = targets["carbs_g"] * (1.0 - carbs_tolerance)
    carbs_max = targets["carbs_g"] * (1.0 + carbs_tolerance)
    carbs_valid = carbs_min <= total_carbs <= carbs_max
    if not carbs_valid:
        issues.append(
            f"Daily carbohydrates {round(total_carbs, 2)}g are outside "
            f"the target range ({round(carbs_min, 2)}–{round(carbs_max, 2)} g)"
        )

    # ── Restriction check (forbidden_foods + allergies + dietary_restrictions) ──
    user_restrictions = _collect_user_restrictions(profile)
    restriction_violations: List[dict] = []

    if user_restrictions:
        for food in all_foods:
            matched = _check_food_against_restrictions(food, user_restrictions)
            if matched:
                violation = {
                    "food_name_original": food.get("food_name_original"),
                    "food_name": food.get("food_name"),
                    "matched_restrictions": sorted(matched),
                }
                restriction_violations.append(violation)
                issues.append(
                    f"Food '{violation['food_name_original']}' violates "
                    f"restriction(s): {', '.join(violation['matched_restrictions'])}"
                )

    restrictions_valid = len(restriction_violations) == 0

    return {
        "valid": len(issues) == 0,
        "daily_total": {
            "total_calories": round(total_calories, 2),
            "total_protein_g": round(total_protein, 2),
            "total_carbs_g": round(total_carbs, 2),
            "total_fat_g": round(total_fat, 2),
        },
        "targets": {
            "target_calories": target_calories,
            "protein_g": targets["protein_g"],
            "carbs_g": targets["carbs_g"],
            "fat_g": targets["fat_g"],
        },
        "checks": {
            "calories": calories_valid,
            "protein": protein_valid,
            "fat": fat_valid,
            "carbs": carbs_valid,
            "restrictions": restrictions_valid,
        },
        "restriction_violations": restriction_violations,
        "issues": issues,
    }


def validate_meal_nutrition(
    meal: dict,
    profile: dict,
    target_calories: Optional[float] = None,
    calorie_tolerance: float = MEAL_CALORIE_TOLERANCE,
    min_protein_g: Optional[float] = None,
) -> dict:
    """
    Validate a SINGLE meal.

    Use case: "Make me a 400-calorie breakfast without eggs" → Task 4 builds
    the meal, then calls this to check it before returning to the user.

    NOTE: unlike validate_daily_nutrition(), this function does NOT require
    the 6 mandatory profile fields (birth_date, gender, weight, height,
    activity_level, goal).  It only reads restriction-related keys
    (forbidden_foods, allergies, dietary_restrictions) plus the
    target_calories and min_protein_g arguments.  This lets Task 4 validate a
    single-meal request BEFORE the user has completed full onboarding.

    Args:
        meal: output of `calculate_meal_total`.
        profile: profile dict — may be partial or even {}.  Only the keys
                 forbidden_foods, allergies, dietary_restrictions are read.
                 Pass {} if no restrictions are known yet.
        target_calories: optional per-meal calorie budget (e.g. 400).
                         If None, calorie check is skipped.
        calorie_tolerance: ±% around target_calories (default MEAL_CALORIE_TOLERANCE).
        min_protein_g: optional minimum protein floor for this meal (g).
                       If None, protein check is skipped.
                       Task 4 typically computes this as a fraction of
                       calculate_macros(profile)["protein_g"] / num_meals.

    Returns:
        {
          "valid":                  bool,
          "meal_total":             {total_calories, total_protein_g, ...} | None,
          "checks":                 {calories, protein, restrictions},
          "restriction_violations": [...],
          "issues":                 [str, ...]
        }
    """
    profile = profile or {}
    issues: List[str] = []

    if not meal.get("success", False):
        unmatched = meal.get("unmatched_foods", [])
        if unmatched:
            for u in unmatched:
                name = u.get("food_name_original", "unknown")
                reason = u.get("reason", "unknown_error")
                issues.append(f"Could not compute nutrition for '{name}' ({reason})")
        else:
            issues.append("Meal could not be calculated")
        return {
            "valid": False,
            "meal_total": None,
            "checks": {"calories": False, "protein": False, "restrictions": False},
            "restriction_violations": [],
            "issues": issues,
        }

    meal_total = {
        "total_calories": meal["total_calories"],
        "total_protein_g": meal["total_protein_g"],
        "total_carbs_g": meal["total_carbs_g"],
        "total_fat_g": meal["total_fat_g"],
    }

    # ── Calorie check (only if target_calories was given) ─────────────────
    if target_calories is not None:
        cal_min = target_calories * (1.0 - calorie_tolerance)
        cal_max = target_calories * (1.0 + calorie_tolerance)
        calories_valid = cal_min <= meal["total_calories"] <= cal_max

        if not calories_valid:
            issues.append(
                f"Meal calories {round(meal['total_calories'], 2)} are outside "
                f"the target range ({round(cal_min, 2)}–{round(cal_max, 2)} kcal)"
            )
    else:
        calories_valid = True

    # ── Protein floor check (only if min_protein_g was given) ─────────────
    if min_protein_g is not None:
        protein_valid = meal["total_protein_g"] >= min_protein_g
        if not protein_valid:
            issues.append(
                f"Meal protein {round(meal['total_protein_g'], 2)}g is below "
                f"the minimum ({min_protein_g}g)"
            )
    else:
        protein_valid = True

    # ── Restriction check ─────────────────────────────────────────────────
    user_restrictions = _collect_user_restrictions(profile)
    restriction_violations: List[dict] = []

    if user_restrictions:
        for food in meal.get("foods", []):
            matched = _check_food_against_restrictions(food, user_restrictions)
            if matched:
                violation = {
                    "food_name_original": food.get("food_name_original"),
                    "food_name": food.get("food_name"),
                    "matched_restrictions": sorted(matched),
                }
                restriction_violations.append(violation)
                issues.append(
                    f"Food '{violation['food_name_original']}' violates "
                    f"restriction(s): {', '.join(violation['matched_restrictions'])}"
                )

    restrictions_valid = len(restriction_violations) == 0

    return {
        "valid": len(issues) == 0,
        "meal_total": meal_total,
        "checks": {
            "calories": calories_valid,
            "protein": protein_valid,
            "restrictions": restrictions_valid,
        },
        "restriction_violations": restriction_violations,
        "issues": issues,
    }
