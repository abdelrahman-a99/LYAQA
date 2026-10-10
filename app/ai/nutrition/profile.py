"""Deterministic age, BMR, TDEE, calorie and macro calculations."""

from __future__ import annotations

import logging
from datetime import date

from app.ai.contracts.profile import (
    MANDATORY_PROFILE_FIELDS as _MANDATORY_PROFILE_FIELDS,
)
from app.ai.nutrition.constants import (
    _ACTIVITY_FACTORS,
    DEFAULT_DEFICIT_KCAL,
    DEFAULT_SURPLUS_KCAL,
    MIN_SAFE_CALORIES,
    PROTEIN_GRAMS_PER_KG,
)

logger = logging.getLogger(__name__)


def _validate_profile_fields(profile: dict) -> None:
    """
    Guard called at the top of every calculation function that requires a
    complete profile.

    Checks that `profile` is a non-None dict and that all 6 mandatory fields
    (birth_date, gender, weight, height, activity_level, goal) are present and
    non-None.

    Args:
        profile: dict produced by Task 1's get_user_profile() or NLPSession.

    Raises:
        ValueError: with a clear, field-specific message listing every missing
                    or None field so Task 4 can relay the exact error to the user.
    """
    if not isinstance(profile, dict):
        raise ValueError(
            "Profile must be a dict (received "
            f"{type(profile).__name__!r}). "
            "Pass the dict returned by Task 1's get_user_profile()."
        )

    missing = [
        field for field in _MANDATORY_PROFILE_FIELDS if profile.get(field) is None
    ]

    if missing:
        raise ValueError(
            f"Cannot calculate: the following mandatory profile field(s) are "
            f"missing or None: {missing}. "
            "Collect them via NLPSession onboarding before calling this function."
        )


def calculate_age(profile: dict) -> int:
    """
    Compute age in years from `birth_date` (YYYY-MM-DD).

    Correctly handles the case where the birthday hasn't happened yet this year.

    Args:
        profile: dict from Task 1's get_user_profile() / NLPSession.
                 Required field: birth_date (str, "YYYY-MM-DD", non-None).

    Raises:
        ValueError: if birth_date is None or any other mandatory field is missing.
    """
    _validate_profile_fields(profile)
    birth_date = date.fromisoformat(profile["birth_date"])
    today = date.today()

    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    logger.debug("Age computed: %s (birth_date=%s)", age, birth_date)
    return age


def calculate_bmr(profile: dict) -> float:
    """
    Basal Metabolic Rate using the Mifflin-St Jeor equation.

    Male:   BMR = 10*weight + 6.25*height − 5*age + 5
    Female: BMR = 10*weight + 6.25*height − 5*age − 161

    weight in kg, height in cm.

    Args:
        profile: dict from Task 1's get_user_profile() / NLPSession.
                 Required fields: birth_date, gender, weight, height,
                 activity_level, goal (all non-None).

    Raises:
        ValueError: if any mandatory field is None, or gender is not
                    'male'/'female'.
    """
    _validate_profile_fields(profile)
    age = calculate_age(profile)
    weight = profile["weight"]
    height = profile["height"]
    gender = profile["gender"].lower()

    if gender == "male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender == "female":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

    bmr = round(bmr, 2)
    logger.debug("BMR computed: %s kcal (gender=%s, age=%s)", bmr, gender, age)
    return bmr


def calculate_tdee(profile: dict) -> float:
    """
    Total Daily Energy Expenditure = BMR × activity_factor.

    Activity levels:
        sedentary    1.2     (no exercise / desk job)
        light        1.375   (1–3 days/week)
        moderate     1.55    (3–5 days/week)
        active       1.725   (6–7 days/week)
        very_active  1.9     (intense daily training)

    Args:
        profile: dict from Task 1's get_user_profile() / NLPSession.
                 Required fields: all 6 mandatory fields (non-None).
                 activity_level must be one of the 5 valid literals.

    Raises:
        ValueError: if any mandatory field is None or activity_level is invalid.
    """
    _validate_profile_fields(profile)
    bmr = calculate_bmr(profile)
    activity_level = profile["activity_level"].lower()

    if activity_level not in _ACTIVITY_FACTORS:
        raise ValueError(
            f"Invalid activity level: {activity_level!r}. "
            f"Must be one of {list(_ACTIVITY_FACTORS.keys())}"
        )

    tdee = round(bmr * _ACTIVITY_FACTORS[activity_level], 2)
    logger.debug("TDEE computed: %s kcal (activity=%s)", tdee, activity_level)
    return tdee


def calculate_target_calories(
    profile: dict,
    deficit_kcal: int = DEFAULT_DEFICIT_KCAL,
    surplus_kcal: int = DEFAULT_SURPLUS_KCAL,
) -> float:
    """
    Compute daily target calories based on TDEE and goal.

    Approach: additive adjustment (TDEE ± fixed kcal), not multiplicative.
    This keeps the deficit/surplus inside the evidence-based 300–500 kcal range
    regardless of the user's TDEE.

    Args:
        profile: dict from Task 1's get_user_profile() / NLPSession.
                 Required fields: all 6 mandatory fields (non-None).
                 goal must be one of: weight_loss, maintenance,
                 weight_gain, muscle_gain.
        deficit_kcal: kcal to subtract for weight_loss (default 400).
        surplus_kcal: kcal to add for weight_gain / muscle_gain (default 400).

    Returns:
        Target calories (float, rounded to 2 decimals).

    Raises:
        ValueError: if any mandatory field is None or goal is invalid.
    """
    _validate_profile_fields(profile)
    tdee = calculate_tdee(profile)
    goal = profile["goal"].lower()

    if goal == "maintenance":
        target = tdee
    elif goal == "weight_loss":
        target = tdee - deficit_kcal
    elif goal in ("weight_gain", "muscle_gain"):
        target = tdee + surplus_kcal
    else:
        raise ValueError(f"Invalid goal: {goal!r}")

    # Safety floor — do not go below medically safe minimums
    gender = profile["gender"].lower()
    floor = MIN_SAFE_CALORIES.get(gender)
    if floor is not None and target < floor:
        logger.warning(
            "Target %s kcal for goal=%s was clamped to the %s kcal floor.",
            round(target, 2),
            goal,
            floor,
        )
        target = floor

    return round(target, 2)


def calculate_protein_target(profile: dict) -> float:
    """Calculate daily protein grams from weight and goal only.

    Accept a partial profile containing those two fields. Return the
    unrounded value so callers can round after applying a meal share.

    Raise ValueError when weight is missing or the goal is unsupported.
    """
    weight = profile.get("weight")
    if weight is None:
        raise ValueError("Cannot calculate protein target without weight.")

    goal = (profile.get("goal") or "").lower()
    if goal not in PROTEIN_GRAMS_PER_KG:
        raise ValueError(f"Invalid goal: {goal!r}")

    return weight * PROTEIN_GRAMS_PER_KG[goal]


def calculate_macros(profile: dict) -> dict:
    """
    Compute daily macro targets (protein / carbs / fat) for the user's goal.

    Protein is set by goal (g per kg bodyweight).
    Fat is set at 25% of target calories.
    Carbs fill the remainder — clamped to 0 to prevent negative values.

    Args:
        profile: dict from Task 1's get_user_profile() / NLPSession.
                 Required fields: all 6 mandatory fields (non-None).

    Returns:
        {"calories": float, "protein_g": float, "carbs_g": float, "fat_g": float}

    Raises:
        ValueError: if any mandatory field is None or goal is invalid.
    """
    _validate_profile_fields(profile)
    target_calories = calculate_target_calories(profile)
    protein_grams = calculate_protein_target(profile)

    fat_calories = target_calories * 0.25
    fat_grams = fat_calories / 9

    protein_calories = protein_grams * 4

    # Clamp to 0 — a very low target + high protein can otherwise go negative.
    carbs_calories = max(0.0, target_calories - protein_calories - fat_calories)
    carbs_grams = carbs_calories / 4

    return {
        "calories": round(target_calories, 2),
        "protein_g": round(protein_grams, 2),
        "carbs_g": round(carbs_grams, 2),
        "fat_g": round(fat_grams, 2),
    }
