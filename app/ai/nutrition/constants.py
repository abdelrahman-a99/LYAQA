"""Existing calorie policy, tolerances and activity multipliers."""

from __future__ import annotations

DEFAULT_DEFICIT_KCAL = 400  # weight_loss
DEFAULT_SURPLUS_KCAL = 400  # weight_gain / muscle_gain

PROTEIN_GRAMS_PER_KG = {
    "weight_loss": 2.0,
    "weight_gain": 1.6,
    "maintenance": 1.6,
    "muscle_gain": 2.0,
}

MIN_SAFE_CALORIES = {
    "male": 1500,
    "female": 1200,
}

MEAL_CALORIE_TOLERANCE = 0.15
DAILY_CALORIE_TOLERANCE = 0.20
DAILY_MACRO_TOLERANCE = 0.20

# Standard activity multipliers applied to BMR to get TDEE.
_ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}
