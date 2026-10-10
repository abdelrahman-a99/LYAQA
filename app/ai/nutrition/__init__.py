from app.ai.nutrition.calculations import (
    calculate_food_nutrition,
    calculate_meal_total,
)
from app.ai.nutrition.constants import (
    DAILY_CALORIE_TOLERANCE,
    DAILY_MACRO_TOLERANCE,
    DEFAULT_DEFICIT_KCAL,
    DEFAULT_SURPLUS_KCAL,
    MEAL_CALORIE_TOLERANCE,
    MIN_SAFE_CALORIES,
)
from app.ai.nutrition.profile import (
    calculate_age,
    calculate_bmr,
    calculate_macros,
    calculate_target_calories,
    calculate_tdee,
)
from app.ai.nutrition.scaling import scale_meal_to_target
from app.ai.nutrition.validation import (
    validate_daily_nutrition,
    validate_meal_nutrition,
)

__all__ = [
    "DEFAULT_DEFICIT_KCAL",
    "DEFAULT_SURPLUS_KCAL",
    "MIN_SAFE_CALORIES",
    "MEAL_CALORIE_TOLERANCE",
    "DAILY_CALORIE_TOLERANCE",
    "DAILY_MACRO_TOLERANCE",
    "calculate_age",
    "calculate_bmr",
    "calculate_tdee",
    "calculate_target_calories",
    "calculate_macros",
    "calculate_food_nutrition",
    "calculate_meal_total",
    "scale_meal_to_target",
    "validate_daily_nutrition",
    "validate_meal_nutrition",
]
