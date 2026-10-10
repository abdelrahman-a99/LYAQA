from typing import List

from app.ai.agent.food_lookup import collect_nutrition_for_foods
from app.ai.agent.tools import calculate_meal_nutrition
from app.ai.presentation.responses import (
    format_calorie_macro_response,
    format_partial_failure,
)


def handle_calorie_macro_flow(
    foods: List[dict], profile: dict, language: str, intent: str
) -> str:
    """
    calculate_calories / calculate_macros / food_nutrition flow (spec section 4).

    Tool 2 per food → Tool 4. No validate_plan, no RAG.
    Partial failures reported explicitly; total rejection only when list is empty.
    Note: calculate_macros follows this same flow — Task 3's calculate_macros()
    function is never called for this intent (spec decision #4).
    """
    if not foods:
        if language == "arabic":
            return "لم أتمكن من استخراج أي أطعمة من رسالتك. يرجى ذكر الأطعمة والكميات."
        return "I could not extract any food items from your message. Please specify foods and quantities."

    matched_foods, matched_nutrition, failed = collect_nutrition_for_foods(
        foods, intent
    )

    if not matched_foods:
        if language == "arabic":
            return "لم أتمكن من العثور على بيانات غذائية لأي من الأطعمة المذكورة. يرجى المحاولة مجدداً بأسماء مختلفة."
        return "Could not find nutrition data for any of the mentioned foods. Please try again with different names."

    meal_wrap = calculate_meal_nutrition(
        matched_foods, matched_nutrition, intent=intent
    )
    if not meal_wrap["success"]:
        if language == "arabic":
            return "حدث خطأ في حساب القيم الغذائية."
        return "An error occurred while calculating nutrition values."

    meal = meal_wrap["data"]
    partial_note = format_partial_failure(failed, language)

    return format_calorie_macro_response(meal, intent, language, partial_note)
