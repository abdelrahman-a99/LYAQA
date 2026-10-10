"""Collect nutrition data for food items through the retrieval tool."""

from typing import List

from app.ai.agent.tools import get_nutrition


def collect_nutrition_for_foods(foods: List[dict], intent: str) -> tuple:
    """
    Call get_nutrition for each food item individually.

    Returns (matched_foods, matched_nutrition, failed_names) where
    matched_* are parallel lists of the successful items only, and
    failed_names lists food_name_en values that could not be retrieved.
    """
    matched_foods = []
    matched_nutrition = []
    failed_names = []

    for food in foods:
        food_en = food.get("food_name_en", "")
        food_orig = food.get("food_name_original", food_en)
        result = get_nutrition(food_en, food_orig, intent=intent)
        if result["success"]:
            matched_foods.append(food)
            matched_nutrition.append(result["data"])
        else:
            failed_names.append(food_en or food_orig)

    return matched_foods, matched_nutrition, failed_names
