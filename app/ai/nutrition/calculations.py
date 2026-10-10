"""Scale nutrients by grams and aggregate meal totals."""

from __future__ import annotations

from typing import List

from app.ai.nutrition.portions import convert_to_grams


def calculate_food_nutrition(
    food_nutrition_data: dict,
    quantity_g: float,
) -> dict:
    """Scale retrieved per-100g nutrients to a positive gram quantity.

    Args:
        food_nutrition_data: Successful retrieval data containing food_name
            and calories_per_100g, protein_per_100g, carbs_per_100g,
            and fat_per_100g. The caller checks retrieval success.
        quantity_g: Food weight in grams, greater than zero.

    Returns:
        Food name, gram quantity, calories, protein_g, carbs_g, and fat_g.
        Numeric outputs are rounded to two decimal places.

    Raises:
        ValueError: If quantity_g is zero or negative.
    """
    if quantity_g <= 0:
        raise ValueError("Quantity must be greater than 0")

    factor = quantity_g / 100.0

    return {
        "food_name": food_nutrition_data["food_name"],
        "quantity_g": round(quantity_g, 2),
        "calories": round(food_nutrition_data["calories_per_100g"] * factor, 2),
        "protein_g": round(food_nutrition_data["protein_per_100g"] * factor, 2),
        "carbs_g": round(food_nutrition_data["carbs_per_100g"] * factor, 2),
        "fat_g": round(food_nutrition_data["fat_per_100g"] * factor, 2),
    }


def calculate_meal_total(
    foods: List[dict],
    food_nutrition_data: List[dict],
) -> dict:
    """Convert food quantities to grams and aggregate their nutrients.

    Args:
        foods: Parsed or generated food dictionaries containing
            food_name_original, food_name_en, quantity, and unit.
        food_nutrition_data: Retrieval results aligned with foods by index.
            Successful results provide per-100g nutrients and portions.

    When original_query is provided, it is checked against the original
    and English food names to detect mismatched retrieval results.
    Lookup failures, name mismatches, unavailable conversions, and
    non-positive converted quantities are recorded in unmatched_foods.
    Totals include successfully processed items only.

    Returns:
        A dictionary containing success, foods, unmatched_foods,
        total_calories, total_protein_g, total_carbs_g, and total_fat_g.
        Each processed food includes quantity_g, quantity_input, unit_used,
        and conversion_source for explaining quantity approximations.
        success is True when unmatched_foods is empty.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(foods) != len(food_nutrition_data):
        raise ValueError(
            f"Number of foods ({len(foods)}) must match "
            f"number of nutrition data ({len(food_nutrition_data)})"
        )

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    food_results: List[dict] = []
    unmatched_foods: List[dict] = []

    for meal_food, nutrition_data in zip(foods, food_nutrition_data):
        original_name = meal_food.get("food_name_original", "unknown")
        english_name = meal_food.get("food_name_en", "").strip().lower()

        # Skip failed retrievals and record their error codes.
        if not nutrition_data.get("success", False):
            unmatched_foods.append(
                {
                    "food_name_original": original_name,
                    "food_name_en": english_name,
                    "reason": nutrition_data.get("error", "unknown_error"),
                }
            )
            continue

        # Accept either the original or English name as the retrieval query.
        returned_query = (nutrition_data.get("original_query") or "").strip().lower()
        acceptable_names = {
            english_name,
            (meal_food.get("food_name_original") or "").strip().lower(),
        }
        acceptable_names.discard("")

        if (
            returned_query
            and acceptable_names
            and returned_query not in acceptable_names
        ):
            unmatched_foods.append(
                {
                    "food_name_original": original_name,
                    "food_name_en": english_name,
                    "reason": "food_mismatch",
                    "details": {
                        "task1_food_name_en": english_name,
                        "task1_food_name_original": meal_food.get("food_name_original"),
                        "task2_original_query": nutrition_data.get("original_query"),
                    },
                }
            )
            continue

        # Convert to grams using the shared portion policy.
        quantity = meal_food.get("quantity")
        unit = meal_food.get("unit", "gram")
        portions = nutrition_data.get("portions", [])

        conversion = convert_to_grams(quantity, unit, portions)

        if conversion is None:
            unmatched_foods.append(
                {
                    "food_name_original": original_name,
                    "food_name_en": english_name,
                    "reason": "unit_conversion_failed",
                    "details": {
                        "quantity": quantity,
                        "unit": unit,
                        "portions_available": len(portions) if portions else 0,
                    },
                }
            )
            continue

        quantity_g = conversion["grams"]

        # Record invalid converted weights without discarding other foods.
        try:
            result = calculate_food_nutrition(nutrition_data, quantity_g)
        except ValueError as exc:
            unmatched_foods.append(
                {
                    "food_name_original": original_name,
                    "food_name_en": english_name,
                    "reason": "invalid_quantity",
                    "details": {"quantity_g": quantity_g, "error": str(exc)},
                }
            )
            continue

        result["food_name_original"] = original_name
        result["food_name_en"] = english_name
        result["unit_used"] = unit
        result["quantity_input"] = quantity
        result["conversion_source"] = conversion["source"]

        food_results.append(result)

        total_calories += result["calories"]
        total_protein += result["protein_g"]
        total_carbs += result["carbs_g"]
        total_fat += result["fat_g"]

    return {
        "success": len(unmatched_foods) == 0,
        "foods": food_results,
        "unmatched_foods": unmatched_foods,
        "total_calories": round(total_calories, 2),
        "total_protein_g": round(total_protein, 2),
        "total_carbs_g": round(total_carbs, 2),
        "total_fat_g": round(total_fat, 2),
    }
