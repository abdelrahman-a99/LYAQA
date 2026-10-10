"""Build meal memory from final calculation results, never from answer text."""

from copy import deepcopy
from uuid import uuid4

from app.ai.contracts.meal_memory import MealSnapshot


def build_meal_snapshot(
    flow: dict,
    target_calories: float,
    meal_type: str | None,
    min_protein_g: float | None,
) -> dict | None:
    """Copy a final candidate, including failed checks and quantity approximations."""
    meal = flow.get("meal_result")
    foods = flow.get("foods")
    if not flow.get("success") or not meal or not foods:
        return None

    report = flow.get("report", {})
    partial_failures = flow.get("partial_failures", [])
    unmatched_foods = meal.get("unmatched_foods", [])
    snapshot = MealSnapshot(
        meal_id=uuid4(),
        meal_type=meal_type,
        target_calories=target_calories,
        min_protein_g=min_protein_g,
        foods=[
            {
                "food_name_original": food.get("food_name_original")
                or food["food_name_en"],
                "food_name_en": food["food_name_en"],
                "quantity": food["quantity"],
                "unit": food["unit"],
            }
            for food in foods
        ],
        nutrition={
            "total_calories": meal["total_calories"],
            "total_protein_g": meal["total_protein_g"],
            "total_carbs_g": meal["total_carbs_g"],
            "total_fat_g": meal["total_fat_g"],
            "foods": deepcopy(meal.get("foods", [])),
            "unmatched_foods": deepcopy(unmatched_foods),
        },
        valid=bool(
            flow.get("valid")
            and report.get("valid")
            and meal.get("success", True)
            and not partial_failures
            and not unmatched_foods
        ),
        violations=deepcopy(report.get("violations", [])),
        partial_failures=deepcopy(partial_failures),
    )
    return snapshot.model_dump(mode="json")
