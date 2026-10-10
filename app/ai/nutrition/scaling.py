"""Rescale meal quantities to a calorie target."""

from __future__ import annotations

from typing import List


def scale_meal_to_target(
    foods: List[dict],
    current_calories: float,
    target_calories: float,
    min_quantity_g: float = 20.0,
    max_scale_factor: float = 2.0,
    round_to_g: float = 5.0,
) -> dict:
    """Rescale food quantities toward a calorie target using a bounded factor.

    Args:
        foods: Food dictionaries used to calculate current_calories.
        current_calories: Calculated calorie total before scaling.
        target_calories: Requested calorie total.
        min_quantity_g: Minimum gram-equivalent quantity before rounding
            for gram and kilogram units. Other units have a 0.25-unit floor.
        max_scale_factor: Upper factor limit; the lower limit is its reciprocal.
        round_to_g: Rounding step for gram and kilogram quantities.

    Returns:
        scaled_foods, scale_factor, clamped, and achievable.
        achievable is False for non-positive current calories or when the
        requested scale factor is clamped. Rounding and quantity floors
        can still prevent an exact target match when achievable is True.

    The function adjusts quantities only. The caller must recompute nutrients
    and validate the resulting meal. It makes no LLM or USDA calls.
    """
    if not current_calories or current_calories <= 0:
        return {
            "scaled_foods": foods,
            "scale_factor": 1.0,
            "clamped": False,
            "achievable": False,
        }

    raw_factor = target_calories / current_calories
    factor = max(1.0 / max_scale_factor, min(max_scale_factor, raw_factor))
    clamped = abs(factor - raw_factor) > 1e-9

    scaled_foods = []
    for food in foods:
        new_food = dict(food)
        qty = food.get("quantity", 0) or 0
        unit = (food.get("unit") or "").strip().lower()

        new_qty = qty * factor

        if unit in ("gram", "kilogram"):
            floor = min_quantity_g if unit == "gram" else min_quantity_g / 1000.0
            new_qty = max(new_qty, floor)
            step = round_to_g if unit == "gram" else round_to_g / 1000.0
            new_qty = round(new_qty / step) * step
        else:
            # Non-weight units (piece, cup, etc.) — scale but never below a
            # sane minimum fraction of one unit.
            new_qty = max(new_qty, 0.25)
            new_qty = round(new_qty, 2)

        new_food["quantity"] = new_qty
        scaled_foods.append(new_food)

    return {
        "scaled_foods": scaled_foods,
        "scale_factor": round(factor, 4),
        "clamped": clamped,
        "achievable": not clamped,
    }
