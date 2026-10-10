"""Apply changes against original food indices without mutating saved context."""

from copy import deepcopy

from app.ai.contracts.meal_edits import MealChanges


def apply_food_changes(foods: list[dict], changes: MealChanges) -> list[dict]:
    replacements = {}
    additions = []
    for change in changes.changes:
        index = change.food_index
        food = change.food.model_dump() if change.food is not None else None
        if index is None:
            if food is None:
                raise ValueError("An addition requires a food")
            additions.append(food)
        else:
            if index >= len(foods) or index in replacements:
                raise ValueError("Unknown or repeated food index")
            replacements[index] = food

    result = []
    for index, original in enumerate(foods):
        food = replacements.get(index, original)
        if food is not None:
            result.append(deepcopy(food))
    result.extend(additions)
    if not result or len(result) > 10:
        raise ValueError("The edited meal must contain between one and ten foods")
    return result
