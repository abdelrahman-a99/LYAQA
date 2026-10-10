"""Present recalculated meal edits and explicit rejection reasons."""

from app.ai.presentation.responses import format_calorie_macro_response


def format_meal_edit_response(meal: dict, report: dict, language: str) -> str:
    arabic = language in ("arabic", "mixed")
    valid = report["valid"]
    header = (
        "الوجبة المعدلة:"
        if arabic and valid
        else "التعديل لم يحقق الشروط؛ الوجبة السابقة لم تتغير."
        if arabic
        else "Updated meal:"
        if valid
        else "The edit did not meet the checks; the previous meal is unchanged."
    )
    lines = [header]
    for food in meal["foods"]:
        name = food.get("food_name_original") if arabic else food.get("food_name_en")
        lines.append(
            f"  - {name or food['food_name']}: "
            f"{food['quantity_input']:g} {food['unit_used']}"
        )
    lines += [
        "",
        format_calorie_macro_response(
            meal, "calculate_macros", "arabic" if arabic else "english", ""
        ),
    ]
    lines.extend(f"  - {issue}" for issue in report["violations"])
    return "\n".join(lines)
