"""Format user-facing nutrition responses."""

from typing import List


def format_quantity_approximation_note(meal: dict, language: str) -> str:
    """Explain approximate conversions used in the final calculated meal."""
    arabic = language == "arabic"
    reasons = {
        "household_fallback": (
            "تقدير عام لوزن الوحدة المنزلية"
            if arabic
            else "generic household weight estimate"
        ),
        "liquid_density_approx": (
            "بافتراض كثافة 1 جرام لكل ملليلتر"
            if arabic
            else "assuming a density of 1 g/ml"
        ),
    }
    entries = []

    for food in meal.get("foods", []):
        reason = reasons.get(food.get("conversion_source"))
        if reason is None:
            continue

        name = food.get("food_name_original") if arabic else food.get("food_name_en")
        name = (
            name
            or food.get("food_name_en")
            or food.get("food_name")
            or ("طعام" if arabic else "food")
        )

        quantity = food.get("quantity_input")
        unit = food.get("unit_used")
        grams = food.get("quantity_g")
        conversion = ""

        if quantity is not None and unit and grams is not None:
            gram_label = "جرام" if arabic else "g"
            conversion = f"{quantity:g} {unit} ≈ {grams:g} {gram_label}; "

        entries.append(f"  - {name}: {conversion}{reason}.")

    if not entries:
        return ""

    header = (
        "ملاحظة: السعرات والماكروز تتضمن كميات تقديرية:"
        if arabic
        else "Note: Calories and macros include estimated quantities:"
    )
    return "\n".join([header, *entries])


def format_partial_failure(failed_names: List[str], language: str) -> str:
    """Build the explicit partial-failure line shown to the user."""
    if not failed_names:
        return ""
    names = ", ".join(failed_names)
    if language == "arabic":
        return f"ملاحظة: لم يتمكن النظام من حساب بيانات: {names}."
    return f"Note: Could not compute nutrition for: {names}."


def format_calorie_macro_response(
    meal: dict, intent: str, language: str, partial_note: str
) -> str:
    """Format the final user-facing string for calorie/macro/food_nutrition intents."""
    cals = round(meal.get("total_calories", 0), 1)
    prot = round(meal.get("total_protein_g", 0), 1)
    carbs = round(meal.get("total_carbs_g", 0), 1)
    fat = round(meal.get("total_fat_g", 0), 1)

    if language == "arabic":
        if intent == "calculate_macros":
            lines = [
                "المغذيات الكبرى لوجبتك:",
                f"  بروتين:     {prot} جرام",
                f"  كربوهيدرات: {carbs} جرام",
                f"  دهون:       {fat} جرام",
                f"  (إجمالي:    {cals} سعرة)",
            ]
        else:
            lines = [
                f"إجمالي السعرات: {cals} سعرة حرارية",
                f"  بروتين:     {prot} جرام",
                f"  كربوهيدرات: {carbs} جرام",
                f"  دهون:       {fat} جرام",
            ]
    else:
        if intent == "calculate_macros":
            lines = [
                "Macronutrient breakdown:",
                f"  Protein:      {prot} g",
                f"  Carbohydrates:{carbs} g",
                f"  Fat:          {fat} g",
                f"  (Total:       {cals} kcal)",
            ]
        else:
            lines = [
                f"Total calories: {cals} kcal",
                f"  Protein:       {prot} g",
                f"  Carbohydrates: {carbs} g",
                f"  Fat:           {fat} g",
            ]

    if partial_note:
        lines.append("")
        lines.append(partial_note)

    approximation_note = format_quantity_approximation_note(meal, language)
    if approximation_note:
        lines.append("")
        lines.append(approximation_note)

    return "\n".join(lines)


def format_meal_recommendation_response(
    flow: dict, target_calories: float, language: str, valid: bool
) -> str:
    """Build the final user-facing string for a meal recommendation."""
    report = flow.get("report", {})
    meal = flow.get("meal_result", {})
    partial_note = format_partial_failure(flow.get("partial_failures", []), language)

    cals = round(meal.get("total_calories", 0), 1)
    prot = round(meal.get("total_protein_g", 0), 1)
    carbs = round(meal.get("total_carbs_g", 0), 1)
    fat = round(meal.get("total_fat_g", 0), 1)

    foods = flow.get("foods", [])
    food_lines = [
        f"  - {f.get('food_name_en', '')} — {f.get('quantity')} {f.get('unit')}"
        for f in foods
    ]

    if language == "arabic":
        lines = [f"وجبة مقترحة (~{target_calories:.0f} سعرة):"]
        lines += [
            f"  - {f.get('food_name_original', f.get('food_name_en', ''))} "
            f"— {f.get('quantity')} {f.get('unit')}"
            for f in foods
        ]
        lines += [
            "",
            f"السعرات:       {cals} سعرة",
            f"بروتين:        {prot} جرام",
            f"كربوهيدرات:    {carbs} جرام",
            f"دهون:          {fat} جرام",
        ]
        if not valid:
            lines.append("")
            lines.append("تنبيه: لم تتحقق جميع الأهداف الغذائية بعد المحاولات المتاحة.")
            issues = report.get("violations", [])
            for iss in issues:
                lines.append(f"  - {iss}")
    else:
        lines = [f"Suggested meal (~{target_calories:.0f} kcal):"]
        lines += food_lines
        lines += [
            "",
            f"Calories:     {cals} kcal",
            f"Protein:      {prot} g",
            f"Carbohydrates:{carbs} g",
            f"Fat:          {fat} g",
        ]
        if not valid:
            lines.append("")
            lines.append(
                "Note: Not all nutritional targets were met after the allowed revision attempts."
            )
            issues = report.get("violations", [])
            for iss in issues:
                lines.append(f"  - {iss}")

    if partial_note:
        lines.append("")
        lines.append(partial_note)

    approximation_note = format_quantity_approximation_note(meal, language)
    if approximation_note:
        lines.append("")
        lines.append(approximation_note)

    return "\n".join(lines)


def format_daily_plan_response(
    meal_results: dict,
    day_totals: dict,
    daily_valid: bool,
    day_report: dict,
    partial_failures: list,
    language: str,
    daily_target: float,
) -> str:
    """Build the final user-facing string for daily_plan."""
    lines = []
    has_approximations = False

    if language == "arabic":
        lines.append(f"خطة يومية مقترحة (~{daily_target:.0f} سعرة في اليوم):\n")
        meal_labels = {
            "breakfast": "الفطار",
            "lunch": "الغداء",
            "dinner": "العشاء",
            "snack": "وجبة خفيفة",
        }
    else:
        lines.append(f"Suggested daily plan (~{daily_target:.0f} kcal/day):\n")
        meal_labels = {
            "breakfast": "Breakfast",
            "lunch": "Lunch",
            "dinner": "Dinner",
            "snack": "Snack",
        }

    for meal_name, flow in meal_results.items():
        label = meal_labels.get(meal_name, meal_name.capitalize())
        lines.append(f"{label}:")
        if not flow.get("success"):
            if language == "arabic":
                lines.append("  لم يتمكن النظام من إنشاء هذه الوجبة.")
            else:
                lines.append("  Could not generate this meal.")
        else:
            m = flow.get("meal_result", {})
            foods = flow.get("foods", [])
            for f in foods:
                name = (
                    f.get("food_name_original", f.get("food_name_en", ""))
                    if language == "arabic"
                    else f.get("food_name_en", "")
                )
                lines.append(f"  - {name} — {f.get('quantity')} {f.get('unit')}")
            lines.append(
                f"  ({round(m.get('total_calories', 0), 1)} kcal | "
                f"P:{round(m.get('total_protein_g', 0), 1)}g "
                f"C:{round(m.get('total_carbs_g', 0), 1)}g "
                f"F:{round(m.get('total_fat_g', 0), 1)}g)"
            )

            approximation_note = format_quantity_approximation_note(m, language)
            if approximation_note:
                lines.append(approximation_note)
                has_approximations = True

            if not flow.get("valid"):
                if language == "arabic":
                    lines.append("  تنبيه: لم تتحقق جميع الأهداف الغذائية لهذه الوجبة.")
                else:
                    lines.append("  Note: Not all targets met for this meal.")
        lines.append("")

    # Daily totals
    tc = round(day_totals.get("total_calories", 0), 1)
    tp = round(day_totals.get("total_protein_g", 0), 1)
    tc2 = round(day_totals.get("total_carbs_g", 0), 1)
    tf = round(day_totals.get("total_fat_g", 0), 1)

    if language == "arabic":
        lines.append(
            f"الإجمالي اليومي: {tc} سعرة | بروتين {tp}ج | كارب {tc2}ج | دهون {tf}ج"
        )
    else:
        lines.append(
            f"Daily total: {tc} kcal | Protein {tp}g | Carbs {tc2}g | Fat {tf}g"
        )

    if has_approximations:
        lines.append(
            "الإجمالي اليومي يتضمن كميات تقديرية."
            if language == "arabic"
            else "The daily total includes estimated quantities."
        )

    # If daily check fails despite individual meals passing, show issues plainly (spec v1)
    if not daily_valid:
        issues = day_report.get("violations", [])
        if issues:
            if language == "arabic":
                lines.append("\nملاحظات على الإجمالي اليومي:")
            else:
                lines.append("\nDaily total issues:")
            for iss in issues:
                lines.append(f"  - {iss}")

    if partial_failures:
        lines.append("")
        lines.append(format_partial_failure(partial_failures, language))

    return "\n".join(lines)
