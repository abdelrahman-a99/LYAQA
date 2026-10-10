import logging

from app.ai.nutrition import (
    DAILY_CALORIE_TOLERANCE,
    DAILY_MACRO_TOLERANCE,
    MEAL_CALORIE_TOLERANCE,
    calculate_age,
    calculate_bmr,
    calculate_macros,
    calculate_meal_total,
    calculate_target_calories,
    calculate_tdee,
    scale_meal_to_target,
    validate_meal_nutrition,
)
from app.ai.nutrition.portions import convert_to_grams
from app.ai.nutrition.restrictions import _restriction_keys

if __name__ == "__main__":
    from app.config import load_environment

    load_environment(".env")

    import io
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )

    print("=" * 70)
    print("Task 3 — Nutrition Engine self-test (including P0/P1 fixes)")
    print("=" * 70)

    # ── Demo profile ──────────────────────────────────────────────────────
    profile = {
        "birth_date": "2004-05-10",
        "gender": "male",
        "weight": 90.0,
        "height": 175.0,
        "activity_level": "moderate",
        "goal": "weight_loss",
        "dietary_restrictions": [],
        "forbidden_foods": ["بيض"],
        "preferred_foods": [],
        "allergies": ["dairy"],
    }

    # 1) Targets
    print("\n[1] Targets for weight_loss, male, 90kg")
    print(f"  Age      : {calculate_age(profile)}")
    print(f"  BMR      : {calculate_bmr(profile)} kcal")
    print(f"  TDEE     : {calculate_tdee(profile)} kcal")
    print(f"  Target   : {calculate_target_calories(profile)} kcal")
    print(f"  Macros   : {json.dumps(calculate_macros(profile))}")

    # 2) P0-1 — punctuation-safe restriction matching
    print("\n[2] P0-1 Restriction matching on USDA-style comma-separated names")
    for name, expected in [
        ("Egg, whole, raw, fresh", "eggs"),
        ("Cheese, cheddar", "dairy"),
        ("Fish, salmon, Atlantic, raw", "fish"),
        ("Milk, whole, 3.25% milkfat", "dairy"),
    ]:
        keys = _restriction_keys(name)
        status = "OK" if expected in keys else "FAIL"
        print(f"  [{status}] '{name}' -> {keys}")

    # 3) P0-2 — original_query accepts Arabic food_name_original
    print("\n[3] P0-2 original_query with Arabic name")
    foods_ar = [
        {
            "food_name_original": "فراخ مشوية",
            "food_name_en": "grilled chicken breast",
            "quantity": 200.0,
            "unit": "gram",
        }
    ]
    nutrition_ar = [
        {
            "success": True,
            "original_query": "فراخ مشوية",
            "food_name": "Chicken, breast, grilled",
            "calories_per_100g": 165,
            "protein_per_100g": 31,
            "carbs_per_100g": 0,
            "fat_per_100g": 3.6,
            "portions": [],
        }
    ]
    r = calculate_meal_total(foods_ar, nutrition_ar)
    print(
        f"  success={r['success']}  calories={r['total_calories']}  unmatched={r['unmatched_foods']}"
    )

    # 4) P0-3 — household fallback units
    print("\n[4] P0-3 Household fallback for 7 units")
    for u in ["slice", "tablespoon", "teaspoon", "bowl", "can", "bottle", "loaf"]:
        conv = convert_to_grams(1.0, u, [])
        status = "OK" if conv and conv["source"] == "household_fallback" else "FAIL"
        print(f"  [{status}] {u} -> {conv}")

    # 5) P0-4 — validate_meal_nutrition works with incomplete profile
    print(
        "\n[5] P0-4 validate_meal_nutrition with partial profile (no mandatory fields)"
    )
    partial = {"forbidden_foods": ["eggs"], "allergies": [], "dietary_restrictions": []}
    empty_meal = {
        "success": True,
        "foods": [],
        "total_calories": 400,
        "total_protein_g": 20,
        "total_carbs_g": 40,
        "total_fat_g": 10,
    }
    res = validate_meal_nutrition(empty_meal, partial, target_calories=400)
    print(f"  valid={res['valid']}  issues={res['issues']}")

    # 6) P1-6 — scale_meal_to_target
    print("\n[6] P1-6 scale_meal_to_target")
    foods_scale = [
        {
            "food_name_original": "فراخ مشوية",
            "food_name_en": "grilled chicken breast",
            "quantity": 200.0,
            "unit": "gram",
        }
    ]
    scaled = scale_meal_to_target(
        foods_scale, current_calories=460, target_calories=400
    )
    print(f"  scale_factor={scaled['scale_factor']}  achievable={scaled['achievable']}")
    print(f"  new quantity: {scaled['scaled_foods'][0]['quantity']}g")

    # 7) P1-7 — min_protein_g
    print("\n[7] P1-7 validate_meal_nutrition with min_protein_g")
    low_protein_meal = {
        "success": True,
        "foods": [],
        "total_calories": 400,
        "total_protein_g": 5,
        "total_carbs_g": 70,
        "total_fat_g": 8,
    }
    res7 = validate_meal_nutrition(
        low_protein_meal, {}, target_calories=400, min_protein_g=30
    )
    print(f"  valid={res7['valid']}  issues={res7['issues']}")

    # 8) P1-8 — tolerance constants visible
    print("\n[8] P1-8 Tolerance constants")
    print(f"  MEAL_CALORIE_TOLERANCE  = {MEAL_CALORIE_TOLERANCE}")
    print(f"  DAILY_CALORIE_TOLERANCE = {DAILY_CALORIE_TOLERANCE}")
    print(f"  DAILY_MACRO_TOLERANCE   = {DAILY_MACRO_TOLERANCE}")

    print("\nDone.")
