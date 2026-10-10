from copy import deepcopy

import pytest

from app.ai.nutrition.calculations import calculate_meal_total
from app.ai.presentation.responses import (
    format_calorie_macro_response,
    format_daily_plan_response,
    format_meal_recommendation_response,
    format_quantity_approximation_note,
)


def build_meal(unit="cup", quantity=2.0, portions=None):
    food = {
        "food_name_original": "رز",
        "food_name_en": "rice",
        "quantity": quantity,
        "unit": unit,
    }
    nutrition = {
        "success": True,
        "original_query": "rice",
        "food_name": "Rice",
        "calories_per_100g": 130.0,
        "protein_per_100g": 2.7,
        "carbs_per_100g": 28.0,
        "fat_per_100g": 0.3,
        "portions": portions or [],
    }
    return calculate_meal_total([food], [nutrition])


@pytest.mark.parametrize(
    "language,name",
    [("english", "rice"), ("arabic", "رز")],
)
@pytest.mark.parametrize(
    "unit,quantity,grams,reason_en,reason_ar",
    [
        ("cup", 2.0, 480.0, "household weight estimate", "الوحدة المنزلية"),
        ("milliliter", 100.0, 100.0, "density of 1 g/ml", "كثافة 1 جرام"),
    ],
)
def test_approximation_note_explains_the_actual_conversion(
    language,
    name,
    unit,
    quantity,
    grams,
    reason_en,
    reason_ar,
):
    meal = build_meal(unit, quantity)
    before = deepcopy(meal)

    note = format_quantity_approximation_note(meal, language)

    assert name in note
    assert f"{quantity:g} {unit} ≈ {grams:g}" in note
    assert (reason_ar if language == "arabic" else reason_en) in note
    assert meal == before


@pytest.mark.parametrize(
    "unit,portions,source",
    [
        ("gram", [], "direct"),
        (
            "cup",
            [
                {
                    "unit": "cup",
                    "amount": 1,
                    "gram_weight": 185,
                    "modifier": None,
                }
            ],
            "usda_portion",
        ),
    ],
)
def test_non_approximate_conversions_have_no_estimate_note(
    unit,
    portions,
    source,
):
    meal = build_meal(unit, portions=portions)

    assert meal["foods"][0]["conversion_source"] == source
    for language in ("english", "arabic"):
        assert format_quantity_approximation_note(meal, language) == ""


@pytest.mark.parametrize("language", ["english", "arabic"])
@pytest.mark.parametrize(
    "intent",
    ["calculate_calories", "calculate_macros", "food_nutrition"],
)
def test_calorie_responses_keep_both_partial_failure_and_estimate_note(
    language,
    intent,
):
    meal = build_meal()

    answer = format_calorie_macro_response(
        meal,
        intent,
        language,
        "partial failure",
    )

    assert "partial failure" in answer
    assert format_quantity_approximation_note(meal, language) in answer


@pytest.mark.parametrize("language", ["english", "arabic"])
def test_recommended_meal_shows_the_final_converted_quantity(language):
    meal = build_meal(quantity=1.5)
    flow = {
        "meal_result": meal,
        "foods": [],
        "report": {},
        "partial_failures": [],
    }

    answer = format_meal_recommendation_response(
        flow,
        468.0,
        language,
        True,
    )

    assert "1.5 cup ≈ 360" in answer
    assert format_quantity_approximation_note(meal, language) in answer


@pytest.mark.parametrize("language", ["english", "arabic"])
def test_daily_plan_marks_meal_estimates_and_the_daily_total(language):
    estimated = build_meal()
    exact = build_meal("gram", 100.0)
    flows = {
        "breakfast": {
            "success": True,
            "valid": True,
            "meal_result": estimated,
        },
        "lunch": {
            "success": True,
            "valid": True,
            "meal_result": exact,
        },
    }

    answer = format_daily_plan_response(
        flows,
        {},
        True,
        {},
        [],
        language,
        2000.0,
    )
    note = format_quantity_approximation_note(estimated, language)

    assert answer.count(note) == 1
    assert (
        "الإجمالي اليومي يتضمن كميات تقديرية."
        if language == "arabic"
        else "The daily total includes estimated quantities."
    ) in answer


def test_failed_meals_do_not_add_approximation_notes_to_the_daily_total():
    flows = {
        "breakfast": {
            "success": False,
            "meal_result": build_meal(),
        },
    }

    answer = format_daily_plan_response(
        flows,
        {},
        False,
        {},
        [],
        "english",
        2000.0,
    )

    assert "estimated quantities" not in answer
