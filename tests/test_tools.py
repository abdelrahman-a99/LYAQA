"""
tests/test_tools.py — Unit tests for the 7 tools in agent/tools.py.
"""

from unittest.mock import MagicMock

from app.ai.agent import tools


def test_calculate_targets_success():
    profile = {
        "birth_date": "1990-01-01",
        "gender": "male",
        "weight": 75.0,
        "height": 178.0,
        "activity_level": "moderate",
        "goal": "maintenance",
    }
    res = tools.calculate_targets(profile)
    assert res["success"] is True
    assert res["error"] is None
    assert "bmr" in res["data"]
    assert "tdee" in res["data"]
    assert "target_calories" in res["data"]


def test_calculate_targets_invalid_profile():
    # missing required fields
    profile = {"gender": "male"}
    res = tools.calculate_targets(profile)
    assert res["success"] is False
    assert res["error_code"] == "incomplete_profile"
    assert res["data"] is None


def test_calculate_meal_nutrition_success():
    foods = [
        {
            "food_name_en": "egg",
            "food_name_original": "egg",
            "quantity": 2,
            "unit": "piece",
        },
        {
            "food_name_en": "toast",
            "food_name_original": "toast",
            "quantity": 1,
            "unit": "slice",
        },
    ]
    nutrition_list = [
        {
            "success": True,
            "original_query": "egg",
            "food_name": "egg",
            "calories_per_100g": 143,
            "protein_per_100g": 13,
            "carbs_per_100g": 1,
            "fat_per_100g": 10,
            "portions": [
                {"amount": 1.0, "unit": "piece", "modifier": "large", "gram_weight": 50}
            ],
        },
        {
            "success": True,
            "original_query": "toast",
            "food_name": "toast",
            "calories_per_100g": 260,
            "protein_per_100g": 9,
            "carbs_per_100g": 49,
            "fat_per_100g": 3,
            "portions": [{"modifier": "slice", "gram_weight": 30}],
        },
    ]
    res = tools.calculate_meal_nutrition(foods, nutrition_list)
    assert res["success"] is True
    assert "total_calories" in res["data"]
    assert len(res["data"]["foods"]) == 2


def test_validate_plan_valid():
    profile = {
        "birth_date": "1990-01-01",
        "gender": "male",
        "weight": 75.0,
        "height": 178.0,
        "activity_level": "moderate",
        "goal": "maintenance",
    }
    meal = {
        "success": True,
        "total_calories": 500.0,
        "total_protein_g": 30.0,
        "total_carbs_g": 50.0,
        "total_fat_g": 15.0,
        "foods": [],
    }
    res = tools.validate_plan(meal, profile, mode="meal", target_calories=500.0)
    assert res["success"] is True
    assert res["data"]["valid"] is True


def test_parse_query_mocked():
    session = MagicMock()
    session.process.return_value = {
        "parsed": {"intent": "log_food", "entities": {}},
        "status": "ready",
        "prompt_for_user": None,
        "profile": {},
        "missing_fields": [],
    }
    res = tools.parse_query("I ate 2 eggs", session=session)
    assert res["success"] is True
    assert res["data"]["parsed"]["intent"] == "log_food"
