"""
tests/test_orchestrator.py — Tests for the Orchestrator intent handling and flows.
"""

from unittest.mock import MagicMock, patch

from app.ai.agent import orchestrator
from app.ai.agent.state import AgentState


def test_needs_profile_status_gate():
    # When status is "needs_profile", orchestrator must return the onboarding prompt immediately
    session = MagicMock()
    app_state = AgentState()
    with patch("app.ai.agent.orchestrator.parse_query") as mock_parse:
        mock_parse.return_value = {
            "success": True,
            "data": {
                "parsed": {"intent": "general_chat", "language_detected": "english"},
                "status": "needs_profile",
                "prompt_for_user": "Please tell me your age, weight, height...",
                "profile": {},
                "missing_fields": ["birth_date", "weight"],
            },
        }
        res = orchestrator.handle("Hello", session=session, state=app_state)
        assert res == "Please tell me your age, weight, height..."
        assert app_state.status == "needs_profile"


def test_calculate_calories_flow():
    session = MagicMock()
    app_state = AgentState()
    with (
        patch("app.ai.agent.orchestrator.parse_query") as mock_parse,
        patch(
            "app.ai.agent.flows.calories.collect_nutrition_for_foods"
        ) as mock_collect,
        patch("app.ai.agent.flows.calories.calculate_meal_nutrition") as mock_calc,
    ):
        mock_parse.return_value = {
            "success": True,
            "data": {
                "parsed": {
                    "intent": "calculate_calories",
                    "foods": [
                        {"food_name_en": "apple", "quantity": 1, "unit": "piece"}
                    ],
                    "language_detected": "english",
                },
                "status": "ready",
                "prompt_for_user": None,
                "profile": {"weight": 70},
                "missing_fields": [],
            },
        }
        mock_collect.return_value = (
            [{"food_name_en": "apple", "quantity": 1, "unit": "piece"}],
            [{"food_name": "apple", "calories_per_100g": 52}],
            [],
        )
        mock_calc.return_value = {
            "success": True,
            "data": {
                "total_calories": 95.0,
                "total_protein_g": 0.5,
                "total_carbs_g": 25.0,
                "total_fat_g": 0.3,
                "foods": [
                    {
                        "food_name_en": "apple",
                        "quantity": 1,
                        "unit": "piece",
                        "calories": 95.0,
                        "protein_g": 0.5,
                        "carbs_g": 25.0,
                        "fat_g": 0.3,
                        "conversion_source": "household_fallback",
                    }
                ],
            },
        }

        res = orchestrator.handle("I ate an apple", session=session, state=app_state)
        assert "95.0" in res or "95" in res
        assert "calories" in res.lower()


def test_daily_plan_flow():
    session = MagicMock()
    app_state = AgentState()
    profile = {
        "birth_date": "1990-01-01",
        "gender": "male",
        "weight": 75.0,
        "height": 178.0,
        "activity_level": "moderate",
        "goal": "maintenance",
    }

    mock_meal_result = {
        "success": True,
        "total_calories": 500.0,
        "total_protein_g": 30.0,
        "total_carbs_g": 60.0,
        "total_fat_g": 15.0,
        "foods": [
            {
                "food_name_en": "oats",
                "food_name_original": "oats",
                "quantity": 100.0,
                "unit": "g",
                "calories": 500.0,
                "protein_g": 30.0,
                "carbs_g": 60.0,
                "fat_g": 15.0,
            }
        ],
        "unmatched_foods": [],
    }

    dummy_flow_result = {
        "success": True,
        "meal_result": mock_meal_result,
        "foods": mock_meal_result["foods"],
        "nutrition_list": [{"success": True, "food_name": "oats"}],
        "valid": True,
        "partial_failures": [],
        "report": {"valid": True, "violations": []},
    }

    with (
        patch("app.ai.agent.orchestrator.parse_query") as mock_parse,
        patch("app.ai.agent.flows.daily.calculate_targets") as mock_targets,
        patch("app.ai.agent.flows.daily.meal_recommendation_flow") as mock_meal_flow,
        patch("app.ai.agent.flows.daily.validate_plan") as mock_val_plan,
        patch(
            "app.ai.agent.flows.daily.validate_daily_nutrition_report"
        ) as mock_val_report,
    ):
        mock_parse.return_value = {
            "success": True,
            "data": {
                "parsed": {
                    "intent": "daily_plan",
                    "language_detected": "english",
                },
                "status": "ready",
                "prompt_for_user": None,
                "profile": profile,
                "missing_fields": [],
            },
        }

        mock_targets.return_value = {
            "success": True,
            "data": {
                "bmr": 1687.5,
                "tdee": 2615.6,
                "target_calories": 2000.0,
                "macros": {"protein_g": 120.0, "carbs_g": 250.0, "fat_g": 55.0},
            },
        }

        mock_meal_flow.return_value = dummy_flow_result

        mock_val_plan.return_value = {
            "success": True,
            "data": {"valid": True, "issues": [], "checks": {}},
        }

        mock_val_report.return_value = {"valid": True, "violations": []}

        res = orchestrator.handle(
            "give me a daily meal plan", session=session, state=app_state
        )

        # Ensure validate_plan received a list of meal_result dicts
        assert mock_val_plan.called
        args, kwargs = mock_val_plan.call_args
        meals_arg = args[0]
        assert isinstance(meals_arg, list)
        assert len(meals_arg) == 4
        assert kwargs.get("mode") == "day"

        # Check response content
        assert "suggested daily plan" in res.lower()
        assert "daily total" in res.lower()
