from unittest.mock import MagicMock, patch

import pytest

from app.ai.agent import orchestrator
from app.ai.agent.flows import meals
from app.ai.agent.state import AgentState
from app.ai.nutrition.profile import (
    calculate_macros,
    calculate_protein_target,
)


def complete_profile(weight=70.0, goal="weight_loss"):
    return {
        "birth_date": "2000-01-01",
        "gender": "male",
        "weight": weight,
        "height": 175.0,
        "activity_level": "moderate",
        "goal": goal,
    }


@pytest.mark.parametrize(
    "goal,expected",
    [
        ("weight_loss", 140.0),
        ("muscle_gain", 140.0),
        ("weight_gain", 112.0),
        ("maintenance", 112.0),
    ],
)
def test_daily_and_partial_profile_protein_follow_the_same_policy(goal, expected):
    assert calculate_protein_target({"weight": 70.0, "goal": goal}) == pytest.approx(
        expected
    )
    assert calculate_macros(complete_profile(goal=goal))["protein_g"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize("goal", ["weight_loss", "muscle_gain"])
def test_explicit_meal_works_with_partial_profile_and_current_protein_policy(goal):
    with patch.object(
        meals,
        "meal_recommendation_flow",
        return_value={"success": False, "error": "test boundary"},
    ) as flow:
        meals.handle_meal_recommendation(
            400.0,
            {"weight": 80.0, "goal": goal},
            "breakfast",
            "english",
            AgentState(),
        )

    assert flow.call_args.args[0] == 400.0
    assert flow.call_args.args[4] == 40.0


def test_explicit_meal_does_not_use_preexisting_targets():
    state = AgentState(targets={"macros": {"protein_g": 160.0}})
    with patch.object(
        meals,
        "meal_recommendation_flow",
        return_value={"success": False, "error": "test boundary"},
    ) as flow:
        meals.handle_meal_recommendation(
            400.0,
            complete_profile(weight=60.0),
            "breakfast",
            "english",
            state,
        )

    assert flow.call_args.args[0] == 400.0
    assert flow.call_args.args[4] == 30.0


def test_profile_update_clears_old_targets_before_the_next_meal():
    old_profile = complete_profile(weight=90.0)
    new_profile = complete_profile(weight=60.0)

    def parsed_turn(intent, profile, target=None):
        return {
            "success": True,
            "data": {
                "status": "onboarding_complete"
                if intent == "update_profile"
                else "ready",
                "profile": profile,
                "prompt_for_user": "Profile updated.",
                "parsed": {
                    "intent": intent,
                    "language_detected": "english",
                    "meal_type": "breakfast",
                    "target_calories": target,
                },
            },
        }

    state = AgentState()
    session = MagicMock()
    with (
        patch.object(
            orchestrator,
            "parse_query",
            side_effect=[
                parsed_turn("meal_recommendation", old_profile),
                parsed_turn("update_profile", new_profile),
                parsed_turn("meal_recommendation", new_profile, 400.0),
            ],
        ),
        patch.object(
            meals,
            "meal_recommendation_flow",
            return_value={"success": False, "error": "test boundary"},
        ) as flow,
    ):
        orchestrator.handle("Suggest breakfast", session, state)
        assert state.targets["macros"]["protein_g"] == 180.0

        orchestrator.handle("My weight is now 60kg", session, state)
        assert state.targets is None
        assert state.user_profile["weight"] == 60.0

        orchestrator.handle("Make a 400-calorie breakfast", session, state)

    assert flow.call_args_list[0].args[4] == 45.0
    assert flow.call_args_list[1].args[0] == 400.0
    assert flow.call_args_list[1].args[4] == 30.0
    assert state.targets is None


def test_explicit_meal_preserves_optional_protein_check_for_maintenance():
    with patch.object(
        meals,
        "meal_recommendation_flow",
        return_value={"success": False, "error": "test boundary"},
    ) as flow:
        meals.handle_meal_recommendation(
            400.0,
            {"weight": 70.0, "goal": "maintenance"},
            "breakfast",
            "english",
            AgentState(),
        )

    assert flow.call_args.args[4] is None


def test_missing_calorie_target_and_incomplete_profile_asks_for_calories():
    with patch.object(meals, "meal_recommendation_flow") as flow:
        answer = meals.handle_meal_recommendation(
            None,
            {},
            "breakfast",
            "english",
            AgentState(),
        )

    assert answer == "How many calories would you like this meal to be?"
    flow.assert_not_called()


def test_meal_report_includes_the_protein_failure_used_by_the_revision_loop():
    foods = [{"food_name_en": "rice", "quantity": 100.0, "unit": "gram"}]
    nutrition = [{"success": True}]
    meal_result = {
        "success": True,
        "total_calories": 400.0,
        "total_protein_g": 10.0,
        "total_carbs_g": 75.0,
        "total_fat_g": 6.7,
        "foods": [],
        "unmatched_foods": [],
    }

    with (
        patch.object(meals, "call_meal_generation", return_value=foods),
        patch.object(
            meals,
            "collect_nutrition_for_foods",
            return_value=(foods, nutrition, []),
        ),
        patch.object(
            meals,
            "run_revision_loop",
            return_value=(meal_result, foods, nutrition, False),
        ) as revision,
    ):
        result = meals.meal_recommendation_flow(
            400.0,
            {},
            "breakfast",
            "english",
            30.0,
            AgentState(),
            "meal_recommendation",
        )

    assert revision.call_args.args[4] == 30.0
    assert result["report"]["valid"] is False
    assert any("protein" in issue.lower() for issue in result["report"]["violations"])
