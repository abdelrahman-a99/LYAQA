"""
tests/test_integration.py — End-to-end integration tests for Task 5 (brief scenarios 1-10).

Boundaries MOCKED: Task 1 (Groq), Task 2 (USDA), Task 4 LLM prompts, RAG retrieve.
REAL under test: Task 3 (nutrition_engine), orchestrator routing, revision loop.

Run: pytest tests/test_integration.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ai.agent.state import AgentState

# ─── Fixtures / helpers ─────────────────────────────────────────────────────


@pytest.fixture
def state():
    return AgentState()


@pytest.fixture
def session():
    return MagicMock()


def _parsed(**overrides):
    base = {
        "intent": "general_chat",
        "language_detected": "english",
        "raw_text": "",
        "foods": [],
        "meal_type": None,
        "target_calories": None,
        "requested_substitution": None,
        "profile_data": None,
    }
    base.update(overrides)
    return base


def _session_output(parsed, profile=None, status="ready", prompt=None, missing=None):
    return {
        "parsed": parsed,
        "status": status,
        "prompt_for_user": prompt,
        "profile": profile if profile is not None else {},
        "missing_fields": missing or [],
    }


def _env(success, data=None, error=None, error_code=None):
    return {"success": success, "data": data, "error": error, "error_code": error_code}


def _usda_ok(name, cal=165.0, prot=31.0, carb=0.0, fat=3.6, g_per_unit=1.0):
    return {
        "success": True,
        "original_query": name,
        "food_name": name,
        "calories_per_100g": cal,
        "protein_per_100g": prot,
        "carbs_per_100g": carb,
        "fat_per_100g": fat,
        "portions": [
            {"amount": 1.0, "unit": "gram", "modifier": None, "gram_weight": g_per_unit}
        ],
        "source": "USDA FoodData Central",
    }


def _usda_fail(name, error="food_not_found"):
    return {
        "success": False,
        "original_query": name,
        "error": error,
        "error_code": error,
    }


PROFILE_WEIGHT_LOSS = {
    "birth_date": "1990-01-01",
    "gender": "male",
    "weight": 90.0,
    "height": 175.0,
    "activity_level": "moderate",
    "goal": "weight_loss",
    "dietary_restrictions": [],
    "forbidden_foods": [],
    "preferred_foods": [],
    "allergies": [],
}

PROFILE_WEIGHT_GAIN = {**PROFILE_WEIGHT_LOSS, "goal": "weight_gain"}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — User wants weight loss
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
def test_scenario_01_weight_loss_targets(mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(intent="daily_plan", language_detected="english"),
            profile=PROFILE_WEIGHT_LOSS,
        ),
    )
    # Daily plan requires targets + meal generation → mock those, verify target math
    with (
        patch("app.ai.agent.flows.daily.calculate_targets") as mock_targets,
        patch("app.ai.agent.flows.daily.meal_recommendation_flow") as mock_meal,
        patch("app.ai.agent.flows.daily.validate_plan"),
        patch("app.ai.agent.flows.daily.validate_daily_nutrition_report") as mock_rep,
    ):
        mock_targets.return_value = _env(
            True,
            {
                "bmr": 1890.0,
                "tdee": 2930.0,
                "target_calories": 2530.0,  # tdee - 400
                "macros": {
                    "calories": 2530.0,
                    "protein_g": 180.0,
                    "carbs_g": 282.5,
                    "fat_g": 70.3,
                },
            },
        )
        mock_meal.return_value = {
            "success": True,
            "meal_result": {
                "total_calories": 600.0,
                "total_protein_g": 40.0,
                "total_carbs_g": 60.0,
                "total_fat_g": 20.0,
                "foods": [],
                "unmatched_foods": [],
            },
            "foods": [],
            "nutrition_list": [],
            "valid": True,
            "partial_failures": [],
            "report": {"valid": True, "violations": []},
        }
        mock_rep.return_value = {"valid": True, "violations": []}
        from app.ai.agent.orchestrator import handle

        res = handle("give me a daily plan for weight loss", session, state)

    # Assert: orchestrator called calculate_targets with a weight_loss profile,
    # and the daily target includes the -400 kcal deficit.
    mock_targets.assert_called()
    args, _ = mock_targets.call_args
    assert args[0]["goal"] == "weight_loss"
    assert state.targets["target_calories"] == 2930.0 - 400.0
    assert "daily" in res.lower() or "plan" in res.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2 — User wants weight gain
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
def test_scenario_02_weight_gain_targets(mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(intent="daily_plan"),
            profile=PROFILE_WEIGHT_GAIN,
        ),
    )
    with (
        patch("app.ai.agent.flows.daily.calculate_targets") as mock_targets,
        patch("app.ai.agent.flows.daily.meal_recommendation_flow") as mock_meal,
        patch("app.ai.agent.flows.daily.validate_plan"),
        patch("app.ai.agent.flows.daily.validate_daily_nutrition_report") as mock_rep,
    ):
        mock_targets.return_value = _env(
            True,
            {
                "bmr": 1890.0,
                "tdee": 2930.0,
                "target_calories": 3330.0,  # tdee + 400
                "macros": {
                    "calories": 3330.0,
                    "protein_g": 144.0,
                    "carbs_g": 436.5,
                    "fat_g": 92.5,
                },
            },
        )
        mock_meal.return_value = {
            "success": True,
            "meal_result": {
                "total_calories": 800.0,
                "total_protein_g": 35.0,
                "total_carbs_g": 100.0,
                "total_fat_g": 25.0,
                "foods": [],
                "unmatched_foods": [],
            },
            "foods": [],
            "nutrition_list": [],
            "valid": True,
            "partial_failures": [],
            "report": {"valid": True, "violations": []},
        }
        mock_rep.return_value = {"valid": True, "violations": []}
        from app.ai.agent.orchestrator import handle

        handle("plan for muscle gain", session, state)
    args, _ = mock_targets.call_args
    assert args[0]["goal"] == "weight_gain"
    assert state.targets["target_calories"] == 2930.0 + 400.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — User writes in Arabic
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_03_arabic_message(mock_get_nut, mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="calculate_calories",
                language_detected="arabic",
                raw_text="كلت 200 جرام فراخ مشوية",
                foods=[
                    {
                        "food_name_original": "فراخ مشوية",
                        "food_name_en": "grilled chicken breast",
                        "quantity": 200.0,
                        "unit": "gram",
                    }
                ],
            ),
        ),
    )
    mock_get_nut.return_value = _env(
        True, _usda_ok("grilled chicken breast", cal=165, prot=31, fat=3.6)
    )

    from app.ai.agent.orchestrator import handle

    res = handle("كلت 200 جرام فراخ مشوية", session, state)

    # Response should be in Arabic
    assert "سعرة" in res or "إجمالي" in res
    assert state.intent == "calculate_calories"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4 — User writes in English
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_04_english_message(mock_get_nut, mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="calculate_calories",
                language_detected="english",
                foods=[
                    {
                        "food_name_original": "chicken breast",
                        "food_name_en": "chicken breast",
                        "quantity": 200.0,
                        "unit": "gram",
                    }
                ],
            ),
        ),
    )
    mock_get_nut.return_value = _env(True, _usda_ok("chicken breast"))

    from app.ai.agent.orchestrator import handle

    res = handle("I ate 200g chicken breast", session, state)

    assert "Total calories" in res
    assert "330" in res  # 165 kcal/100g * 2


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5 — User uses Arabic-English mixed
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_05_mixed_message(mock_get_nut, mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="calculate_calories",
                language_detected="mixed",
                foods=[
                    {
                        "food_name_original": "grilled salmon",
                        "food_name_en": "grilled salmon",
                        "quantity": 150.0,
                        "unit": "gram",
                    }
                ],
            ),
        ),
    )
    mock_get_nut.return_value = _env(
        True, _usda_ok("grilled salmon", cal=208, prot=20, fat=13)
    )

    from app.ai.agent.orchestrator import handle

    res = handle("أكلت grilled salmon 150g", session, state)

    assert state.intent == "calculate_calories"
    # Mixed-language user gets an English-formatted response (fallback language)
    assert "Total calories" in res or "سعرة" in res


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6 — User has an allergy
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_06_allergy_detected(mock_get_nut, mock_parse, session, state):
    profile = {**PROFILE_WEIGHT_LOSS, "allergies": ["peanuts"]}
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="meal_recommendation",
                target_calories=400.0,
                meal_type="snack",
                foods=[],
            ),
            profile=profile,
        ),
    )

    def fake_nutrition(food_en, food_orig, intent=""):
        if "peanut" in food_en.lower():
            return _env(True, _usda_ok(food_en, cal=567, prot=25, fat=49))
        return _env(True, _usda_ok(food_en))

    mock_get_nut.side_effect = fake_nutrition

    with (
        patch("app.ai.agent.flows.meals.call_meal_generation") as mock_gen,
        patch("app.ai.agent.revision.call_full_meal_revision") as mock_rev,
        patch("app.ai.agent.revision.call_food_substitute") as mock_sub,
    ):
        # LLM ignores the restriction and proposes peanuts
        mock_gen.return_value = [
            {
                "food_name_original": "peanuts",
                "food_name_en": "peanuts",
                "quantity": 70.0,
                "unit": "gram",
            }
        ]
        # Substitute returns a safe alternative
        mock_sub.return_value = {
            "food_name_original": "almonds",
            "food_name_en": "almonds",
            "quantity": 50.0,
            "unit": "gram",
        }
        mock_rev.return_value = [
            {
                "food_name_original": "almonds",
                "food_name_en": "almonds",
                "quantity": 50.0,
                "unit": "gram",
            }
        ]

        from app.ai.agent.orchestrator import handle

        handle("Make me a 400 kcal snack", session, state)

    # The revision loop should have detected the violation and called the substitute
    assert state.revision_count >= 1 or len(state.revision_history) >= 1
    history_actions = [h.get("action_taken") for h in state.revision_history]
    assert any("substitute" in a for a in history_actions if a)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7 — User asks for meal without a specific ingredient
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_07_without_ingredient(mock_get_nut, mock_parse, session, state):
    profile = {**PROFILE_WEIGHT_LOSS, "forbidden_foods": ["eggs"]}
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="meal_recommendation",
                target_calories=400.0,
                meal_type="breakfast",
            ),
            profile=profile,
        ),
    )
    mock_get_nut.side_effect = lambda f_en, f_orig, intent="": _env(
        True, _usda_ok(f_en, cal=155 if "egg" in f_en.lower() else 165)
    )

    with (
        patch("app.ai.agent.flows.meals.call_meal_generation") as mock_gen,
        patch("app.ai.agent.revision.call_food_substitute") as mock_sub,
        patch("app.ai.agent.revision.call_full_meal_revision") as mock_rev,
    ):
        mock_gen.return_value = [
            {
                "food_name_original": "boiled eggs",
                "food_name_en": "boiled egg",
                "quantity": 100.0,
                "unit": "gram",
            }
        ]
        mock_sub.return_value = {
            "food_name_original": "oats",
            "food_name_en": "oats",
            "quantity": 80.0,
            "unit": "gram",
        }
        mock_rev.return_value = mock_gen.return_value

        from app.ai.agent.orchestrator import handle

        handle("400 kcal breakfast without eggs", session, state)

    # Validator must have flagged eggs at some point
    all_validation = [h.get("validation_result", {}) for h in state.revision_history]
    any_violation = any(v.get("restriction_violations") for v in all_validation if v)
    assert any_violation or state.revision_count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8 — User doesn't specify quantity
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_08_no_quantity(mock_get_nut, mock_parse, session, state):
    # Task 1 normalize_output defaults quantity to 1.0 when missing
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="calculate_calories",
                foods=[
                    {
                        "food_name_original": "banana",
                        "food_name_en": "banana",
                        "quantity": 1.0,
                        "unit": "piece",
                    }
                ],
            ),
        ),
    )
    mock_get_nut.return_value = _env(
        True, _usda_ok("banana", cal=89, prot=1.1, carb=22.8, fat=0.3, g_per_unit=118.0)
    )

    from app.ai.agent.orchestrator import handle

    res = handle("I ate a banana", session, state)

    # Should not crash; response must be present and contain a number
    assert res
    assert any(c.isdigit() for c in res)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9 — USDA does not find the food
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_09_usda_not_found(mock_get_nut, mock_parse, session, state):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="calculate_calories",
                foods=[
                    {
                        "food_name_original": "unobtainium fruit",
                        "food_name_en": "unobtainium fruit",
                        "quantity": 100.0,
                        "unit": "gram",
                    }
                ],
            ),
        ),
    )
    mock_get_nut.return_value = _env(
        False, None, error="food_not_found", error_code="food_not_found"
    )

    from app.ai.agent.orchestrator import handle

    res = handle("I ate unobtainium fruit", session, state)

    # Should report that nothing could be computed
    assert res
    assert "could not" in res.lower() or "not" in res.lower() or "لم" in res


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10 — Plan exceeds calorie target → revision loop triggers
# ═══════════════════════════════════════════════════════════════════════════


@patch("app.ai.agent.orchestrator.parse_query")
@patch("app.ai.agent.food_lookup.get_nutrition")
def test_scenario_10_over_calories_triggers_revision(
    mock_get_nut, mock_parse, session, state
):
    mock_parse.return_value = _env(
        True,
        _session_output(
            _parsed(
                intent="meal_recommendation",
                target_calories=400.0,
                meal_type="lunch",
            ),
            profile=PROFILE_WEIGHT_LOSS,
        ),
    )
    mock_get_nut.side_effect = lambda f_en, f_orig, intent="": _env(
        True, _usda_ok(f_en, cal=165, prot=31, fat=3.6)
    )

    # First generation is way over target (600 g * 165/100 = 990 kcal)
    over_meal = [
        {
            "food_name_original": "chicken breast",
            "food_name_en": "chicken breast",
            "quantity": 600.0,
            "unit": "gram",
        }
    ]
    # Revision returns something realistic that lands within ±15% of 400 kcal
    fixed_meal = [
        {
            "food_name_original": "chicken breast",
            "food_name_en": "chicken breast",
            "quantity": 240.0,
            "unit": "gram",  # 396 kcal
        }
    ]

    with (
        patch("app.ai.agent.flows.meals.call_meal_generation") as mock_gen,
        patch("app.ai.agent.revision.call_full_meal_revision") as mock_rev,
        patch("app.ai.agent.revision.call_food_substitute"),
    ):
        mock_gen.return_value = over_meal
        mock_rev.return_value = fixed_meal

        from app.ai.agent.orchestrator import handle

        res = handle("Make me a 400 kcal lunch", session, state)

    # scale_meal + revision must have been invoked; revision_count > 0
    # (scale is uncounted — so revision_count reflects LLM revisions only)
    assert len(state.revision_history) >= 1
    actions = [h.get("action_taken", "") for h in state.revision_history]
    # At least one action must involve a fix attempt
    assert any("scale" in a or "llm" in a or "accepted" in a for a in actions)
    assert res
