"""Exercise meal editing through HTTP with real calculation and validation."""

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.ai.agent import food_lookup, meal_edit_llm
from app.ai.agent.flows import meal_edits
from app.ai.agent.meal_edit_prompts import build_meal_selection_messages
from app.ai.contracts.errors import ProcessingError
from app.ai.contracts.meal_edits import MealChanges, MealSelection
from app.ai.conversation import session as session_module
from app.ai.conversation.meal_edits import apply_food_changes
from app.main import create_app


def food(name="rice", quantity=200.0):
    return {
        "food_name_original": name,
        "food_name_en": name,
        "quantity": quantity,
        "unit": "gram",
    }


def saved_meal(kind="lunch"):
    return {
        "meal_id": str(uuid4()),
        "meal_type": kind,
        "target_calories": 260.0,
        "min_protein_g": None,
        "foods": [food()],
        "nutrition": {
            "total_calories": 9999.0,
            "total_protein_g": 0.0,
            "total_carbs_g": 0.0,
            "total_fat_g": 0.0,
            "foods": [],
            "unmatched_foods": [],
        },
        "valid": True,
        "violations": [],
        "partial_failures": [],
    }


def changes(items):
    return MealChanges.model_validate({"changes": items})


@pytest.fixture
def editing(monkeypatch):
    saved = saved_meal()
    controls = {
        "selection": MealSelection(meal_id=saved["meal_id"]),
        "changes": changes([{"food_index": 0, "food": food("pasta")}]),
        "target": None,
        "meal_type": None,
        "fail_lookup": False,
    }
    lookups = []
    selections = []
    original_calculation = meal_edits.calculate_meal_nutrition

    def parse(message, history):
        return {
            "intent": "meal_edit",
            "language_detected": "english",
            "raw_text": message,
            "foods": [],
            "profile_data": None,
            "meal_type": controls["meal_type"],
            "target_calories": controls["target"],
        }

    def select(message, history, meals, language):
        selections.append(deepcopy(meals))
        return controls["selection"]

    def lookup(name, original, **kwargs):
        lookups.append(name)
        if controls["fail_lookup"]:
            return {"success": False}
        return {
            "success": True,
            "data": {
                "success": True,
                "original_query": name,
                "food_name": name,
                "calories_per_100g": 130.0,
                "protein_per_100g": 25.0,
                "carbs_per_100g": 5.0,
                "fat_per_100g": 1.0,
                "portions": [],
            },
        }

    monkeypatch.setattr(session_module, "parse_with_context", parse)
    monkeypatch.setattr(meal_edits, "call_meal_selection", select)
    monkeypatch.setattr(
        meal_edits, "call_meal_changes", lambda *args: controls["changes"]
    )
    monkeypatch.setattr(food_lookup, "get_nutrition", lookup)
    monkeypatch.setattr(
        meal_edits,
        "calculate_meal_nutrition",
        lambda *args, **kwargs: (
            {"success": False, "data": None}
            if controls.get("calculation_error")
            else original_calculation(*args, **kwargs)
        ),
    )
    with TestClient(create_app()) as client:
        yield client, saved, controls, lookups, selections


def test_valid_edit_recalculates_and_preserves_the_other_meals(editing):
    client, saved, _, lookups, _ = editing
    other = saved_meal("breakfast")
    context = [other, saved]
    before = deepcopy(context)
    response = client.post(
        "/v1/chat",
        json={"message": "Replace lunch rice with pasta", "meal_context": context},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "meal_edit"
    assert body["status"] == "ready"
    assert body["meals"][0] == other
    edited = body["meals"][1]
    assert edited["meal_id"] == saved["meal_id"]
    assert edited["foods"] == [food("pasta")]
    assert edited["nutrition"]["total_calories"] == 260.0
    assert edited["valid"] is True
    assert lookups == ["pasta"]
    assert context == before


def test_explicit_target_replaces_the_saved_target(editing):
    client, saved, controls, _, _ = editing
    controls["target"] = 390.0
    controls["changes"] = changes([{"food_index": 0, "food": food("rice", 300.0)}])
    body = client.post(
        "/v1/chat",
        json={"message": "Make that lunch 390 calories", "meal_context": [saved]},
    ).json()
    assert body["meals"][0]["target_calories"] == 390.0
    assert body["meals"][0]["nutrition"]["total_calories"] == 390.0


@pytest.mark.parametrize("reason", ["calories", "protein", "restriction"])
def test_failed_checks_do_not_publish_a_replacement(editing, reason):
    client, saved, controls, _, _ = editing
    profile = {}
    if reason == "calories":
        controls["changes"] = changes([{"food_index": 0, "food": food("pasta", 100.0)}])
    elif reason == "protein":
        saved["min_protein_g"] = 100.0
    else:
        profile = {"forbidden_foods": ["pasta"]}
    body = client.post(
        "/v1/chat",
        json={
            "message": "Replace the rice",
            "meal_context": [saved],
            "profile": profile,
        },
    ).json()
    assert body["status"] == "meal_edit_rejected"
    assert body["meals"] == []
    assert "previous meal is unchanged" in body["response"]
    if reason == "calories":
        assert "130.0 kcal" in body["response"]


@pytest.mark.parametrize(
    "reason",
    [
        "ambiguous",
        "unknown_id",
        "selection_error",
        "changes_error",
        "empty_changes",
        "lookup_error",
        "missing",
        "wrong_meal",
        "bad_index",
        "bad_unit",
        "delete_all",
        "calculation_error",
    ],
)
def test_unresolved_or_failed_edits_leave_meals_unchanged(editing, reason):
    client, saved, controls, lookups, selections = editing
    context = [saved]
    if reason == "ambiguous":
        controls["selection"] = MealSelection(meal_id=None)
    elif reason == "unknown_id":
        controls["selection"] = MealSelection(meal_id=str(uuid4()))
    elif reason == "selection_error":
        controls["selection"] = None
    elif reason == "changes_error":
        controls["changes"] = None
    elif reason == "empty_changes":
        controls["changes"] = changes([])
    elif reason == "lookup_error":
        controls["fail_lookup"] = True
    elif reason == "missing":
        context = []
    elif reason == "bad_index":
        controls["changes"] = changes([{"food_index": 10, "food": food()}])
    elif reason == "bad_unit":
        controls["changes"] = changes(
            [{"food_index": 0, "food": {**food(), "unit": "bucket"}}]
        )
    elif reason == "delete_all":
        controls["changes"] = changes([{"food_index": 0, "food": None}])
    elif reason == "calculation_error":
        controls["calculation_error"] = True
    else:
        controls["meal_type"] = "dinner"
    response = client.post(
        "/v1/chat", json={"message": "Change it", "meal_context": context}
    )

    if reason in {"selection_error", "changes_error"}:
        assert response.status_code == 502
        assert response.json()["error_code"] == "llm_invalid_output"
        assert lookups == []
        return

    assert response.status_code == 200
    assert response.json()["meals"] == []
    assert response.json()["status"] != "ready"
    if reason not in ("lookup_error", "calculation_error"):
        assert lookups == []
    if reason in ("missing", "wrong_meal"):
        assert selections == []


def test_changes_use_original_indices_and_preserve_untouched_foods():
    original = [food("rice"), food("chicken"), food("broccoli")]
    output = apply_food_changes(
        original,
        changes(
            [
                {"food_index": 0, "food": None},
                {"food_index": 1, "food": food("chicken", 150.0)},
                {"food_index": None, "food": food("pasta", 100.0)},
            ]
        ),
    )
    assert output == [food("chicken", 150.0), food("broccoli"), food("pasta", 100.0)]
    output[1]["quantity"] = 999
    assert original[2]["quantity"] == 200.0


@pytest.mark.parametrize(
    "items",
    [
        [{"food_index": 3, "food": None}],
        [{"food_index": 0, "food": food()}, {"food_index": 0, "food": None}],
        [{"food_index": None, "food": None}],
        [{"food_index": 0, "food": None}],
    ],
)
def test_invalid_structural_changes_are_rejected(items):
    with pytest.raises(ValueError):
        apply_food_changes([food()], changes(items))


def test_provider_outputs_are_validated_and_failures_are_not_ambiguity(monkeypatch):
    outputs = iter(
        [
            '{"meal_id":null}',
            '{"changes":[]}',
            '{"changes":[{"food_index":0,"food":{"food_name_original":"rice","food_name_en":"rice","quantity":-1,"unit":"gram"}}]}',
            "not json",
        ]
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))]
        )

    monkeypatch.setattr(
        meal_edit_llm,
        "get_client",
        lambda role: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    messages = [
        {"role": "system", "content": "Return JSON"},
        {"role": "user", "content": "Change it"},
    ]
    assert meal_edit_llm.call_edit_output(messages, MealSelection).meal_id is None
    assert meal_edit_llm.call_edit_output(messages, MealChanges).changes == []

    for schema in (MealChanges, MealSelection):
        with pytest.raises(ProcessingError) as caught:
            meal_edit_llm.call_edit_output(messages, schema)

        assert caught.value.code == "llm_invalid_output"

    assert calls[0]["response_format"] == {"type": "json_object"}
    assert messages[0]["content"] == "Return JSON"


def test_selection_prompt_uses_ids_and_recent_history():
    saved = saved_meal()
    history = [{"role": "user", "content": str(i)} for i in range(8)]
    messages = build_meal_selection_messages("Change it", history, [saved], "english")
    assert saved["meal_id"] in messages[0]["content"]
    assert messages[1:-1] == history[-6:]
    assert messages[-1] == {"role": "user", "content": "Change it"}
