"""Verify meal output against final calculations and request-local state."""

from copy import deepcopy
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.ai.agent import food_lookup, orchestrator
from app.ai.agent.flows import chat, daily, meals
from app.ai.conversation import session as session_module
from app.ai.conversation.meals import build_meal_snapshot
from app.ai.nutrition.calculations import calculate_meal_total
from app.ai.service import NutritionAssistant
from app.main import create_app

FOOD = {
    "food_name_original": "rice",
    "food_name_en": "rice",
    "quantity": 100.0,
    "unit": "gram",
}
NUTRITION = {
    "success": True,
    "original_query": "rice",
    "food_name": "Cooked rice",
    "calories_per_100g": 130.0,
    "protein_per_100g": 2.5,
    "carbs_per_100g": 28.0,
    "fat_per_100g": 0.3,
    "portions": [],
}
PROFILE = {
    "birth_date": "1990-01-01",
    "gender": "male",
    "weight": 80,
    "height": 180,
    "activity_level": "moderate",
    "goal": "maintenance",
}


def meal_flow(target=260.0):
    food = {**FOOD, "quantity": target / 1.3}
    return {
        "success": True,
        "foods": [food],
        "meal_result": calculate_meal_total([food], [NUTRITION]),
        "valid": True,
        "report": {"valid": True, "violations": []},
        "partial_failures": [],
    }


def parser(message, history):
    return {
        "intent": "meal_recommendation" if message == "meal" else "general_chat",
        "language_detected": "english",
        "raw_text": message,
        "foods": [],
        "profile_data": None,
        "meal_type": "lunch",
        "target_calories": 260.0,
    }


def test_http_snapshot_uses_scaled_final_meal_and_does_not_leak(monkeypatch):
    monkeypatch.setattr(session_module, "parse_with_context", parser)
    monkeypatch.setattr(meals, "call_meal_generation", lambda *args: [deepcopy(FOOD)])
    monkeypatch.setattr(
        food_lookup,
        "get_nutrition",
        lambda *args, **kwargs: {"success": True, "data": deepcopy(NUTRITION)},
    )
    monkeypatch.setattr(chat, "call_general_chat", lambda *args: "Hello")
    with TestClient(create_app()) as client:
        first = client.post("/v1/chat", json={"message": "meal"})
        second = client.post("/v1/chat", json={"message": "hello"})
    assert first.status_code == second.status_code == 200
    snapshot = first.json()["meals"][0]
    UUID(snapshot["meal_id"])
    assert snapshot["foods"][0]["quantity"] == 200.0
    assert snapshot["nutrition"]["total_calories"] == 260.0
    assert snapshot["nutrition"]["foods"][0]["quantity_g"] == 200.0
    assert snapshot["nutrition"]["foods"][0]["conversion_source"] == "direct"
    assert snapshot["valid"] is True
    assert second.json()["meals"] == []


def test_snapshot_preserves_failed_checks_and_is_independent():
    flow = meal_flow()
    flow["valid"] = False
    flow["report"] = {"valid": False, "violations": ["Protein below target"]}
    snapshot = build_meal_snapshot(flow, 260.0, "lunch", 30.0)
    assert snapshot["valid"] is False
    assert snapshot["min_protein_g"] == 30.0
    flow["foods"][0]["quantity"] = 999
    flow["meal_result"]["foods"][0]["quantity_g"] = 999
    flow["report"]["violations"].clear()
    assert snapshot["foods"][0]["quantity"] == 200.0
    assert snapshot["nutrition"]["foods"][0]["quantity_g"] == 200.0
    assert snapshot["violations"] == ["Protein below target"]


def test_partial_data_cannot_be_marked_valid():
    flow = meal_flow()
    flow["partial_failures"] = ["Unknown food"]
    snapshot = build_meal_snapshot(flow, 260.0, "lunch", None)
    assert snapshot["valid"] is False
    assert snapshot["partial_failures"] == ["Unknown food"]


def test_failed_generation_has_no_snapshot():
    assert build_meal_snapshot({"success": False}, 260.0, "lunch", None) is None


def test_daily_output_keeps_successful_meals_with_distinct_ids(monkeypatch):
    def parse_day(message, history):
        return {**parser(message, history), "intent": "daily_plan"}

    def generate(target, profile, meal_type, language, minimum, state, intent):
        return {"success": False} if meal_type == "dinner" else meal_flow(target)

    monkeypatch.setattr(session_module, "parse_with_context", parse_day)
    monkeypatch.setattr(daily, "meal_recommendation_flow", generate)
    with TestClient(create_app()) as client:
        response = client.post("/v1/chat", json={"message": "day", "profile": PROFILE})
    assert response.status_code == 200
    snapshots = response.json()["meals"]
    assert [meal["meal_type"] for meal in snapshots] == ["breakfast", "lunch", "snack"]
    assert len({meal["meal_id"] for meal in snapshots}) == 3
    assert "Could not generate this meal" in response.json()["response"]


def test_exception_discards_unfinished_meal_output(monkeypatch):
    def fail(message, session, state):
        state.meals.append(build_meal_snapshot(meal_flow(), 260.0, "lunch", None))
        raise RuntimeError("Formatter failed")

    monkeypatch.setattr(orchestrator, "handle", fail)
    assistant = NutritionAssistant()

    with pytest.raises(RuntimeError, match="Formatter failed"):
        assistant.respond("meal")

    assert assistant.agent_state.meals == []
