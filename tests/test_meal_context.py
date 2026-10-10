"""Verify caller-supplied meal context reaches routing without leaking."""

from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.ai.agent import orchestrator
from app.ai.agent.flows import chat
from app.ai.conversation import session as session_module
from app.ai.service import NutritionAssistant
from app.main import create_app


def meal():
    return {
        "meal_id": str(uuid4()),
        "meal_type": "lunch",
        "target_calories": 600.0,
        "min_protein_g": None,
        "foods": [
            {
                "food_name_original": "rice",
                "food_name_en": "cooked rice",
                "quantity": 200.0,
                "unit": "gram",
            }
        ],
        "nutrition": {
            "total_calories": 260.0,
            "total_protein_g": 5.0,
            "total_carbs_g": 56.0,
            "total_fat_g": 0.6,
            "foods": [],
            "unmatched_foods": [],
        },
        "valid": False,
        "violations": ["Calories below target"],
        "partial_failures": [],
    }


@pytest.fixture
def api(monkeypatch):
    contexts = []
    parse_calls = []
    original_handle = orchestrator.handle

    def parse(message, history):
        parse_calls.append(message)
        return {
            "intent": "general_chat",
            "language_detected": "english",
            "raw_text": message,
            "foods": [],
            "profile_data": None,
        }

    def handle(message, session, state):
        result = original_handle(message, session, state)
        contexts.append(deepcopy(state.meal_context))
        return result

    monkeypatch.setattr(session_module, "parse_with_context", parse)
    monkeypatch.setattr(orchestrator, "handle", handle)
    monkeypatch.setattr(chat, "call_general_chat", lambda *args: "Hello")
    with TestClient(create_app()) as client:
        yield client, contexts, parse_calls


def test_context_reaches_routing_but_is_not_echoed_or_retained(api):
    client, contexts, _ = api
    saved = meal()
    first = client.post("/v1/chat", json={"message": "Hi", "meal_context": [saved]})
    second = client.post("/v1/chat", json={"message": "Hi"})
    assert first.status_code == second.status_code == 200
    assert contexts == [[saved], []]
    assert first.json()["meals"] == second.json()["meals"] == []
    assert "meal_context" not in first.json()


def test_four_distinct_meals_are_accepted(api):
    client, contexts, _ = api
    saved = [meal() for _ in range(4)]
    response = client.post("/v1/chat", json={"message": "Hi", "meal_context": saved})
    assert response.status_code == 200
    assert contexts == [saved]


def test_context_is_copied_and_survives_turn_reset():
    supplied = [meal()]
    assistant = NutritionAssistant(meal_context=supplied)
    assistant.agent_state.meals = [meal()]
    assistant.agent_state.reset_turn()
    assert assistant.agent_state.meals == []
    assert assistant.agent_state.meal_context == supplied
    assistant.agent_state.meal_context[0]["foods"][0]["quantity"] = 999
    assert supplied[0]["foods"][0]["quantity"] == 200.0
    assert NutritionAssistant().agent_state.meal_context == []


@pytest.mark.parametrize(
    "invalid", ["duplicate", "count", "quantity", "total", "id", "null"]
)
def test_invalid_context_is_rejected_before_parsing(api, invalid):
    client, contexts, parse_calls = api
    saved = meal()
    payload = [saved]
    if invalid == "duplicate":
        payload = [saved, deepcopy(saved)]
    elif invalid == "count":
        payload = [meal() for _ in range(5)]
    elif invalid == "quantity":
        saved["foods"][0]["quantity"] = -1
    elif invalid == "total":
        saved["nutrition"]["total_calories"] = -1
    elif invalid == "id":
        saved["meal_id"] = "unknown"
    else:
        payload = None
    response = client.post("/v1/chat", json={"message": "Hi", "meal_context": payload})
    assert response.status_code == 422
    assert contexts == parse_calls == []
