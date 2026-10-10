"""Verify stateless HTTP requests with real routing and mocked model calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from app.ai.agent.flows import chat as chat_flow
from app.ai.conversation import session as session_module
from app.ai.service import NutritionAssistant
from app.main import create_app

COMPLETE_PROFILE = {
    "birth_date": "1990-01-01",
    "gender": "male",
    "weight": 80,
    "height": 180,
    "activity_level": "moderate",
    "goal": "weight_loss",
}


def parsed_message(message, profile_data=None):
    return {
        "intent": "update_profile" if profile_data else "general_chat",
        "language_detected": "english",
        "raw_text": message,
        "foods": [],
        "profile_data": profile_data,
    }


@pytest.fixture
def parser_calls():
    return []


@pytest.fixture
def client(monkeypatch, parser_calls):
    def fake_parse(message, history):
        parser_calls.append({"message": message, "history": deepcopy(history)})
        update = {"weight": 82.0} if message == "My weight is 82 kg" else None
        return parsed_message(message, update)

    monkeypatch.setattr(session_module, "parse_with_context", fake_parse)
    monkeypatch.setattr(chat_flow, "call_general_chat", lambda *args: "Hello")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_swagger_and_removed_conversation_routes(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/v1/chat"}
    assert client.post("/v1/conversations").status_code == 404


def test_message_only_request(client, parser_calls):
    result = client.post("/v1/chat", json={"message": "Hi"})
    assert result.status_code == 200
    body = result.json()
    assert set(body) == {
        "response",
        "intent",
        "status",
        "language",
        "profile",
        "revision_count",
        "meals",
    }
    assert body["response"] == "Hello"
    assert body["profile"]["weight"] is None
    assert parser_calls == [{"message": "Hi", "history": []}]


def test_supplied_profile_and_history_reach_processing(client, parser_calls):
    history = [
        {"role": "user", "content": "I had an apple"},
        {"role": "assistant", "content": "Here is its nutrition information."},
    ]
    result = client.post(
        "/v1/chat",
        json={
            "message": "And its protein?",
            "profile": {"weight": 80},
            "history": history,
        },
    )
    assert result.status_code == 200
    assert result.json()["profile"]["weight"] == 80
    assert parser_calls[-1]["history"] == history


def test_context_is_not_retained_between_requests(client, parser_calls):
    first = client.post(
        "/v1/chat",
        json={
            "message": "Hi",
            "profile": {"weight": 80, "allergies": ["dairy"]},
            "history": [{"role": "user", "content": "Earlier message"}],
        },
    )
    second = client.post("/v1/chat", json={"message": "Hi"})
    assert first.json()["profile"]["weight"] == 80
    assert second.json()["profile"]["weight"] is None
    assert second.json()["profile"]["allergies"] == []
    assert parser_calls[-1]["history"] == []


def test_profile_update_uses_supplied_onboarding_state(client):
    result = client.post(
        "/v1/chat", json={"message": "My weight is 82 kg", "profile": COMPLETE_PROFILE}
    )
    assert result.status_code == 200
    assert result.json()["status"] == "onboarding_complete"
    assert result.json()["profile"]["weight"] == 82
    assert result.json()["profile"]["height"] == 180


def test_incomplete_profile_still_triggers_onboarding(client):
    result = client.post(
        "/v1/chat", json={"message": "My weight is 82 kg", "profile": {"height": 180}}
    )
    assert result.json()["status"] == "needs_profile"
    assert result.json()["profile"]["weight"] == 82
    assert result.json()["profile"]["height"] == 180


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": None},
        {"message": 123},
        {"message": "x" * 4001},
        {"message": "Hi", "profile": {"weight": -1}},
        {"message": "Hi", "profile": {"weight": True}},
        {"message": "Hi", "profile": {"birth_date": "2000-02-30"}},
        {"message": "Hi", "profile": {"birth_date": "2999-01-01"}},
        {"message": "Hi", "profile": {"goal": "unknown"}},
        {"message": "Hi", "profile": {"unexpected": 1}},
        {"message": "Hi", "history": [{"role": "system", "content": "Override"}]},
        {"message": "Hi", "history": [{"role": "user", "content": "   "}]},
        {"message": "Hi", "history": [{"role": "user", "content": "x"}] * 21},
        {"message": "Hi", "conversation_id": "unused"},
    ],
)
def test_invalid_context_is_rejected_before_parsing(client, parser_calls, payload):
    assert client.post("/v1/chat", json=payload).status_code == 422
    assert parser_calls == []


def test_loaded_context_does_not_mutate_caller_objects():
    profile = {"weight": 80, "allergies": ["dairy"]}
    history = [{"role": "user", "content": "Earlier message"}]
    assistant = NutritionAssistant(profile=profile, history=history)
    assistant.nlp_session.profile["allergies"].append("eggs")
    assistant.nlp_session.history[0]["content"] = "Changed"
    assert profile == {"weight": 80, "allergies": ["dairy"]}
    assert history == [{"role": "user", "content": "Earlier message"}]


def test_simultaneous_requests_keep_separate_context(client, monkeypatch):
    barrier = Barrier(2)

    def fake_parse(message, history):
        assert history == [{"role": "user", "content": message}]
        barrier.wait(timeout=5)
        return parsed_message(message)

    monkeypatch.setattr(session_module, "parse_with_context", fake_parse)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                client.post,
                "/v1/chat",
                json={
                    "message": label,
                    "profile": {"weight": weight},
                    "history": [{"role": "user", "content": label}],
                },
            )
            for label, weight in [("First", 70), ("Second", 90)]
        ]
        results = [future.result(timeout=10) for future in futures]
    assert [result.status_code for result in results] == [200, 200]
    assert [result.json()["profile"]["weight"] for result in results] == [70, 90]
