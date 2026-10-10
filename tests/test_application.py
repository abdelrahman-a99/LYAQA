import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.ai.agent import orchestrator
from app.ai.service import NutritionAssistant
from app.paths import (
    DEFAULT_RAG_DIR,
    EVALUATION_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
)


def test_application_imports_from_repo_root_without_keys():
    env = {
        key: value
        for key, value in os.environ.items()
        if "API_KEY" not in key.upper() and key != "PYTHONPATH"
    }

    code = """
import sys
from app.ai.service import NutritionAssistant

assistant = NutritionAssistant()
assert assistant.agent_state.user_profile == {}

for name in ("groq", "torch", "requests", "fastapi"):
    assert name not in sys.modules, name
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_conversations_are_independent_and_reset_is_local():
    first = NutritionAssistant()
    second = NutritionAssistant()

    first.agent_state.user_profile["weight"] = 80
    second.agent_state.user_profile["weight"] = 70

    first.nlp_session.history.append({"role": "user", "content": "Hi"})

    assert second.nlp_session.history == []

    first.reset()

    assert first.agent_state.user_profile == {}
    assert first.nlp_session.history == []
    assert second.agent_state.user_profile["weight"] == 70


def test_response_is_a_snapshot_of_the_processed_turn(monkeypatch):
    assistant = NutritionAssistant()

    def fake_handle(message, session, state):
        assert message == "Hello"
        assert session is assistant.nlp_session
        assert state is assistant.agent_state

        state.intent = "update_profile"
        state.status = "ready"
        state.user_profile = {"allergies": ["dairy"]}
        session.language = "arabic"

        return "Done"

    monkeypatch.setattr(orchestrator, "handle", fake_handle)

    result = assistant.respond("Hello")
    assistant.agent_state.user_profile["allergies"].append("eggs")

    assert result.response == "Done"
    assert result.intent == "update_profile"
    assert result.language == "arabic"
    assert result.profile == {"allergies": ["dairy"]}
    assert result.to_dict()["status"] == "ready"


def test_response_preserves_the_previous_backend_error_fallback(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Mock failure")

    monkeypatch.setattr(orchestrator, "handle", fail)

    result = NutritionAssistant().respond("Hello")

    assert result.response == (
        "Sorry, something went wrong while processing your message. Please try again."
    )


@pytest.mark.parametrize("message", ["", "   ", None])
def test_invalid_messages_do_not_run_the_agent(message, monkeypatch):
    def unexpected_call(**kwargs):
        pytest.fail("Agent must not run for an invalid message")

    monkeypatch.setattr(orchestrator, "handle", unexpected_call)

    with pytest.raises(ValueError):
        NutritionAssistant().respond(message)


def test_required_service_resources_are_available():
    assert (DEFAULT_RAG_DIR / "faiss_index.bin").read_bytes()
    assert (DEFAULT_RAG_DIR / "chunks.json").read_text(encoding="utf-8")
    assert (KNOWLEDGE_BASE_DIR / "nutrition_basics.md").is_file()
    assert (EVALUATION_DATA_DIR / "dataset.json").is_file()
    assert (EVALUATION_DATA_DIR / "predictions_sample.json").is_file()


def test_session_info_is_an_independent_snapshot():
    assistant = NutritionAssistant()
    assistant.agent_state.status = "ready"
    assistant.agent_state.intent = "update_profile"
    assistant.agent_state.user_profile = {"allergies": ["dairy"]}
    assistant.nlp_session.language = "arabic"
    assistant.nlp_session.history.extend(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )
    assistant.nlp_session.onboarding_complete = True

    snapshot = assistant.get_session_info()

    assert snapshot == {
        "status": "ready",
        "intent": "update_profile",
        "language": "arabic",
        "profile": {"allergies": ["dairy"]},
        "history_length": 2,
        "onboarding_complete": True,
    }

    assistant.agent_state.user_profile["allergies"].append("eggs")
    assert snapshot["profile"]["allergies"] == ["dairy"]

    snapshot["profile"]["allergies"].append("nuts")
    assert assistant.get_session_info()["profile"]["allergies"] == ["dairy", "eggs"]


def test_reset_clears_session_metadata():
    assistant = NutritionAssistant()
    initial = assistant.get_session_info()
    assistant.agent_state.user_profile = {"weight": 80}
    assistant.agent_state.status = "ready"
    assistant.nlp_session.history.append({"role": "user", "content": "Hello"})
    assistant.nlp_session.onboarding_complete = True

    assistant.reset()

    assert assistant.get_session_info() == initial
