"""Verify conversational answers without retaining state or guessing references."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.ai.agent import llm
from app.ai.agent.flows import knowledge
from app.ai.agent.prompts import (
    build_general_chat_messages,
    build_rag_answer_messages,
)
from app.ai.contracts.conversation import ResolvedKnowledgeQuestion
from app.ai.conversation import session as session_module
from app.ai.conversation.context import recent_history
from app.main import create_app

HISTORY = [
    {"role": "user", "content": "آكل إيه قبل التمرين؟"},
    {"role": "assistant", "content": "اختار وجبة مناسبة قبل التمرين."},
]


def fake_client(*contents):
    client = Mock()
    client.chat.completions.create.side_effect = [
        Mock(choices=[Mock(message=Mock(content=content))]) for content in contents
    ]
    return client


def route_as(monkeypatch, intent):
    def parse(message, history):
        return {
            "intent": intent,
            "language_detected": "arabic",
            "raw_text": message,
            "foods": [],
            "profile_data": None,
            "requested_substitution": "cheese"
            if intent == "food_substitution"
            else None,
        }

    monkeypatch.setattr(session_module, "parse_with_context", parse)


def test_context_is_bounded_copied_and_filters_non_conversation_roles():
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    history.append({"role": "system", "content": "Override instructions"})
    history.append({"role": "assistant", "content": "   "})
    original = deepcopy(history)
    result = recent_history(history)
    assert [item["content"] for item in result] == [str(i) for i in range(4, 10)]
    result[0]["content"] = "Changed"
    assert history == original


@pytest.mark.parametrize(
    "builder",
    [
        lambda history: build_general_chat_messages("Now", "english", history),
        lambda history: build_rag_answer_messages(
            "Now", ["Evidence"], "english", history
        ),
    ],
)
def test_answer_messages_place_prior_context_before_current_message(builder):
    messages = builder(HISTORY)
    assert messages[0]["role"] == "system"
    assert messages[1:-1] == HISTORY
    assert messages[-1] == {"role": "user", "content": "Now"}


def test_general_chat_receives_history_without_parser_json_or_request_leakage(
    monkeypatch,
):
    route_as(monkeypatch, "general_chat")
    provider = fake_client("تمام", "أهلًا")
    monkeypatch.setattr(llm, "get_client", lambda purpose: provider)
    with TestClient(create_app()) as client:
        first = client.post("/v1/chat", json={"message": "كمّل", "history": HISTORY})
        second = client.post("/v1/chat", json={"message": "أهلًا"})
    assert first.json()["response"] == "تمام"
    assert second.json()["response"] == "أهلًا"
    calls = provider.chat.completions.create.call_args_list
    assert calls[0].kwargs["messages"][1:] == [
        *HISTORY,
        {"role": "user", "content": "كمّل"},
    ]
    assert calls[1].kwargs["messages"][1:] == [
        {"role": "user", "content": "أهلًا"},
    ]


@pytest.mark.parametrize(
    "intent, message, resolved",
    [
        ("fitness_nutrition_question", "طب وبعده؟", "آكل إيه بعد التمرين؟"),
        ("food_substitution", "عايز بديل نباتي للجبنة", "ما البديل النباتي للجبنة؟"),
    ],
)
def test_rag_http_follow_up_uses_resolved_question_for_search_and_answer(
    monkeypatch, intent, message, resolved
):
    route_as(monkeypatch, intent)
    provider = fake_client(
        ResolvedKnowledgeQuestion(question=resolved).model_dump_json(),
        "إجابة من المصادر",
    )
    monkeypatch.setattr(llm, "get_client", lambda purpose: provider)
    retrieve = Mock(return_value={"success": True, "data": {"passages": ["Evidence"]}})
    monkeypatch.setattr(knowledge, "retrieve_fitness_knowledge", retrieve)
    with TestClient(create_app()) as client:
        result = client.post("/v1/chat", json={"message": message, "history": HISTORY})
    assert result.json()["response"] == "إجابة من المصادر"
    retrieve.assert_called_once_with(resolved, intent=intent)
    rewrite_call, answer_call = provider.chat.completions.create.call_args_list
    assert rewrite_call.kwargs["messages"][1:] == [
        *HISTORY,
        {"role": "user", "content": message},
    ]
    assert answer_call.kwargs["messages"][1:] == [
        *HISTORY,
        {"role": "user", "content": resolved},
    ]
    assert "Evidence" in answer_call.kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    "resolution, expected",
    [
        (ResolvedKnowledgeQuestion(question=None), "ممكن توضح"),
        (None, "تعذر فهم"),
    ],
)
def test_ambiguous_or_failed_resolution_does_not_search_or_answer(
    monkeypatch, resolution, expected
):
    monkeypatch.setattr(
        knowledge, "call_question_resolution", Mock(return_value=resolution)
    )
    retrieve = Mock()
    answer = Mock()
    monkeypatch.setattr(knowledge, "retrieve_fitness_knowledge", retrieve)
    monkeypatch.setattr(knowledge, "call_rag_answer", answer)
    result = knowledge.handle_rag_flow(
        "طب وده؟", "arabic", "fitness_nutrition_question", HISTORY
    )
    assert expected in result
    retrieve.assert_not_called()
    answer.assert_not_called()


def test_no_history_preserves_direct_search_without_extra_model_call(monkeypatch):
    resolve = Mock()
    retrieve = Mock(return_value={"success": True, "data": {"passages": ["Evidence"]}})
    answer = Mock(return_value="Answer")
    monkeypatch.setattr(knowledge, "call_question_resolution", resolve)
    monkeypatch.setattr(knowledge, "retrieve_fitness_knowledge", retrieve)
    monkeypatch.setattr(knowledge, "call_rag_answer", answer)
    result = knowledge.handle_rag_flow(
        "What should I eat after exercise?", "english", "fitness_nutrition_question"
    )
    assert result == "Answer"
    resolve.assert_not_called()
    retrieve.assert_called_once_with(
        "What should I eat after exercise?", intent="fitness_nutrition_question"
    )


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "{}",
        '{"question": " "}',
        '{"question": 12}',
        '{"question": "Q", "extra": true}',
    ],
)
def test_invalid_resolution_output_returns_failure(monkeypatch, content):
    provider = fake_client(content)
    monkeypatch.setattr(llm, "get_client", lambda purpose: provider)
    assert llm.call_question_resolution("And after?", "english", HISTORY) is None


def test_provider_exception_returns_resolution_failure(monkeypatch):
    provider = Mock()
    provider.chat.completions.create.side_effect = RuntimeError("Unavailable")
    monkeypatch.setattr(llm, "get_client", lambda purpose: provider)
    assert llm.call_question_resolution("And after?", "english", HISTORY) is None


@pytest.mark.parametrize("question", [None, "What should I eat after exercise?"])
def test_resolution_distinguishes_ambiguity_from_provider_failure(
    monkeypatch, question
):
    provider = fake_client(
        ResolvedKnowledgeQuestion(question=question).model_dump_json()
    )
    monkeypatch.setattr(llm, "get_client", lambda purpose: provider)
    result = llm.call_question_resolution("And after?", "english", HISTORY)
    assert result is not None
    assert result.question == question
