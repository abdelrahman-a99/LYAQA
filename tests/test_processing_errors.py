"""Verify provider failures, conversational outcomes, and safe RAG reloading."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai.agent import llm, meal_edit_llm
from app.ai.agent.flows import knowledge
from app.ai.contracts.conversation import ResolvedKnowledgeQuestion
from app.ai.contracts.errors import ProcessingError
from app.ai.contracts.meal_edits import MealSelection
from app.ai.conversation import session
from app.ai.nlp import parser
from app.ai.providers.groq import get_client
from app.ai.rag import retrieve
from app.main import create_app

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def test_missing_credentials_have_a_configuration_code(monkeypatch):
    for role in ("NLP", "MEALS", "RAG"):
        monkeypatch.delenv(f"GROQ_API_KEY_{role}", raising=False)

    with pytest.raises(ProcessingError) as caught:
        get_client.__wrapped__("nlp")

    assert caught.value.code == "llm_configuration_error"


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_chat_completion_is_not_a_greeting(monkeypatch, content):
    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    monkeypatch.setattr(llm, "get_client", lambda role: provider)

    with pytest.raises(ProcessingError) as caught:
        llm.call_general_chat("Hello", "english")

    assert caught.value.code == "llm_invalid_output"


def provider_error(kind):
    if kind == "timeout":
        return groq.APITimeoutError(request=REQUEST)

    if kind == "connection":
        return groq.APIConnectionError(request=REQUEST)

    status, error_type = {
        "auth": (401, groq.AuthenticationError),
        "permission": (403, groq.PermissionDeniedError),
        "rate": (429, groq.RateLimitError),
        "server": (500, groq.InternalServerError),
        "request": (400, groq.BadRequestError),
    }[kind]

    return error_type(
        "private-provider-diagnostic",
        response=httpx.Response(status, request=REQUEST),
        body=None,
    )


@pytest.mark.parametrize(
    "kind,status,code,retryable",
    [
        ("auth", 502, "llm_authentication_failed", False),
        ("permission", 502, "llm_authentication_failed", False),
        ("rate", 503, "llm_rate_limited", True),
        ("timeout", 504, "llm_timeout", True),
        ("connection", 503, "llm_unavailable", True),
        ("server", 503, "llm_unavailable", True),
        ("request", 502, "llm_request_rejected", False),
    ],
)
def test_nlp_provider_failure_is_not_a_successful_chat(
    monkeypatch, kind, status, code, retryable
):
    provider = Mock()
    provider.chat.completions.create.side_effect = provider_error(kind)
    monkeypatch.setattr(parser, "get_client", lambda role: provider)

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat", json={"message": "Hello"})

    assert response.status_code == status
    assert response.json()["error_code"] == code
    assert response.json()["retryable"] is retryable
    assert "private-provider-diagnostic" not in response.text
    assert provider.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    "operation",
    ["chat", "answer", "resolve", "meal", "substitute", "revise", "edit"],
)
def test_every_agent_model_boundary_propagates_authentication_failure(
    monkeypatch, operation
):
    provider = Mock()
    provider.chat.completions.create.side_effect = provider_error("auth")

    for module in (llm, meal_edit_llm):
        monkeypatch.setattr(module, "get_client", lambda role: provider)

    calls = {
        "chat": lambda: llm.call_general_chat("Hello", "english"),
        "answer": lambda: llm.call_rag_answer("Question", ["Evidence"], "english"),
        "resolve": lambda: llm.call_question_resolution("And after?", "english", []),
        "meal": lambda: llm.call_meal_generation(600, {}, "lunch", "english"),
        "substitute": lambda: llm.call_food_substitute("rice", {}, "english"),
        "revise": lambda: llm.call_full_meal_revision(
            [], 600, {}, "lunch", "english", []
        ),
        "edit": lambda: meal_edit_llm.call_edit_output(
            [{"role": "system", "content": "Return JSON"}],
            MealSelection,
        ),
    }

    with pytest.raises(ProcessingError) as caught:
        calls[operation]()

    assert caught.value.code == "llm_authentication_failed"


def test_malformed_nlp_output_retries_once_without_general_chat_fallback(monkeypatch):
    call = Mock(side_effect=ValueError("Invalid model output"))
    monkeypatch.setattr(parser, "_groq_call", call)

    with pytest.raises(ProcessingError) as caught:
        parser.parse_with_context("Hello", [])

    assert caught.value.code == "llm_invalid_output"
    assert call.call_count == 2


@pytest.mark.parametrize(
    "kind,status,expected",
    [
        ("ambiguous", 200, "needs_clarification"),
        ("low_confidence_retrieval", 200, "insufficient_knowledge"),
        ("index_not_found", 500, "rag_configuration_error"),
        ("rag_error", 503, "rag_unavailable"),
    ],
)
def test_rag_http_outcomes(monkeypatch, kind, status, expected):
    monkeypatch.setattr(
        session,
        "parse_with_context",
        lambda message, history: {
            "intent": "fitness_nutrition_question",
            "language_detected": "english",
            "raw_text": message,
            "foods": [],
            "profile_data": None,
        },
    )

    resolve = Mock(return_value=ResolvedKnowledgeQuestion(question=None))
    search = Mock(return_value={"success": False, "error_code": kind})

    monkeypatch.setattr(knowledge, "call_question_resolution", resolve)
    monkeypatch.setattr(knowledge, "retrieve_fitness_knowledge", search)

    payload = {"message": "And after?"}
    if kind == "ambiguous":
        payload["history"] = [{"role": "user", "content": "Earlier question"}]

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat", json=payload)

    assert response.status_code == status
    assert response.json()["status" if status == 200 else "error_code"] == expected

    if kind == "ambiguous":
        search.assert_not_called()


def test_failed_rag_model_load_does_not_publish_partial_resources(
    monkeypatch, tmp_path
):
    index_file = tmp_path / "faiss_index.bin"
    chunks_file = tmp_path / "chunks.json"

    index_file.write_bytes(b"index")
    chunks_file.write_text('[{"text":"Evidence","source_file":"doc.md"}]')

    monkeypatch.setattr(retrieve, "_INDEX_PATH", str(index_file))
    monkeypatch.setattr(retrieve, "_CHUNKS_PATH", str(chunks_file))

    for name in ("_index", "_chunks", "_embed_model"):
        monkeypatch.setattr(retrieve, name, None)

    model = object()
    load_model = Mock(side_effect=[RuntimeError("Download failed"), model])

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(read_index=lambda path: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=load_model),
    )

    with pytest.raises(RuntimeError, match="Download failed"):
        retrieve._ensure_loaded()

    assert retrieve._index is retrieve._chunks is retrieve._embed_model is None

    retrieve._ensure_loaded()

    assert retrieve._embed_model is model
    assert load_model.call_count == 2
