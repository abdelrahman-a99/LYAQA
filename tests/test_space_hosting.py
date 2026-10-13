"""Preserve REST validation and controlled errors around the Gradio GPU transport."""

from concurrent.futures import TimeoutError
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from gradio_client.exceptions import AppError

from app import hosting
from app.ai import runtime
from app.ai.agent.tools import retrieval as tools
from app.ai.contracts.errors import ProcessingError


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "cpu")
    runtime.get_runtime.cache_clear()
    runtime.get_device.cache_clear()
    cached_client = hosting.get_gpu_client
    cached_client.cache_clear()
    yield
    runtime.get_runtime.cache_clear()
    runtime.get_device.cache_clear()
    cached_client.cache_clear()


def job_returning(monkeypatch, value=None, error=None):
    job = Mock()
    if error:
        job.result.side_effect = error
    else:
        job.result.return_value = value
    client = Mock()
    client.submit.return_value = job
    monkeypatch.setattr(hosting, "get_gpu_client", lambda: client)
    return client, job


def test_real_gradio_server_preserves_rest_routes_and_validation():
    application = hosting.create_space_app()
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok"}
        schema = client.get("/openapi.json").json()
        assert "/v1/chat" in schema["paths"]
        assert (
            "meal_context"
            in schema["components"]["schemas"]["ChatRequest"]["properties"]
        )
        response = client.post("/v1/chat", json={"message": " "})
    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_request"
    assert response.json()["retryable"] is False


def test_queued_operation_uses_named_engine_endpoint(monkeypatch):
    client, job = job_returning(monkeypatch, {"ok": True, "data": [0.8]})
    result = hosting.execute_queued(
        "semantic", {"text": "rice", "descriptions": ["Rice, cooked"]}
    )
    assert result == [0.8]
    assert client.submit.call_args.kwargs["api_name"] == "/model_compute"
    job.result.assert_called_once_with(timeout=120)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"ok": "yes"},
        {"ok": False, "error": "private"},
        {"ok": False, "error": {"error_code": "unknown"}},
        {"ok": True, "data": "bad"},
        {"ok": True, "data": [float("nan")]},
        {"ok": True, "data": [True]},
        {"ok": True, "data": []},
    ],
)
def test_malformed_gpu_result_fails_explicitly(monkeypatch, body):
    job_returning(monkeypatch, body)
    with pytest.raises(ProcessingError) as caught:
        hosting.execute_queued("semantic", {"text": "rice", "descriptions": ["Rice"]})
    assert caught.value.code == "zerogpu_invalid_output"


def test_quota_error_does_not_expose_upstream_diagnostics(monkeypatch):
    job_returning(
        monkeypatch,
        {
            "ok": False,
            "error": {
                "error_code": "zerogpu_quota_exceeded",
                "detail": "private upstream diagnostics",
            },
        },
    )
    with pytest.raises(ProcessingError) as caught:
        hosting.execute_queued("translate", {"text": "أرز"})
    assert caught.value.code == "zerogpu_quota_exceeded"
    assert caught.value.status_code == 503
    assert caught.value.retryable is False
    assert "private" not in str(caught.value)


def test_gradio_quota_exception_has_controlled_contract(monkeypatch):
    job_returning(
        monkeypatch, error=AppError("You have exceeded your free ZeroGPU quota")
    )
    with pytest.raises(ProcessingError) as caught:
        hosting.execute_queued("translate", {"text": "أرز"})
    assert caught.value.code == "zerogpu_quota_exceeded"


def test_timeout_cancels_job_without_replaying(monkeypatch):
    client, job = job_returning(monkeypatch, error=TimeoutError())
    with pytest.raises(ProcessingError) as caught:
        hosting.execute_queued("translate", {"text": "أرز"})
    assert caught.value.code == "zerogpu_timeout"
    client.submit.assert_called_once()
    job.cancel.assert_called_once()


def test_known_gpu_failure_reaches_rest_handler(monkeypatch):
    application = hosting.create_space_app()
    from app.api import routes

    assistant = Mock()
    assistant.respond.side_effect = hosting.gpu_error("zerogpu_quota_exceeded")
    monkeypatch.setattr(routes, "NutritionAssistant", Mock(return_value=assistant))
    with TestClient(application) as client:
        response = client.post("/v1/chat", json={"message": "Give me a meal"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "zerogpu_quota_exceeded"
    assert response.json()["retryable"] is False


def test_rag_tool_does_not_hide_quota_errors(monkeypatch):
    from app.ai.rag import retrieve

    monkeypatch.setattr(
        retrieve,
        "retrieve_fitness_knowledge",
        Mock(side_effect=hosting.gpu_error("zerogpu_quota_exceeded")),
    )
    with pytest.raises(ProcessingError) as caught:
        tools.retrieve_fitness_knowledge("Question")
    assert caught.value.code == "zerogpu_quota_exceeded"


def test_private_hf_token_is_only_used_for_client_auth(monkeypatch):
    from gradio_client import Client

    monkeypatch.setenv("SPACE_ID", "team/nutrition")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    constructor = Mock(spec=Client)
    monkeypatch.setattr("gradio_client.Client", constructor)
    hosting.get_gpu_client()
    constructor.assert_called_once_with(
        "team/nutrition", token="test-token", verbose=False
    )
