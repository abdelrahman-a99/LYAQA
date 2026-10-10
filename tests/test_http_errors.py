"""Verify public errors without exposing internal exceptions or request values."""

from fastapi.testclient import TestClient

from app.ai.agent import orchestrator
from app.main import create_app


def test_unexpected_agent_error_is_http_500(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("private-provider-diagnostic")

    monkeypatch.setattr(orchestrator, "handle", fail)

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post("/v1/chat", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred while processing the request.",
        "error_code": "internal_error",
        "retryable": False,
    }
    assert "private-provider-diagnostic" not in response.text


def test_invalid_request_has_field_locations_without_input_values():
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "Hello", "history": "private-input"},
        )

    body = response.json()
    assert response.status_code == 422
    assert body["error_code"] == "invalid_request"
    assert body["retryable"] is False
    assert body["issues"][0]["loc"] == ["body", "history"]
    assert "private-input" not in response.text


def test_missing_route_and_swagger_use_public_error_contract():
    with TestClient(create_app()) as client:
        missing = client.get("/missing")
        schema = client.get("/openapi.json").json()

    assert missing.status_code == 404
    assert missing.json()["error_code"] == "not_found"

    errors = schema["paths"]["/v1/chat"]["post"]["responses"]
    for code in ("422", "500"):
        assert errors[code]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ErrorResponse"
        )
