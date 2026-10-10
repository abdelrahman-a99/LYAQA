"""Distinguish USDA outages from valid empty or incomplete food results."""

import json
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from app.ai.contracts.errors import ProcessingError
from app.ai.conversation import session
from app.ai.retrieval import config, service, usda_client
from app.ai.retrieval import http as usda_http
from app.ai.retrieval.cache import DETAILS_CACHE, SEARCH_CACHE, clear_caches
from app.main import create_app


@pytest.fixture(autouse=True)
def isolate_usda(monkeypatch):
    clear_caches()
    monkeypatch.setattr(usda_http, "get_api_key", lambda: "test-key")
    yield
    clear_caches()


def response(status, payload):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.invalid/usda?api_key=private-test-value"
    result._content = json.dumps(payload).encode()
    return result


def search_payload():
    return {"foods": [{"fdcId": 1, "description": "Rice", "dataType": "Foundation"}]}


def details_payload():
    return {
        "fdcId": 1,
        "description": "Rice",
        "foodNutrients": [
            {"nutrient": {"id": 1008}, "amount": 130},
            {"nutrient": {"id": 1003}, "amount": 2.7},
            {"nutrient": {"id": 1005}, "amount": 28},
            {"nutrient": {"id": 1004}, "amount": 0.3},
        ],
    }


def lookup(operation):
    if operation == "search":
        return usda_client.search_food_usda("rice")
    return usda_client.get_food_details(1)


@pytest.mark.parametrize("operation", ["search", "details"])
@pytest.mark.parametrize(
    "upstream,status,code,retryable",
    [
        (401, 502, "usda_authentication_failed", False),
        (403, 502, "usda_authentication_failed", False),
        (429, 503, "usda_rate_limited", True),
        (500, 503, "usda_unavailable", True),
        (503, 503, "usda_unavailable", True),
        (400, 502, "usda_request_rejected", False),
    ],
)
def test_http_failure_is_not_a_missing_food(
    monkeypatch, operation, upstream, status, code, retryable
):
    monkeypatch.setattr(
        usda_http.requests, "get", Mock(return_value=response(upstream, {}))
    )
    with pytest.raises(ProcessingError) as caught:
        lookup(operation)
    error = caught.value
    assert (error.status_code, error.code, error.retryable) == (status, code, retryable)
    assert "private-test-value" not in str(error)
    assert SEARCH_CACHE == DETAILS_CACHE == {}


@pytest.mark.parametrize("operation", ["search", "details"])
@pytest.mark.parametrize(
    "failure,code,status",
    [
        (requests.exceptions.Timeout(), "usda_timeout", 504),
        (requests.exceptions.ConnectionError(), "usda_unavailable", 503),
    ],
)
def test_transport_failure_can_recover_without_cached_failure(
    monkeypatch, operation, failure, code, status
):
    payload = search_payload() if operation == "search" else details_payload()
    get = Mock(side_effect=[failure, response(200, payload)])
    monkeypatch.setattr(usda_http.requests, "get", get)
    with pytest.raises(ProcessingError) as caught:
        lookup(operation)
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert caught.value.retryable is True
    assert SEARCH_CACHE == DETAILS_CACHE == {}
    assert lookup(operation)
    assert lookup(operation)
    assert get.call_count == 2


@pytest.mark.parametrize("operation", ["search", "details"])
def test_invalid_json_is_not_cached(monkeypatch, operation):
    result = response(200, {})
    result._content = b"not-json"
    monkeypatch.setattr(usda_http.requests, "get", Mock(return_value=result))
    with pytest.raises(ProcessingError) as caught:
        lookup(operation)
    assert caught.value.code == "usda_invalid_response"
    assert SEARCH_CACHE == DETAILS_CACHE == {}


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("search", []),
        ("search", {}),
        ("search", {"foods": {}}),
        ("search", {"foods": [None]}),
        ("search", {"foods": [{"fdcId": 1}]}),
        ("details", {}),
        ("details", {"fdcId": 2, "description": "Wrong food"}),
        ("details", {**details_payload(), "foodNutrients": {}}),
        ("details", {**details_payload(), "foodNutrients": [None]}),
        (
            "details",
            {
                **details_payload(),
                "foodNutrients": [{"nutrient": {"id": 1008}, "amount": "130"}],
            },
        ),
    ],
)
def test_malformed_payload_is_not_a_missing_food(monkeypatch, operation, payload):
    monkeypatch.setattr(
        usda_http.requests, "get", Mock(return_value=response(200, payload))
    )
    with pytest.raises(ProcessingError) as caught:
        lookup(operation)
    assert caught.value.code == "usda_invalid_response"
    assert SEARCH_CACHE == DETAILS_CACHE == {}


def test_missing_credentials_fail_before_http(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)
    monkeypatch.setattr(usda_http, "get_api_key", config.get_api_key)
    get = Mock()
    monkeypatch.setattr(usda_http.requests, "get", get)
    with pytest.raises(ProcessingError) as caught:
        lookup("search")
    assert caught.value.code == "usda_configuration_error"
    assert caught.value.status_code == 500
    get.assert_not_called()


def test_valid_empty_search_is_still_a_domain_result(monkeypatch):
    get = Mock(return_value=response(200, {"foods": []}))
    monkeypatch.setattr(usda_http.requests, "get", get)
    assert lookup("search") == []
    assert lookup("search") == []
    get.assert_called_once()


def test_missing_detail_is_not_an_outage(monkeypatch):
    monkeypatch.setattr(usda_http.requests, "get", Mock(return_value=response(404, {})))
    assert lookup("details") is None
    assert DETAILS_CACHE == {}


def test_search_endpoint_404_is_a_configuration_or_request_failure(monkeypatch):
    monkeypatch.setattr(usda_http.requests, "get", Mock(return_value=response(404, {})))
    with pytest.raises(ProcessingError) as caught:
        lookup("search")
    assert caught.value.code == "usda_request_rejected"


def test_missing_nutrients_are_not_a_transport_failure(monkeypatch):
    payload = {"fdcId": 1, "description": "Rice", "foodNutrients": []}
    monkeypatch.setattr(
        usda_http.requests, "get", Mock(return_value=response(200, payload))
    )
    assert usda_client.nutrition_is_complete(lookup("details")) is False


def test_detail_outage_aborts_candidate_fallback(monkeypatch):
    candidates = [
        {
            "fdc_id": food_id,
            "description": "Rice",
            "data_type": "Foundation",
            "semantic_score": 0.9,
            "cross_encoder_score": 1.0,
            "final_score": 0.8,
        }
        for food_id in (1, 2)
    ]
    monkeypatch.setattr(service, "search_food_usda", lambda *a, **kw: candidates)
    monkeypatch.setattr(service, "semantic_retrieve", lambda *a, **kw: candidates)
    monkeypatch.setattr(service, "dynamic_rerank", lambda *a, **kw: candidates)
    get = Mock(side_effect=requests.exceptions.Timeout())
    monkeypatch.setattr(usda_http.requests, "get", get)
    with pytest.raises(ProcessingError) as caught:
        service.get_food_nutrition("rice")
    assert caught.value.code == "usda_timeout"
    get.assert_called_once()


@pytest.mark.parametrize("missing", ["record", "nutrients"])
def test_domain_missing_data_still_allows_candidate_fallback(monkeypatch, missing):
    candidates = [
        {
            "fdc_id": food_id,
            "description": "Rice",
            "data_type": "Foundation",
            "semantic_score": 0.9,
            "cross_encoder_score": 1.0,
            "final_score": 0.8,
        }
        for food_id in (1, 2)
    ]
    monkeypatch.setattr(service, "search_food_usda", lambda *a, **kw: candidates)
    monkeypatch.setattr(service, "semantic_retrieve", lambda *a, **kw: candidates)
    monkeypatch.setattr(service, "dynamic_rerank", lambda *a, **kw: candidates)
    first = (
        response(404, {})
        if missing == "record"
        else response(200, {"fdcId": 1, "description": "Rice", "foodNutrients": []})
    )
    second = response(200, {**details_payload(), "fdcId": 2})
    get = Mock(side_effect=[first, second])
    monkeypatch.setattr(usda_http.requests, "get", get)
    result = service.get_food_nutrition("rice")
    assert result["success"] is True
    assert result["fdc_id"] == 2
    assert result["selected_rank"] == 2
    assert get.call_count == 2


@pytest.mark.parametrize(
    "upstream,status,code",
    [(403, 502, "usda_authentication_failed"), (429, 503, "usda_rate_limited")],
)
def test_usda_failure_reaches_chat_http_handler(monkeypatch, upstream, status, code):
    monkeypatch.setattr(
        session,
        "parse_with_context",
        lambda message, history: {
            "intent": "calculate_calories",
            "language_detected": "english",
            "raw_text": message,
            "profile_data": None,
            "foods": [
                {
                    "food_name_original": "rice",
                    "food_name_en": "rice",
                    "quantity": 100,
                    "unit": "gram",
                }
            ],
        },
    )
    monkeypatch.setattr(
        usda_http.requests, "get", Mock(return_value=response(upstream, {}))
    )
    with TestClient(create_app()) as client:
        result = client.post("/v1/chat", json={"message": "Calories in 100 g rice"})
    assert result.status_code == status
    assert result.json()["error_code"] == code
    assert "private-test-value" not in result.text
