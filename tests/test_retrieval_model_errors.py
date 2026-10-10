"""Verify recoverable model initialization and concurrent first requests."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.ai.contracts.errors import ProcessingError
from app.ai.conversation import session
from app.ai.retrieval import models, service
from app.main import create_app


@pytest.fixture
def loaders(monkeypatch):
    bi_encoder = Mock(return_value=object())
    cross_encoder = Mock(return_value=object())
    tokenizer = Mock(return_value=object())
    translation = Mock()
    translation.to.return_value = object()
    translation_model = Mock(return_value=translation)
    monkeypatch.setattr(models, "_MODELS", {})
    monkeypatch.setattr(models, "get_device", lambda: "cpu")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=bi_encoder,
            CrossEncoder=cross_encoder,
            util=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            MarianTokenizer=SimpleNamespace(from_pretrained=tokenizer),
            MarianMTModel=SimpleNamespace(from_pretrained=translation_model),
        ),
    )
    return {
        "bi_encoder": bi_encoder,
        "cross_encoder": cross_encoder,
        "tokenizer": tokenizer,
        "translation_model": translation_model,
    }


def ensure(group):
    if group == "embedding":
        models._ensure_embedding_models()
    else:
        models._ensure_translation_model()


@pytest.mark.parametrize(
    "group,stage",
    [
        ("embedding", "bi_encoder"),
        ("embedding", "cross_encoder"),
        ("translation", "tokenizer"),
        ("translation", "translation_model"),
    ],
)
def test_failed_load_preserves_cache_and_next_call_can_recover(loaders, group, stage):
    unrelated = object()
    models._MODELS["other_resource"] = unrelated
    factory = loaders[stage]
    loaded = factory.return_value
    factory.side_effect = [OSError("private-download-diagnostic"), loaded]
    with pytest.raises(ProcessingError) as caught:
        ensure(group)
    assert caught.value.code == "retrieval_model_unavailable"
    assert "private-download-diagnostic" not in str(caught.value)
    assert models._MODELS == {"other_resource": unrelated}

    ensure(group)
    required = (
        {"bi_encoder", "cross_encoder", "util"}
        if group == "embedding"
        else {"translation_tokenizer", "translation_model"}
    )
    assert required.issubset(models._MODELS)
    assert models._MODELS["other_resource"] is unrelated
    ensure(group)
    assert factory.call_count == 2


@pytest.mark.parametrize("group", ["embedding", "translation"])
@pytest.mark.parametrize("failure", [ImportError, OSError, RuntimeError, ValueError])
def test_expected_loading_failure_has_a_controlled_contract(loaders, group, failure):
    stage = "bi_encoder" if group == "embedding" else "tokenizer"
    loaders[stage].side_effect = failure("private-loading-diagnostic")
    with pytest.raises(ProcessingError) as caught:
        ensure(group)
    error = caught.value
    assert (error.status_code, error.code, error.retryable) == (
        503,
        "retrieval_model_unavailable",
        False,
    )
    assert models._MODELS == {}


def test_partial_embedding_cache_is_not_considered_ready(loaders):
    old_encoder = object()
    models._MODELS["bi_encoder"] = old_encoder
    models._ensure_embedding_models()
    assert models._MODELS["bi_encoder"] is not old_encoder
    assert {"bi_encoder", "cross_encoder", "util"}.issubset(models._MODELS)
    loaders["cross_encoder"].assert_called_once()


@pytest.mark.parametrize("group", ["embedding", "translation"])
def test_concurrent_first_requests_load_one_model_group(loaders, group):
    stage = "bi_encoder" if group == "embedding" else "tokenizer"
    factory = loaders[stage]
    loaded = factory.return_value
    started, release, second_started = Event(), Event(), Event()

    def load(*args, **kwargs):
        started.set()
        assert release.wait(timeout=2)
        return loaded

    def second_request():
        second_started.set()
        ensure(group)

    factory.side_effect = load
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ensure, group)
        try:
            assert started.wait(timeout=2)
            second = pool.submit(second_request)
            assert second_started.wait(timeout=2)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    factory.assert_called_once()


def test_model_loading_failure_reaches_chat_error_handler(monkeypatch, loaders):
    loaders["cross_encoder"].side_effect = OSError("private-loading-diagnostic")
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
        service,
        "search_food_usda",
        lambda *args, **kwargs: [
            {"fdc_id": 1, "description": "Rice", "data_type": "Foundation"}
        ],
    )
    with TestClient(create_app()) as client:
        response = client.post("/v1/chat", json={"message": "Calories in rice"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "retrieval_model_unavailable"
    assert response.json()["retryable"] is False
    assert "private-loading-diagnostic" not in response.text
    assert models._MODELS == {}
