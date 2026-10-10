import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.ai.retrieval import models, service, usda_client
from app.ai.retrieval.cache import clear_caches
from app.ai.retrieval.normalization import normalize_food_query
from app.ai.retrieval.portions import convert_to_grams


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    clear_caches()

    monkeypatch.setattr(
        usda_client,
        "get_api_key",
        lambda: "test-key",
    )

    yield

    clear_caches()


def test_arabic_food_mapping():
    result = normalize_food_query("فَراخ مشوية")

    assert result["normalized_food"] == "grilled chicken"
    assert result["method"] == "direct_food_mapping"


def test_usda_search_cache(monkeypatch):
    response = MagicMock(status_code=200)

    response.json.return_value = {
        "foods": [
            {
                "fdcId": 1,
                "description": "Apples, raw",
                "dataType": "Foundation",
            }
        ]
    }

    get = MagicMock(return_value=response)

    monkeypatch.setattr(usda_client.requests, "get", get)

    first = usda_client.search_food_usda("apple")
    second = usda_client.search_food_usda("apple")

    assert first == second
    assert first[0]["fdc_id"] == 1

    get.assert_called_once()


def test_zero_nutrients_are_valid(monkeypatch):
    response = MagicMock(status_code=200)

    response.json.return_value = {
        "fdcId": 1,
        "description": "Test food",
        "foodNutrients": [
            {"nutrient": {"id": 1008}, "amount": 0},
            {"nutrient": {"id": 2047}, "amount": 99},
            {"nutrient": {"id": 1003}, "amount": 0},
            {"nutrient": {"id": 1005}, "amount": 0},
            {"nutrient": {"id": 1004}, "amount": 0},
        ],
    }

    monkeypatch.setattr(
        usda_client.requests,
        "get",
        MagicMock(return_value=response),
    )

    result = usda_client.get_food_details(1)

    assert result["calories_per_100g"] == 0
    assert result["protein_per_100g"] == 0
    assert usda_client.nutrition_is_complete(result)


def test_portion_conversion_stays_exact():
    assert convert_to_grams(2, "cup", []) is None
    assert convert_to_grams(100, "ml", []) is None

    portions = [
        {
            "unit": "cup",
            "amount": 1,
            "gram_weight": 150,
            "modifier": None,
        }
    ]

    result = convert_to_grams(2, "cup", portions)

    assert result["grams"] == 300
    assert result["source"] == "usda_portion"


def test_embedding_models_are_loaded_once(monkeypatch):
    fake = ModuleType("sentence_transformers")

    fake.SentenceTransformer = MagicMock()
    fake.CrossEncoder = MagicMock()
    fake.util = MagicMock()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        fake,
    )

    monkeypatch.setattr(models, "_MODELS", {})
    monkeypatch.setattr(models, "get_device", lambda: "cpu")

    models._ensure_embedding_models()
    models._ensure_embedding_models()

    fake.SentenceTransformer.assert_called_once_with(
        models.BI_ENCODER_MODEL_NAME,
        device="cpu",
    )

    fake.CrossEncoder.assert_called_once_with(
        models.RERANKER_MODEL_NAME,
        device="cpu",
    )


@pytest.mark.parametrize(
    "score,expected_success",
    [
        (1.0, True),
        (-4.0, False),
    ],
)
def test_food_pipeline_quality_gate(
    monkeypatch,
    score,
    expected_success,
):
    candidate = {
        "fdc_id": 1,
        "description": "Apples, raw",
        "data_type": "Foundation",
        "semantic_score": 0.9,
        "cross_encoder_score": score,
        "final_score": 0.8,
    }

    monkeypatch.setattr(
        service,
        "search_food_usda",
        lambda *a, **kw: [candidate],
    )

    monkeypatch.setattr(
        service,
        "semantic_retrieve",
        lambda *a, **kw: [candidate],
    )

    monkeypatch.setattr(
        service,
        "dynamic_rerank",
        lambda *a, **kw: [candidate],
    )

    details = MagicMock(
        return_value={
            "calories_per_100g": 52,
            "protein_per_100g": 0.3,
            "carbs_per_100g": 14,
            "fat_per_100g": 0.2,
            "portions": [],
        }
    )

    monkeypatch.setattr(service, "get_food_details", details)

    result = service.get_food_nutrition(
        "apple",
        original_query="تفاح",
    )

    assert result["success"] is expected_success
    assert result["original_query"] == "تفاح"

    if expected_success:
        assert result["calories_per_100g"] == 52

        details.assert_called_once_with(1)

    else:
        assert result["error"] == "food_not_found"

        details.assert_not_called()
