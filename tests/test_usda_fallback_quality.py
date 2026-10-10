from unittest.mock import MagicMock, call

import pytest

from app.ai.retrieval import service


def candidate(fdc_id, final_score=0.8, cross_encoder_score=1.0):
    return {
        "fdc_id": fdc_id,
        "description": "Apples, raw",
        "data_type": "Foundation",
        "semantic_score": 0.9,
        "cross_encoder_score": cross_encoder_score,
        "final_score": final_score,
    }


def complete_nutrition():
    return {
        "calories_per_100g": 52.0,
        "protein_per_100g": 0.3,
        "carbs_per_100g": 14.0,
        "fat_per_100g": 0.2,
        "portions": [],
    }


def mock_pipeline(monkeypatch, candidates):
    monkeypatch.setattr(
        service,
        "normalize_food_query",
        lambda _: {"normalized_food": "apple", "method": "test"},
    )

    for name in (
        "search_food_usda",
        "semantic_retrieve",
        "dynamic_rerank",
    ):
        monkeypatch.setattr(
            service,
            name,
            lambda *a, **kw: candidates,
        )

    details = MagicMock(
        side_effect=lambda fdc_id: None if fdc_id == 1 else complete_nutrition(),
    )
    monkeypatch.setattr(service, "get_food_details", details)
    return details


@pytest.mark.parametrize(
    "final_score,ce_score",
    [(0.1, 1.0), (0.7, -9.0), (0.1, -9.0)],
)
def test_incomplete_top_match_does_not_allow_a_poor_fallback(
    monkeypatch,
    final_score,
    ce_score,
):
    details = mock_pipeline(
        monkeypatch,
        [candidate(1), candidate(2, final_score, ce_score)],
    )

    result = service.get_food_nutrition(
        "apple",
        original_query="تفاح",
    )

    assert result["success"] is False
    assert result["error"] == "complete_nutrition_not_found"
    assert result["original_query"] == "تفاح"
    details.assert_called_once_with(1)


@pytest.mark.parametrize(
    "final_score,ce_score",
    [(0.1, 1.0), (0.7, -9.0), (0.1, -9.0)],
)
def test_poor_fallback_is_skipped_before_a_valid_later_match(
    monkeypatch,
    final_score,
    ce_score,
):
    details = mock_pipeline(
        monkeypatch,
        [
            candidate(1),
            candidate(2, final_score, ce_score),
            candidate(3, 0.6),
        ],
    )

    result = service.get_food_nutrition("apple")

    assert result["success"] is True
    assert result["fdc_id"] == 3
    assert result["selected_rank"] == 3
    assert details.call_args_list == [call(1), call(3)]


def test_fallback_at_the_quality_thresholds_is_accepted(monkeypatch):
    details = mock_pipeline(
        monkeypatch,
        [candidate(1), candidate(2, 0.35, -3.0)],
    )

    result = service.get_food_nutrition("apple")

    assert result["success"] is True
    assert result["fdc_id"] == 2
    assert result["selected_rank"] == 2
    assert details.call_args_list == [call(1), call(2)]
