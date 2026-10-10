"""Compose normalization, USDA lookup, ranking and nutrient selection."""

from __future__ import annotations

from app.ai.retrieval.matching import (
    dynamic_rerank,
    semantic_retrieve,
)
from app.ai.retrieval.normalization import normalize_food_query
from app.ai.retrieval.usda_client import (
    get_food_details,
    nutrition_is_complete,
    search_food_usda,
)


def get_food_nutrition(food_name, original_query=None, search_k=50, rerank_k=20):
    """
    Full pipeline: normalize → search → semantic → rerank → fetch details.

    `original_query` lets the caller (Task 4) echo back the user-facing name
    (e.g. "فراخ مشوية") so Task 3's matching check still succeeds even though
    USDA is searched with the English name.
    """
    echoed_original = original_query if original_query is not None else food_name

    if not isinstance(food_name, str) or not food_name.strip():
        return {"success": False, "error": "empty_food_name"}

    normalized = normalize_food_query(food_name)
    query = normalized["normalized_food"]

    if not query:
        return {
            "success": False,
            "original_query": echoed_original,
            "error": "normalization_failed",
        }

    search_results = search_food_usda(query, max_results=search_k)
    if not search_results:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "food_not_found",
        }

    semantic_candidates = semantic_retrieve(
        query, search_results, top_k=rerank_k, prefer_generic=True
    )
    if not semantic_candidates:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "semantic_retrieval_failed",
        }

    ranked_results = dynamic_rerank(query, semantic_candidates)
    if not ranked_results:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "ranking_failed",
        }

    # ── Reject low-quality matches ──────────────────────────────────────
    # Two-signal gate: a match must pass BOTH:
    #   1. final_match_score — composite (semantic + rerank features)
    #   2. cross_encoder_score — raw MS MARCO score. Strong matches are
    #      positive; noise is negative. "unobtainium fruit" vs "Fruit, NFS"
    #      gets ~-8 here even though semantic similarity is high, because
    #      the cross-encoder actually reads the full query-doc pair.
    MIN_FINAL_MATCH_SCORE = 0.35
    MIN_CROSS_ENCODER_SCORE = -3.0

    top = ranked_results[0]
    top_final = top.get("final_score", 0.0)
    top_ce = top.get("cross_encoder_score", -999.0)

    if top_final < MIN_FINAL_MATCH_SCORE or top_ce < MIN_CROSS_ENCODER_SCORE:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "food_not_found",
            "details": {
                "top_final_score": top_final,
                "top_cross_encoder_score": top_ce,
            },
        }

    selected_match = None
    nutrition = None
    selected_rank = None

    for rank, candidate in enumerate(ranked_results, start=1):
        # Apply the same quality gate to every fallback candidate.
        if (
            candidate.get("final_score", 0.0) < MIN_FINAL_MATCH_SCORE
            or candidate.get("cross_encoder_score", -999.0) < MIN_CROSS_ENCODER_SCORE
        ):
            continue

        candidate_details = get_food_details(candidate["fdc_id"])
        if nutrition_is_complete(candidate_details):
            selected_match = candidate
            nutrition = candidate_details
            selected_rank = rank
            break

    if selected_match is None:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "complete_nutrition_not_found",
        }

    return {
        "success": True,
        "original_query": echoed_original,
        "normalized_query": query,
        "normalization_method": normalized["method"],
        "fdc_id": selected_match["fdc_id"],
        "food_name": selected_match["description"],
        "data_type": selected_match["data_type"],
        "selected_rank": selected_rank,
        "semantic_score": selected_match["semantic_score"],
        "cross_encoder_score": selected_match["cross_encoder_score"],
        "final_match_score": selected_match["final_score"],
        "calories_per_100g": nutrition["calories_per_100g"],
        "protein_per_100g": nutrition["protein_per_100g"],
        "carbs_per_100g": nutrition["carbs_per_100g"],
        "fat_per_100g": nutrition["fat_per_100g"],
        "portions": nutrition.get("portions", []),
        "source": "USDA FoodData Central",
    }
