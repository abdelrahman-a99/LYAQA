from app.ai.retrieval.cache import DETAILS_CACHE, SEARCH_CACHE
from app.ai.retrieval.matching import (
    dynamic_rerank,
    semantic_retrieve,
)
from app.ai.retrieval.normalization import normalize_food_query
from app.ai.retrieval.service import get_food_nutrition
from app.ai.retrieval.usda_client import (
    get_food_details,
    search_food_usda,
)


def _inspect_food_matches(food_name, top_n=10):
    """Test helper: return top-N ranked USDA matches for a query."""
    import pandas as pd

    normalized = normalize_food_query(food_name)
    query = normalized["normalized_food"]

    search_results = search_food_usda(query, max_results=50)
    semantic_candidates = semantic_retrieve(query, search_results, top_k=20)
    ranked_results = dynamic_rerank(query, semantic_candidates)

    if not ranked_results:
        return pd.DataFrame()

    df = pd.DataFrame(ranked_results[:top_n])
    return df[
        [
            "fdc_id",
            "description",
            "data_type",
            "semantic_score",
            "cross_encoder_score",
            "query_coverage",
            "candidate_precision",
            "lexical_score",
            "final_score",
        ]
    ]


if __name__ == "__main__":
    from app.config import load_environment

    load_environment(".env")

    import inspect

    import pandas as pd

    # ─── Top candidates ──────────────────────────────────────────────────
    print(_inspect_food_matches("chicken breast"))
    print(_inspect_food_matches("eggs"))
    print(_inspect_food_matches("رز أبيض"))
    print(_inspect_food_matches("عيش بلدي"))

    # ─── Semantic retrieval signature ────────────────────────────────────
    print(inspect.signature(semantic_retrieve))

    check_results = semantic_retrieve(
        "chicken breast",
        search_food_usda("chicken breast", max_results=50),
        top_k=20,
        prefer_generic=True,
    )
    print(
        "Branded candidates:",
        sum(item["data_type"] == "Branded" for item in check_results),
    )
    print(check_results[0])

    # ─── Error handling ──────────────────────────────────────────────────
    error_tests = ["", "     ", "asdasdasdasdasdfood999999"]
    for food in error_tests:
        print("Input:", repr(food))
        print(get_food_nutrition(food))
        print("-" * 70)

    # ─── Full flow ───────────────────────────────────────────────────────
    test_foods = [
        "chicken breast",
        "فراخ مشوية",
        "رز أبيض",
        "eggs",
        "عيش بلدي",
        "banana",
        "salmon",
        "بطاطس مسلوقة",
        "زبادي",
        "oatmeal",
    ]

    test_df = pd.DataFrame([get_food_nutrition(f) for f in test_foods])
    print(test_df)

    # ─── Normalization ───────────────────────────────────────────────────
    normalization_tests = [
        "chicken breast",
        "فراخ مشوية",
        "رز أبيض",
        "عيش بلدي",
        "بطاطس مسلوقة",
        "زبادي",
    ]
    normalization_df = pd.DataFrame(
        [{"input": f, **normalize_food_query(f)} for f in normalization_tests]
    )
    print(normalization_df)

    # ─── Search + details ────────────────────────────────────────────────
    search_test = search_food_usda("grilled chicken", max_results=10)
    print(pd.DataFrame(search_test).head(10))
    print("Number of results:", len(search_test))

    if search_test:
        print(get_food_details(search_test[0]["fdc_id"]))

    # ─── Ranking comparison ──────────────────────────────────────────────
    for food in [
        "chicken breast",
        "eggs",
        "white rice",
        "grilled chicken",
        "pita bread",
    ]:
        print("\n", "=" * 80)
        print("QUERY:", food)
        print(_inspect_food_matches(food, top_n=5))

    # ─── Full flow (again, with selected columns) ────────────────────────
    full_flow_df = pd.DataFrame([get_food_nutrition(f) for f in test_foods])
    print(
        full_flow_df[
            [
                "original_query",
                "normalized_query",
                "food_name",
                "semantic_score",
                "cross_encoder_score",
                "final_match_score",
                "calories_per_100g",
                "protein_per_100g",
                "carbs_per_100g",
                "fat_per_100g",
            ]
        ]
    )

    # ─── Cache behaviour ─────────────────────────────────────────────────
    SEARCH_CACHE.clear()
    DETAILS_CACHE.clear()
    print("Caches cleared.")

    print("FIRST CALL")
    get_food_nutrition("grilled chicken")

    print("\nSECOND CALL")
    get_food_nutrition("grilled chicken")

    print("\nSearch cache items:", len(SEARCH_CACHE))
    print("Details cache items:", len(DETAILS_CACHE))

    # ─── Save results ────────────────────────────────────────────────────
    test_df.to_csv("dynamic_usda_retrieval_results.csv", index=False)
    print("Results saved successfully!")
