import time

from app.ai.agent.tools.common import _envelope, _log
from app.ai.retrieval.service import get_food_nutrition
from app.ai.contracts.errors import ProcessingError


def get_nutrition(food_name_en: str, food_name_original: str, intent: str = "") -> dict:
    """Look up one food and wrap the retrieval result for the agent.

    Search with the English name and preserve the original name for matching
    and presentation. Log duration and return a structured success or failure.
    Exceptions raised by the retrieval layer propagate to the calling flow.
    """
    t0 = time.perf_counter()
    result = get_food_nutrition(food_name_en, original_query=food_name_original)
    duration = (time.perf_counter() - t0) * 1000
    success = result.get("success", False)
    _log(
        "get_nutrition",
        intent,
        {"food_name_en": food_name_en, "food_name_original": food_name_original},
        success,
        duration,
    )
    if success:
        return _envelope(True, result)
    return _envelope(False, error=result.get("error"), error_code=result.get("error"))


def retrieve_fitness_knowledge(query: str, intent: str = "") -> dict:
    """
    Delegate to rag.retrieve.retrieve_fitness_knowledge.

    Imported lazily to avoid loading FAISS/sentence-transformers at module
    import time when the orchestrator is handling a non-RAG intent.
    """
    t0 = time.perf_counter()
    try:
        from app.ai.rag.retrieve import retrieve_fitness_knowledge as _rag

        result = _rag(query)
        duration = (time.perf_counter() - t0) * 1000
        success = result.get("success", False)
        _log(
            "retrieve_fitness_knowledge",
            intent,
            {"query_len": len(query)},
            success,
            duration,
        )
        return result
    except ProcessingError:
        raise
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        _log(
            "retrieve_fitness_knowledge",
            intent,
            {"query_len": len(query)},
            False,
            duration,
        )
        return _envelope(False, error=str(exc), error_code="rag_error")
