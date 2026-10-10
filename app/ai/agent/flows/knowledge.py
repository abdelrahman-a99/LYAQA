"""Distinguish missing context or evidence from failed RAG execution."""

from app.ai.agent.llm import call_question_resolution, call_rag_answer
from app.ai.agent.state import AgentState
from app.ai.agent.tools import retrieve_fitness_knowledge
from app.ai.contracts.errors import ProcessingError
from app.ai.conversation.context import recent_history


def handle_rag_flow(
    query: str,
    language: str,
    intent: str,
    history: list[dict] | None = None,
    state: AgentState | None = None,
) -> str:
    """
    food_substitution / fitness_nutrition_question flow (spec section 4).

    RAG is primary, not optional. If confidence is low, reply honestly.
    No USDA, no validate_plan, no revision loop.
    """
    history = recent_history(history)
    if history:
        resolved = call_question_resolution(query, language, history)
        if resolved is None:
            raise ProcessingError(
                "The language model returned an invalid output.",
                code="llm_invalid_output",
                status_code=502,
            )

        if resolved.question is None:
            if state is not None:
                state.status = "needs_clarification"
            if language in ("arabic", "mixed"):
                return "ممكن توضح المقصود أو تكتب السؤال كاملًا؟"
            return "Could you clarify what you mean or write the full question?"
        query = resolved.question

    result = retrieve_fitness_knowledge(query, intent=intent)

    if not result["success"]:
        code = result.get("error_code")

        if code == "low_confidence_retrieval":
            if state is not None:
                state.status = "insufficient_knowledge"

            if language in ("arabic", "mixed"):
                return "عذراً، لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال بدقة."

            return (
                "I don't have enough reliable information "
                "to answer this question confidently."
            )

        raise ProcessingError(
            "The knowledge retrieval system is unavailable.",
            code=(
                "rag_configuration_error"
                if code == "index_not_found"
                else "rag_unavailable"
            ),
            status_code=500 if code == "index_not_found" else 503,
        )

    passages = result["data"]["passages"]

    if not passages:
        raise ProcessingError(
            "The knowledge retrieval system returned no usable passages.",
            code="rag_unavailable",
            status_code=503,
        )

    answer = call_rag_answer(query, passages, language, history)
    if not answer:
        raise ProcessingError(
            "The language model returned an invalid output.",
            code="llm_invalid_output",
            status_code=502,
        )

    return answer
