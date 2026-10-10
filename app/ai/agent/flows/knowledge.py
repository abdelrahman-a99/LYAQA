from app.ai.agent.llm import call_question_resolution, call_rag_answer
from app.ai.agent.tools import retrieve_fitness_knowledge
from app.ai.conversation.context import recent_history


def handle_rag_flow(
    query: str, language: str, intent: str, history: list[dict] | None = None
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
            if language in ("arabic", "mixed"):
                return "تعذر فهم سياق السؤال الآن. حاول مرة أخرى أو اكتب السؤال كاملًا."
            return "I could not resolve the question context. Please retry or write the full question."
        if resolved.question is None:
            if language in ("arabic", "mixed"):
                return "ممكن توضح المقصود أو تكتب السؤال كاملًا؟"
            return "Could you clarify what you mean or write the full question?"
        query = resolved.question

    rag_result = retrieve_fitness_knowledge(query, intent=intent)

    if not rag_result["success"]:
        error_code = rag_result.get("error_code", "")
        if error_code == "low_confidence_retrieval":
            if language == "arabic":
                return "عذراً، لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال بدقة."
            return "I'm sorry, I don't have enough reliable information to answer this question confidently."
        if language == "arabic":
            return "حدث خطأ في البحث. يرجى المحاولة مرة أخرى."
        return "A retrieval error occurred. Please try again."

    passages = rag_result["data"]["passages"]
    answer = call_rag_answer(query, passages, language, history)
    if not answer:
        if language == "arabic":
            return "عذراً، لم أتمكن من صياغة إجابة. يرجى المحاولة مرة أخرى."
        return "Sorry, I could not compose an answer. Please try again."
    return answer
