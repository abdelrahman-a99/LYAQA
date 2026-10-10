from app.ai.agent.llm import call_rag_answer
from app.ai.agent.tools import retrieve_fitness_knowledge


def handle_rag_flow(query: str, language: str, intent: str) -> str:
    """
    food_substitution / fitness_nutrition_question flow (spec section 4).

    RAG is primary, not optional. If confidence is low, reply honestly.
    No USDA, no validate_plan, no revision loop.
    """
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
    answer = call_rag_answer(query, passages, language)
    if not answer:
        if language == "arabic":
            return "عذراً، لم أتمكن من صياغة إجابة. يرجى المحاولة مرة أخرى."
        return "Sorry, I could not compose an answer. Please try again."
    return answer
