from app.ai.agent.llm import call_general_chat


def handle_general_chat(
    message: str, language: str, history: list[dict] | None = None
) -> str:
    """Delegate general chat to the agent LLM boundary."""
    return call_general_chat(message, language, history)
