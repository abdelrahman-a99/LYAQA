from app.ai.agent.llm import call_general_chat


def handle_general_chat(message: str, language: str) -> str:
    """Delegate general chat to the agent LLM boundary."""
    return call_general_chat(message, language)
