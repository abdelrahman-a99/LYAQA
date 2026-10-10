"""Copy recent user-facing messages for request-local answer generation."""

MAX_CONTEXT_MESSAGES = 6


def recent_history(history: list[dict] | None) -> list[dict]:
    """Keep recent user/assistant text without mutating the caller's history."""
    if not isinstance(history, list):
        return []
    messages = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({"role": role, "content": content})
    return messages[-MAX_CONTEXT_MESSAGES:]
