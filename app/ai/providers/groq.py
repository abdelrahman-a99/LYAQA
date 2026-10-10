import os
from functools import lru_cache


@lru_cache(maxsize=3)
def get_client(role: str = "nlp"):
    if role not in {"nlp", "meals", "rag"}:
        raise ValueError(f"Unknown Groq role: {role}")

    key = os.environ.get(f"GROQ_API_KEY_{role.upper()}")

    if not key and role != "nlp":
        key = os.environ.get("GROQ_API_KEY_NLP")

    if not key:
        raise ValueError(f"Groq API key is missing for role: {role}")

    from groq import Groq

    return Groq(api_key=key)
