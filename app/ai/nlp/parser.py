"""Structured NLP extraction with the original retry and fallback behavior."""

from __future__ import annotations

import json
from typing import List, Optional

from app.ai.contracts.nlp import NLPOutput
from app.ai.nlp.config import GROQ_MODEL
from app.ai.nlp.language import detect_language
from app.ai.nlp.normalization import normalize_output
from app.ai.nlp.prompts import build_parser_messages
from app.ai.providers.groq import get_client


def _groq_call(messages: List[dict]) -> dict:
    """Execute one Groq chat completion with strict JSON schema."""
    response = get_client("nlp").chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "nlp_output",
                "strict": True,
                "schema": NLPOutput.model_json_schema(),
            },
        },
    )
    return json.loads(response.choices[0].message.content)


def _safe_fallback(user_message: str) -> dict:
    """Never crash: return a safe general_chat when all retries fail."""
    return {
        "intent": "general_chat",
        "language_detected": detect_language(user_message),
        "raw_text": user_message,
        "meal_type": None,
        "target_calories": None,
        "requested_substitution": None,
        "foods": [],
        "profile_data": None,
    }


def _parse(
    user_message: str,
    history: Optional[List[dict]] = None,
) -> dict:
    messages = build_parser_messages(user_message, history=history)

    for attempt in range(2):
        try:
            parsed = _groq_call(messages)

            parsed["language_detected"] = detect_language(user_message)
            parsed["raw_text"] = user_message

            return normalize_output(parsed)

        except Exception:
            if attempt == 1:
                return normalize_output(_safe_fallback(user_message))

    return normalize_output(_safe_fallback(user_message))


def parse_user_input(user_message: str) -> dict:
    """Parse one message with one retry and a safe fallback."""
    return _parse(user_message)


def parse_with_context(
    user_message: str,
    history: Optional[List[dict]] = None,
) -> dict:
    """Parse a message with optional conversation history."""
    return _parse(user_message, history)


def get_food_data(user_message: str) -> dict:
    """
    Extract the Meal/Food data along with message metadata (intent, language, raw_text).
    Perfect for Task 2 (USDA API search), Task 3 (food nutrition calculation), and Task 4.

    Example:
        >>> get_food_data("كلت 200g فراخ مشوية مع طبق رز")
        {
            "intent": "calculate_calories",
            "language_detected": "arabic",
            "raw_text": "كلت 200g فراخ مشوية مع طبق رز",
            "foods": [
                {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
                {"food_name_original": "طبق رز", "food_name_en": "cooked white rice", "quantity": 1.0, "unit": "bowl"}
            ]
        }
    """
    parsed = parse_user_input(user_message)
    return {
        "intent": parsed.get("intent"),
        "language_detected": parsed.get("language_detected"),
        "raw_text": parsed.get("raw_text", user_message),
        "meal_type": parsed.get("meal_type"),
        "target_calories": parsed.get("target_calories"),
        "requested_substitution": parsed.get("requested_substitution"),
        "foods": parsed.get("foods", []),
    }


def get_user_profile(user_message: str) -> Optional[dict]:
    """
    Extract ONLY the 10-field User Profile from user message.
    Perfect for Task 3 (BMR, TDEE, target calories engine).

    Example:
        >>> get_user_profile("أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض")
        {
            "birth_date": "2004-05-10",
            "gender": "male",
            "weight": 90.0,
            "height": 175.0,
            "activity_level": "moderate",
            "goal": "weight_loss",
            "dietary_restrictions": [],
            "forbidden_foods": ["eggs"],
            "preferred_foods": [],
            "allergies": []
        }
    """
    parsed = parse_user_input(user_message)
    return parsed.get("profile_data")
