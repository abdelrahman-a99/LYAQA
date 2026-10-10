"""Execute agent LLM calls and validate their structured outputs."""

import json
import logging
from typing import List, Optional

from app.ai.agent.prompts import (
    ALLOWED_UNITS,
    build_food_substitute_messages,
    build_full_meal_revision_messages,
    build_general_chat_messages,
    build_meal_generation_messages,
    build_rag_answer_messages,
)
from app.ai.contracts.meals import CandidateMeal, SubstituteFood
from app.ai.nlp.config import GROQ_MODEL
from app.ai.providers.groq import get_client

logger = logging.getLogger(__name__)


def call_meal_generation(
    target_calories: float,
    profile: dict,
    meal_type: Optional[str],
    language: str,
) -> Optional[List[dict]]:
    """
    Execute the meal-generation LLM call and return a validated list of food dicts.

    Returns None if the call fails or the output fails FoodItem validation.
    Validation uses Task 1's FoodItem schema (extra="forbid") — any unknown
    field or disallowed unit is rejected before the list ever reaches USDA.
    """
    messages = build_meal_generation_messages(
        target_calories, profile, meal_type, language
    )
    try:
        response = get_client("meals").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_meal",
                    "strict": True,
                    "schema": CandidateMeal.model_json_schema(),
                },
            },
        )
        raw = json.loads(response.choices[0].message.content)
        # Validate through CandidateMeal → each FoodItem validated by Pydantic
        validated = CandidateMeal(**raw)
        foods = [f.model_dump() for f in validated.foods]

        # Extra unit guard: reject any unit outside the Task 1 vocabulary
        for f in foods:
            if f["unit"] not in ALLOWED_UNITS:
                return None
        return foods
    except Exception:
        return None


def call_rag_answer(question: str, passages: List[str], language: str) -> Optional[str]:
    """
    Execute the RAG-answer LLM call and return the response text.

    Returns None on failure.
    """
    messages = build_rag_answer_messages(question, passages, language)
    try:
        response = get_client("rag").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def call_food_substitute(
    forbidden_food: str,
    profile: dict,
    language: str,
    context_meal: Optional[List[dict]] = None,
) -> Optional[dict]:
    """
    Execute the food-substitute LLM call and return a single validated FoodItem dict.

    Returns None on failure or validation error.
    """
    messages = build_food_substitute_messages(
        forbidden_food, profile, language, context_meal
    )
    try:
        response = get_client("meals").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "substitute_food",
                    "strict": True,
                    "schema": SubstituteFood.model_json_schema(),
                },
            },
        )
        raw = json.loads(response.choices[0].message.content)
        validated = SubstituteFood(**raw)
        food = validated.food.model_dump()
        if food["unit"] not in ALLOWED_UNITS:
            return None
        return food
    except Exception:
        return None


def call_full_meal_revision(
    current_foods: List[dict],
    target_calories: float,
    profile: dict,
    meal_type: Optional[str],
    language: str,
    issues: List[str],
) -> Optional[List[dict]]:
    """
    LLM revision: regenerate the entire meal acknowledging the current issues.

    Used when scale_meal_to_target failed or when protein is the failing check.
    Returns None on failure.
    """
    messages = build_full_meal_revision_messages(
        current_foods, target_calories, profile, meal_type, language, issues
    )

    try:
        response = get_client("meals").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_meal",
                    "strict": True,
                    "schema": CandidateMeal.model_json_schema(),
                },
            },
        )
        raw = json.loads(response.choices[0].message.content)
        validated = CandidateMeal(**raw)
        foods = [f.model_dump() for f in validated.foods]
        for f in foods:
            if f["unit"] not in ALLOWED_UNITS:
                return None
        return foods
    except Exception:
        return None


def call_general_chat(message: str, language: str) -> str:
    """Plain LLM reply with no tools — no numeric data invented."""
    messages = build_general_chat_messages(message, language)
    try:
        response = get_client("rag").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("general_chat LLM call failed: %s", exc)
        if language == "arabic":
            return "مرحباً! كيف يمكنني مساعدتك؟"
        return "Hello! How can I help you?"
