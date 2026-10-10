"""Execute agent LLM calls and validate their structured outputs."""

import json
from typing import List, Optional

from app.ai.agent.prompts import (
    ALLOWED_UNITS,
    build_food_substitute_messages,
    build_full_meal_revision_messages,
    build_general_chat_messages,
    build_meal_generation_messages,
    build_question_resolution_messages,
    build_rag_answer_messages,
)
from app.ai.contracts.conversation import ResolvedKnowledgeQuestion
from app.ai.contracts.meals import CandidateMeal, SubstituteFood
from app.ai.nlp.config import GROQ_MODEL
from app.ai.providers.errors import completion_text, raise_groq_failure
from app.ai.providers.groq import get_client


def call_meal_generation(
    target_calories: float,
    profile: dict,
    meal_type: Optional[str],
    language: str,
) -> List[dict]:
    """
    Execute the meal-generation LLM call and return a validated list of food dicts.

    Raises a processing error on provider or output validation failure.
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
        raw = json.loads(completion_text(response))
        # Validate through CandidateMeal → each FoodItem validated by Pydantic
        validated = CandidateMeal(**raw)
        foods = [f.model_dump() for f in validated.foods]
        if not foods:
            raise ValueError("Empty candidate meal")

        # Extra unit guard: reject any unit outside the Task 1 vocabulary
        for f in foods:
            if f["unit"] not in ALLOWED_UNITS:
                raise ValueError("Unsupported meal unit")
        return foods
    except Exception as exc:
        raise_groq_failure(exc)


def call_rag_answer(
    question: str,
    passages: List[str],
    language: str,
    history: list[dict] | None = None,
) -> str:
    """
    Execute the RAG-answer LLM call and return the response text.

    Raises a processing error on provider or output validation failure.
    """
    messages = build_rag_answer_messages(question, passages, language, history)
    try:
        response = get_client("rag").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return completion_text(response)
    except Exception as exc:
        raise_groq_failure(exc)


def call_food_substitute(
    forbidden_food: str,
    profile: dict,
    language: str,
    context_meal: Optional[List[dict]] = None,
) -> dict:
    """
    Execute the food-substitute LLM call and return a single validated FoodItem dict.

    Raises a processing error on provider or output validation failure.
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
        raw = json.loads(completion_text(response))
        validated = SubstituteFood(**raw)
        food = validated.food.model_dump()
        if food["unit"] not in ALLOWED_UNITS:
            raise ValueError("Unsupported food unit")
        return food
    except Exception as exc:
        raise_groq_failure(exc)


def call_full_meal_revision(
    current_foods: List[dict],
    target_calories: float,
    profile: dict,
    meal_type: Optional[str],
    language: str,
    issues: List[str],
) -> List[dict]:
    """
    LLM revision: regenerate the entire meal acknowledging the current issues.

    Used when scale_meal_to_target failed or when protein is the failing check.
    Raises a processing error on provider or output validation failure.
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
        raw = json.loads(completion_text(response))
        validated = CandidateMeal(**raw)
        foods = [f.model_dump() for f in validated.foods]
        if not foods:
            raise ValueError("Empty candidate meal")
        for f in foods:
            if f["unit"] not in ALLOWED_UNITS:
                raise ValueError("Unsupported meal unit")
        return foods
    except Exception as exc:
        raise_groq_failure(exc)


def call_general_chat(
    message: str, language: str, history: list[dict] | None = None
) -> str:
    """Plain LLM reply with no tools — no numeric data invented."""
    messages = build_general_chat_messages(message, language, history)
    try:
        response = get_client("rag").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
        )
        return completion_text(response)
    except Exception as exc:
        raise_groq_failure(exc)


def call_question_resolution(
    question: str, language: str, history: list[dict]
) -> ResolvedKnowledgeQuestion:
    """Return a resolved question or explicit ambiguity; raise on failure."""
    messages = build_question_resolution_messages(question, language, history)
    try:
        response = get_client("rag").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "resolved_knowledge_question",
                    "strict": True,
                    "schema": ResolvedKnowledgeQuestion.model_json_schema(),
                },
            },
        )
        return ResolvedKnowledgeQuestion.model_validate_json(completion_text(response))
    except Exception as exc:
        raise_groq_failure(exc)
