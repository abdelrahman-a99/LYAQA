"""Call the meal provider and validate selection and change outputs."""

import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from app.ai.agent.meal_edit_prompts import (
    build_meal_change_messages,
    build_meal_selection_messages,
)
from app.ai.contracts.meal_edits import MealChanges, MealSelection
from app.ai.nlp.config import GROQ_MODEL
from app.ai.providers.groq import get_client

logger = logging.getLogger(__name__)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


def call_edit_output(messages, schema: type[OutputModel]) -> OutputModel | None:
    messages = [dict(item) for item in messages]
    messages[0]["content"] += "\nOUTPUT JSON SCHEMA:\n" + json.dumps(
        schema.model_json_schema()
    )
    options = {"reasoning_format": "hidden"} if GROQ_MODEL.startswith("qwen/") else {}
    try:
        response = get_client("meals").chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            **options,
        )
        return schema.model_validate_json(response.choices[0].message.content)
    except Exception:
        logger.exception("Meal edit %s failed", schema.__name__)
        return None


def call_meal_selection(message, history, meals, language) -> MealSelection | None:
    return call_edit_output(
        build_meal_selection_messages(message, history, meals, language), MealSelection
    )


def call_meal_changes(
    message, history, meal, profile, target, language
) -> MealChanges | None:
    return call_edit_output(
        build_meal_change_messages(message, history, meal, profile, target, language),
        MealChanges,
    )
