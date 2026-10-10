"""Public entry point for the seven agent tools."""

from app.ai.agent.tools.nutrition import (
    calculate_meal_nutrition,
    calculate_targets,
    scale_meal,
    validate_plan,
)
from app.ai.agent.tools.parsing import parse_query
from app.ai.agent.tools.retrieval import (
    get_nutrition,
    retrieve_fitness_knowledge,
)

__all__ = [
    "parse_query",
    "get_nutrition",
    "calculate_targets",
    "calculate_meal_nutrition",
    "validate_plan",
    "retrieve_fitness_knowledge",
    "scale_meal",
]
