from typing import Optional

from app.ai.agent.constants import MEAL_SPLITS
from app.ai.agent.food_lookup import collect_nutrition_for_foods
from app.ai.agent.llm import call_meal_generation
from app.ai.agent.revision import run_revision_loop
from app.ai.agent.state import AgentState
from app.ai.agent.tools import calculate_targets
from app.ai.nutrition.profile import calculate_protein_target
from app.ai.presentation.responses import format_meal_recommendation_response
from app.ai.presentation.validation import validate_meal_nutrition_report


def meal_recommendation_flow(
    target_calories: float,
    profile: dict,
    meal_type: Optional[str],
    language: str,
    min_protein_g: Optional[float],
    state: AgentState,
    intent: str,
) -> dict:
    """
    Full meal_recommendation flow including LLM generation, USDA lookup,
    and the revision loop.

    Returns an orchestrator result dict:
        {success, meal_result, foods, nutrition_list, valid, partial_failures, report}
    """
    # Generate candidate meal from LLM
    foods = call_meal_generation(target_calories, profile, meal_type, language)
    if not foods:
        return {
            "success": False,
            "error": "LLM failed to generate a valid candidate meal.",
        }

    matched_foods, matched_nutrition, failed = collect_nutrition_for_foods(
        foods, intent
    )

    # If all candidates failed USDA, ask LLM for one full replacement list (uncounted)
    if not matched_foods:
        foods = call_meal_generation(target_calories, profile, meal_type, language)
        if foods:
            matched_foods, matched_nutrition, failed = collect_nutrition_for_foods(
                foods, intent
            )

    if not matched_foods:
        return {
            "success": False,
            "error": "All candidate foods failed USDA lookup after two generation attempts.",
        }

    state.revision_count = 0
    meal_result, final_foods, final_nutrition, valid = run_revision_loop(
        matched_foods,
        matched_nutrition,
        profile,
        target_calories,
        min_protein_g,
        language,
        meal_type,
        state,
        intent,
    )

    report = validate_meal_nutrition_report(
        meal_result,
        profile,
        target_calories=target_calories,
        min_protein_g=min_protein_g,
    )

    return {
        "success": True,
        "meal_result": meal_result,
        "foods": final_foods,
        "nutrition_list": final_nutrition,
        "valid": valid,
        "partial_failures": failed,
        "report": report,
    }


def handle_meal_recommendation(
    target_calories_from_message: Optional[float],
    profile: dict,
    meal_type: Optional[str],
    language: str,
    state: AgentState,
) -> str:
    """
    meal_recommendation flow including revision loop (spec section 4 + section 5).

    Target calories priority:
      1. Explicit value in the message (from Task 1).
      2. Meal share of daily target from Task 3 (if profile is complete).
      3. Cannot determine → ask the user.
    """
    intent = "meal_recommendation"
    target_calories = target_calories_from_message
    current_targets = None

    if target_calories is None:
        targets_result = calculate_targets(profile, intent=intent)
        if targets_result["success"]:
            daily = targets_result["data"]["target_calories"]
            share = MEAL_SPLITS.get(meal_type or "", 0.30)
            target_calories = round(daily * share, 0)
            state.targets = targets_result["data"]
            current_targets = targets_result["data"]

    if target_calories is None:
        if language == "arabic":
            return "كم سعرة حرارية تريد في هذه الوجبة؟"
        return "How many calories would you like this meal to be?"

    min_protein_g = None
    meal_share = MEAL_SPLITS.get(meal_type or "", 0.30)

    if current_targets and "macros" in current_targets:
        daily_protein = current_targets["macros"]["protein_g"]
        min_protein_g = round(daily_protein * meal_share, 1)
    elif profile.get("goal") in ("muscle_gain", "weight_loss") and profile.get(
        "weight"
    ):
        daily_protein = calculate_protein_target(profile)
        min_protein_g = round(daily_protein * meal_share, 1)

    flow = meal_recommendation_flow(
        target_calories, profile, meal_type, language, min_protein_g, state, intent
    )

    if not flow["success"]:
        if language == "arabic":
            return f"لم أتمكن من إنشاء وجبة: {flow.get('error', '')}"
        return f"Could not generate a meal: {flow.get('error', '')}"

    return format_meal_recommendation_response(
        flow, target_calories, language, flow["valid"]
    )
