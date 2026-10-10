from app.ai.agent.constants import MEAL_SPLITS
from app.ai.agent.flows.meals import meal_recommendation_flow
from app.ai.agent.state import AgentState
from app.ai.agent.tools import calculate_targets, validate_plan
from app.ai.presentation.responses import format_daily_plan_response
from app.ai.presentation.validation import validate_daily_nutrition_report


def handle_daily_plan(profile: dict, language: str, state: AgentState) -> str:
    """
    daily_plan flow (spec section 4).

    status already guaranteed complete by Task 1 (PROFILE_DEPENDENT_INTENTS).
    Tool 3 → split into 4 meals → meal_recommendation flow per meal (independent
    revision budgets) → validate_daily_nutrition on totals.
    """
    intent = "daily_plan"
    targets_result = calculate_targets(profile, intent=intent)
    if not targets_result["success"]:
        if language == "arabic":
            return "يحتاج الحساب اليومي إلى ملف شخصي مكتمل."
        return "A daily plan requires a complete profile."

    state.targets = targets_result["data"]
    daily_target = targets_result["data"]["target_calories"]

    meal_results = {}
    all_foods = []
    all_nutrition = []
    partial_failures_all = []

    for meal_name, share in MEAL_SPLITS.items():
        meal_target = round(daily_target * share, 0)

        min_protein_g = None
        if profile.get("goal") in ("muscle_gain", "weight_loss"):
            weight = profile.get("weight")
            if weight:
                min_protein_g = round(state.targets["macros"]["protein_g"] * share, 1)

        # Each meal gets its own fresh revision counter
        state.revision_count = 0

        flow = meal_recommendation_flow(
            meal_target, profile, meal_name, language, min_protein_g, state, intent
        )

        meal_results[meal_name] = flow
        if flow.get("success"):
            all_foods += flow.get("foods", [])
            all_nutrition += flow.get("nutrition_list", [])
            partial_failures_all += flow.get("partial_failures", [])

    # Collect the raw meal_result of each successful meal for daily validation
    successful_meal_totals = [
        flow["meal_result"]
        for flow in meal_results.values()
        if flow.get("success") and flow.get("meal_result")
    ]

    daily_val = validate_plan(
        successful_meal_totals, profile, mode="day", intent=intent
    )
    daily_valid = (
        daily_val.get("data", {}).get("valid", False) if daily_val["success"] else False
    )

    day_report = validate_daily_nutrition_report(successful_meal_totals, profile)

    # day_totals is kept strictly for text presentation
    day_totals = _aggregate_day_totals(meal_results)

    return format_daily_plan_response(
        meal_results,
        day_totals,
        daily_valid,
        day_report,
        partial_failures_all,
        language,
        daily_target,
    )


def _aggregate_day_totals(meal_results: dict) -> dict:
    """Sum successful meal totals for the daily-plan response formatter."""
    total_cal = 0.0
    total_prot = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    for flow in meal_results.values():
        if not flow.get("success"):
            continue
        m = flow.get("meal_result", {})
        total_cal += m.get("total_calories", 0.0)
        total_prot += m.get("total_protein_g", 0.0)
        total_carbs += m.get("total_carbs_g", 0.0)
        total_fat += m.get("total_fat_g", 0.0)

    return {
        "success": True,
        "total_calories": round(total_cal, 1),
        "total_protein_g": round(total_prot, 1),
        "total_carbs_g": round(total_carbs, 1),
        "total_fat_g": round(total_fat, 1),
        "foods": [],
        "unmatched_foods": [],
    }
