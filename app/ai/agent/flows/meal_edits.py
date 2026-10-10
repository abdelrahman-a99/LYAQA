"""Edit one previous meal and return a complete group only after valid checks."""

from copy import deepcopy

from app.ai.agent.food_lookup import collect_nutrition_for_foods
from app.ai.agent.meal_edit_llm import call_meal_changes, call_meal_selection
from app.ai.agent.prompts import ALLOWED_UNITS
from app.ai.agent.state import AgentState
from app.ai.agent.tools import calculate_meal_nutrition
from app.ai.contracts.errors import ProcessingError
from app.ai.conversation.meal_edits import apply_food_changes
from app.ai.conversation.meals import build_meal_snapshot
from app.ai.presentation.meal_edits import format_meal_edit_response
from app.ai.presentation.validation import validate_meal_nutrition_report


def edit_reply(state, status, language, arabic, english):
    state.status = status
    return arabic if language in ("arabic", "mixed") else english


def handle_meal_edit(
    message: str,
    profile: dict,
    language: str,
    history: list[dict],
    state: AgentState,
    *,
    meal_type: str | None = None,
    target_calories: float | None = None,
) -> str:
    context = state.meal_context
    if not context:
        return edit_reply(
            state,
            "needs_meal_context",
            language,
            "مفيش وجبة سابقة متاحة للتعديل. اطلب اقتراح وجبة أولًا.",
            "No previous meal is available to edit. Request a meal first.",
        )
    candidates = [
        meal for meal in context if not meal_type or meal["meal_type"] == meal_type
    ]
    selection = (
        call_meal_selection(message, history, candidates, language)
        if candidates
        else None
    )
    if not candidates or (selection is not None and selection.meal_id is None):
        return edit_reply(
            state,
            "needs_clarification",
            language,
            "حدد الوجبة المقصودة والتغيير المطلوب؛ الوجبات السابقة لم تتغير.",
            "Specify the meal and requested change; previous meals are unchanged.",
        )
    if selection is None:
        raise ProcessingError(
            "The language model returned an invalid output.",
            code="llm_invalid_output",
            status_code=502,
        )
    selected = next(
        (meal for meal in candidates if meal["meal_id"] == selection.meal_id), None
    )
    if selected is None:
        return edit_reply(
            state,
            "needs_clarification",
            language,
            "لم أجد الوجبة المقصودة ضمن الوجبات المتاحة. وضّح أي وجبة تريد تعديلها.",
            "That meal is not available. Clarify which previous meal to edit.",
        )
    target = (
        target_calories if target_calories is not None else selected["target_calories"]
    )
    changes = call_meal_changes(message, history, selected, profile, target, language)
    if changes is None:
        raise ProcessingError(
            "The language model returned an invalid output.",
            code="llm_invalid_output",
            status_code=502,
        )
    if not changes.changes:
        return edit_reply(
            state,
            "needs_clarification",
            language,
            "وضح المكوّن والتغيير المطلوب، أو هدف السعرات الجديد.",
            "Clarify the ingredient and change, or the new calorie target.",
        )
    try:
        if any(
            change.food and change.food.unit not in ALLOWED_UNITS
            for change in changes.changes
        ):
            raise ValueError("Unsupported unit")
        foods = apply_food_changes(selected["foods"], changes)
    except ValueError:
        return edit_reply(
            state,
            "meal_edit_rejected",
            language,
            "التعديل يحتوي على مكوّن أو كمية أو وحدة غير صالحة؛ الوجبة السابقة لم تتغير.",
            "The edit contains an invalid food change or unit; the previous meal is unchanged.",
        )
    matched, nutrients, failed = collect_nutrition_for_foods(foods, "meal_edit")
    if failed or len(matched) != len(foods):
        return edit_reply(
            state,
            "meal_edit_failed",
            language,
            "تعذر جلب بيانات كل مكونات التعديل؛ الوجبة السابقة لم تتغير.",
            "Could not retrieve every edited ingredient; the previous meal is unchanged.",
        )
    calculated = calculate_meal_nutrition(matched, nutrients, intent="meal_edit")
    meal = calculated.get("data") or {}
    if (
        not calculated["success"]
        or not meal.get("success")
        or meal.get("unmatched_foods")
    ):
        return edit_reply(
            state,
            "meal_edit_failed",
            language,
            "تعذر حساب التعديل بالكامل؛ الوجبة السابقة لم تتغير.",
            "Could not calculate the full edit; the previous meal is unchanged.",
        )
    minimum = selected["min_protein_g"]
    report = validate_meal_nutrition_report(
        meal, profile, target_calories=target, min_protein_g=minimum
    )
    answer = format_meal_edit_response(meal, report, language)
    if not report["valid"]:
        state.status = "meal_edit_rejected"
        return answer
    snapshot = build_meal_snapshot(
        {
            "success": True,
            "valid": True,
            "foods": foods,
            "meal_result": meal,
            "report": report,
        },
        target,
        selected["meal_type"],
        minimum,
    )
    snapshot["meal_id"] = selected["meal_id"]
    state.meals = [
        snapshot if old["meal_id"] == selected["meal_id"] else deepcopy(old)
        for old in context
    ]
    state.status = "ready"
    return answer
