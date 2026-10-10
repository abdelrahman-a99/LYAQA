"""Validate and revise a candidate meal within the revision budget."""

from typing import List, Optional

from app.ai.agent.food_lookup import collect_nutrition_for_foods
from app.ai.agent.llm import (
    call_food_substitute,
    call_full_meal_revision,
)
from app.ai.agent.state import AgentState
from app.ai.agent.tools import (
    calculate_meal_nutrition,
    scale_meal,
    validate_plan,
)

MAX_REVISIONS = 2  # spec decision #9


def run_revision_loop(
    foods: List[dict],
    nutrition_list: List[dict],
    profile: dict,
    target_calories: float,
    min_protein_g: Optional[float],
    language: str,
    meal_type: Optional[str],
    state: AgentState,
    intent: str,
) -> tuple:
    """
    Run the revision loop (spec section 5) for a single meal.

    Returns (final_meal_result, final_foods, final_nutrition, valid_flag).
    Appends each attempt to state.revision_history with the full validation shape.
    """
    current_foods = foods
    current_nutrition = nutrition_list
    _scale_done = False  # scale_meal_to_target runs at most once (uncounted)

    meal_result_wrap = calculate_meal_nutrition(
        current_foods, current_nutrition, intent=intent
    )
    meal_result = meal_result_wrap["data"] if meal_result_wrap["success"] else {}

    attempt = 0
    while True:
        attempt += 1
        validation = validate_plan(
            meal_result,
            profile,
            mode="meal",
            target_calories=target_calories,
            min_protein_g=min_protein_g,
            intent=intent,
        )
        val_data = validation.get("data", {}) if validation["success"] else {}
        valid = val_data.get("valid", False)
        checks = val_data.get("checks", {})
        violations = val_data.get("restriction_violations", [])

        state.revision_history.append(
            {
                "attempt_number": attempt,
                "meal": meal_result,
                "validation_result": val_data,
                "action_taken": None,  # filled in below
            }
        )

        if valid:
            state.revision_history[-1]["action_taken"] = "accepted"
            return meal_result, current_foods, current_nutrition, True

        if state.revision_count >= MAX_REVISIONS:
            state.revision_history[-1]["action_taken"] = "max_revisions_exhausted"
            return meal_result, current_foods, current_nutrition, False

        cal_ok = checks.get("calories", True)
        prot_ok = checks.get("protein", True)
        rest_ok = checks.get("restrictions", True)

        if not rest_ok:
            action = "llm_substitute_forbidden"
            new_foods = list(current_foods)
            for viol in violations:
                # Task 3 returns dicts, not strings
                viol_name = (
                    viol.get("food_name_original") or viol.get("food_name") or ""
                )
                if not viol_name:
                    continue
                sub = call_food_substitute(viol_name, profile, language, current_foods)
                if not sub:
                    continue
                # Match by either original or English name
                replaced = False
                viol_name_lower = viol_name.lower()
                for i, f in enumerate(new_foods):
                    f_orig = (f.get("food_name_original") or "").lower()
                    f_en = (f.get("food_name_en") or "").lower()
                    if viol_name_lower in (f_orig, f_en):
                        new_foods[i] = sub
                        replaced = True
                        break
                if not replaced:
                    new_foods.append(sub)

            # Apply the substitutions: re-fetch nutrition and recompute the meal
            new_foods_m, new_nutrition_m, _ = collect_nutrition_for_foods(
                new_foods, intent
            )
            if new_foods_m:
                current_foods = new_foods_m
                current_nutrition = new_nutrition_m
                meal_result_wrap = calculate_meal_nutrition(
                    current_foods, current_nutrition, intent=intent
                )
                meal_result = (
                    meal_result_wrap["data"]
                    if meal_result_wrap["success"]
                    else meal_result
                )
            state.revision_count += 1

        elif not cal_ok and prot_ok and rest_ok and not _scale_done:
            # Calorie-only failure: try scale_meal_to_target once (uncounted)
            _scale_done = True
            action = "scale_then_revalidate"
            current_calories = meal_result.get("total_calories", 0.0)
            scale_result = scale_meal(
                current_foods, current_calories, target_calories, intent=intent
            )
            scaled_data = scale_result.get("data", {})
            scaled_foods = scaled_data.get("scaled_foods", current_foods)

            new_foods_m, new_nutrition_m, _ = collect_nutrition_for_foods(
                scaled_foods, intent
            )
            if new_foods_m:
                current_foods = new_foods_m
                current_nutrition = new_nutrition_m
                meal_result_wrap = calculate_meal_nutrition(
                    current_foods, current_nutrition, intent=intent
                )
                meal_result = (
                    meal_result_wrap["data"]
                    if meal_result_wrap["success"]
                    else meal_result
                )

            # Re-validate immediately (still within the uncounted scale step)
            re_val = validate_plan(
                meal_result,
                profile,
                mode="meal",
                target_calories=target_calories,
                min_protein_g=min_protein_g,
                intent=intent,
            )
            re_data = re_val.get("data", {}) if re_val["success"] else {}
            if re_data.get("valid", False):
                state.revision_history[-1]["action_taken"] = action + "_valid"
                return meal_result, current_foods, current_nutrition, True

            # Scale did not fix it — fall through to LLM revision (now counted)
            action = action + "_then_llm"
            issues = _build_issue_list(
                re_data.get("checks", {}), re_data.get("restriction_violations", [])
            )
            revised = call_full_meal_revision(
                current_foods, target_calories, profile, meal_type, language, issues
            )
            if revised:
                new_foods_m, new_nutrition_m, _ = collect_nutrition_for_foods(
                    revised, intent
                )
                if new_foods_m:
                    current_foods = new_foods_m
                    current_nutrition = new_nutrition_m
                    meal_result_wrap = calculate_meal_nutrition(
                        current_foods, current_nutrition, intent=intent
                    )
                    meal_result = (
                        meal_result_wrap["data"]
                        if meal_result_wrap["success"]
                        else meal_result
                    )
            state.revision_count += 1

        else:
            # Protein failure, or calorie+protein, or any other combination
            action = "llm_full_revision"
            issues = _build_issue_list(checks, violations)
            revised = call_full_meal_revision(
                current_foods, target_calories, profile, meal_type, language, issues
            )
            if revised:
                new_foods_m, new_nutrition_m, _ = collect_nutrition_for_foods(
                    revised, intent
                )
                if new_foods_m:
                    current_foods = new_foods_m
                    current_nutrition = new_nutrition_m
                    meal_result_wrap = calculate_meal_nutrition(
                        current_foods, current_nutrition, intent=intent
                    )
                    meal_result = (
                        meal_result_wrap["data"]
                        if meal_result_wrap["success"]
                        else meal_result
                    )
            state.revision_count += 1

        state.revision_history[-1]["action_taken"] = action


def _build_issue_list(checks: dict, violations: list) -> List[str]:
    issues = []
    if not checks.get("calories", True):
        issues.append("Calorie total is outside the target range.")
    if not checks.get("protein", True):
        issues.append("Protein is below the minimum requirement.")
    if not checks.get("restrictions", True) and violations:
        names = [
            v.get("food_name_original") or v.get("food_name")
            for v in violations
            if isinstance(v, dict)
        ]
        names = [n for n in names if n]
        if names:
            issues.append(f"Contains forbidden foods: {', '.join(names)}.")
    return issues or ["Nutritional targets not met."]
