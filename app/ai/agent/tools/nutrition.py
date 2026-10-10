import time
from typing import List, Optional

from app.ai.agent.tools.common import _envelope, _log
from app.ai.nutrition import (
    calculate_bmr,
    calculate_macros,
    calculate_meal_total,
    calculate_target_calories,
    calculate_tdee,
    scale_meal_to_target,
    validate_daily_nutrition,
    validate_meal_nutrition,
)


def calculate_targets(profile: dict, intent: str = "") -> dict:
    """
    Compute bmr, tdee, target_calories, and macros in one call.

    All four Task 3 calls are inside a single try/except so any
    ValueError (missing profile field) collapses to error_code="incomplete_profile"
    rather than crashing the orchestrator.
    """
    t0 = time.perf_counter()
    try:
        bmr = calculate_bmr(profile)
        tdee = calculate_tdee(profile)
        target_cal = calculate_target_calories(profile)
        macros = calculate_macros(profile)
        duration = (time.perf_counter() - t0) * 1000
        _log("calculate_targets", intent, {"goal": profile.get("goal")}, True, duration)
        return _envelope(
            True,
            {
                "bmr": bmr,
                "tdee": tdee,
                "target_calories": target_cal,
                "macros": macros,
            },
        )
    except ValueError as exc:
        duration = (time.perf_counter() - t0) * 1000
        _log(
            "calculate_targets", intent, {"goal": profile.get("goal")}, False, duration
        )
        return _envelope(False, error=str(exc), error_code="incomplete_profile")


def calculate_meal_nutrition(
    foods: List[dict], nutrition_list: List[dict], intent: str = ""
) -> dict:
    """
    Thin wrapper around Task 3's calculate_meal_total.

    Validates that both lists are the same length before calling,
    because Task 3 raises ValueError on a length mismatch.
    """
    t0 = time.perf_counter()
    if len(foods) != len(nutrition_list):
        duration = (time.perf_counter() - t0) * 1000
        msg = (
            f"foods length ({len(foods)}) does not match "
            f"nutrition_list length ({len(nutrition_list)})"
        )
        _log(
            "calculate_meal_nutrition", intent, {"n_foods": len(foods)}, False, duration
        )
        return _envelope(False, error=msg, error_code="list_length_mismatch")

    result = calculate_meal_total(foods, nutrition_list)
    duration = (time.perf_counter() - t0) * 1000
    _log(
        "calculate_meal_nutrition",
        intent,
        {"n_foods": len(foods), "success": result.get("success")},
        True,
        duration,
    )
    return _envelope(True, result)


def validate_plan(
    meal_or_day,
    profile: dict,
    mode: str = "meal",
    target_calories: Optional[float] = None,
    min_protein_g: Optional[float] = None,
    calorie_tolerance: Optional[float] = None,
    intent: str = "",
) -> dict:
    """
    Validate a single meal or a full day using the FULL validator functions,
    not the _report wrappers.  The _report wrappers are used only when composing
    the final user-facing answer (in orchestrator.py) — never here.

    mode="meal"  → validate_meal_nutrition (accepts partial profile)
    mode="day"   → validate_daily_nutrition (requires full profile)

    data carries the complete validator output including checks{calories, protein,
    restrictions} and restriction_violations, which the revision loop branches on.
    """
    t0 = time.perf_counter()
    try:
        if mode == "meal":
            kwargs = {}
            if target_calories is not None:
                kwargs["target_calories"] = target_calories
            if min_protein_g is not None:
                kwargs["min_protein_g"] = min_protein_g
            if calorie_tolerance is not None:
                kwargs["calorie_tolerance"] = calorie_tolerance
            result = validate_meal_nutrition(meal_or_day, profile, **kwargs)
        else:
            kwargs = {}
            if calorie_tolerance is not None:
                kwargs["calorie_tolerance"] = calorie_tolerance
            result = validate_daily_nutrition(meal_or_day, profile, **kwargs)

        duration = (time.perf_counter() - t0) * 1000
        _log(
            "validate_plan",
            intent,
            {"mode": mode, "valid": result.get("valid")},
            True,
            duration,
        )
        return _envelope(True, result)
    except ValueError as exc:
        duration = (time.perf_counter() - t0) * 1000
        _log("validate_plan", intent, {"mode": mode}, False, duration)
        return _envelope(False, error=str(exc), error_code="incomplete_profile")


def scale_meal(
    foods: List[dict], current_calories: float, target_calories: float, intent: str = ""
) -> dict:
    """
    Thin wrapper around Task 3's scale_meal_to_target.
    Used only inside the revision loop — exactly one call per calorie-only failure.
    """
    t0 = time.perf_counter()
    result = scale_meal_to_target(foods, current_calories, target_calories)
    duration = (time.perf_counter() - t0) * 1000
    _log(
        "scale_meal",
        intent,
        {
            "current_calories": current_calories,
            "target_calories": target_calories,
            "scale_factor": result.get("scale_factor"),
        },
        True,
        duration,
    )
    return _envelope(True, result)
