"""Selective onboarding and profile merging for a conversation."""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.ai.contracts.profile import MANDATORY_PROFILE_FIELDS
from app.ai.nlp.normalization import _extract_gender_from_text

MANDATORY_FIELDS: List[str] = list(MANDATORY_PROFILE_FIELDS)


PROFILE_DEPENDENT_INTENTS = {"daily_plan", "update_profile"}


_FIELD_PROMPTS: dict = {
    "birth_date": {
        "ar": "📅 ما هو تاريخ ميلادك؟ (بصيغة YYYY-MM-DD مثلاً: 2000-03-15)",
        "en": "📅 What is your date of birth? (format: YYYY-MM-DD, e.g. 2000-03-15)",
    },
    "gender": {
        "ar": "⚧  ما هو جنسك؟ (ذكر = male / أنثى = female)",
        "en": "⚧  What is your gender? (male / female)",
    },
    "weight": {
        "ar": "⚖️  كم وزنك بالكيلوجرام؟",
        "en": "⚖️  What is your weight in kilograms?",
    },
    "height": {
        "ar": "📏  كم طولك بالسنتيمتر؟",
        "en": "📏  What is your height in centimeters?",
    },
    "activity_level": {
        "ar": (
            "🏃  ما هو مستوى نشاطك البدني؟\n"
            "  • sedentary  — لا تمارس رياضة\n"
            "  • light      — ١-٣ أيام في الأسبوع\n"
            "  • moderate   — ٣-٥ أيام في الأسبوع\n"
            "  • active     — ٦-٧ أيام في الأسبوع\n"
            "  • very_active — تدريب مكثف يومياً"
        ),
        "en": (
            "🏃  What is your activity level?\n"
            "  • sedentary   — no exercise\n"
            "  • light       — 1-3 days/week\n"
            "  • moderate    — 3-5 days/week\n"
            "  • active      — 6-7 days/week\n"
            "  • very_active — intense daily training"
        ),
    },
    "goal": {
        "ar": (
            "🎯  ما هو هدفك؟\n"
            "  • weight_loss  — إنقاص الوزن\n"
            "  • maintenance  — الحفاظ على الوزن\n"
            "  • weight_gain  — زيادة الوزن\n"
            "  • muscle_gain  — بناء العضلات"
        ),
        "en": (
            "🎯  What is your goal?\n"
            "  • weight_loss\n"
            "  • maintenance\n"
            "  • weight_gain\n"
            "  • muscle_gain"
        ),
    },
}


def get_onboarding_prompt(missing_fields: List[str], lang: str = "english") -> str:
    """
    Build a bilingual conversational prompt asking for missing mandatory fields.

    Args:
        missing_fields: list of field names that are still None.
        lang: "arabic" | "english" | "mixed"  (from detect_language)

    Returns:
        A formatted string to show the user.
    """
    key = "ar" if lang == "arabic" else "en"
    if lang == "arabic":
        header = "مرحباً! 👋 قبل ما نبدأ، محتاج بعض المعلومات الأساسية عشان أحسب السعرات الحرارية بدقة:\n\n"
    else:
        header = "👋 Welcome! Before we start, I need a few details to calculate your calories accurately:\n\n"

    lines = [_FIELD_PROMPTS[f][key] for f in missing_fields if f in _FIELD_PROMPTS]
    return header + "\n\n".join(lines)


def validate_mandatory_profile_fields(profile: dict) -> Tuple[bool, List[str]]:
    """
    Check whether all 6 mandatory Task-3 fields are non-null.

    Returns:
        (is_complete: bool, missing_fields: List[str])
    """
    if not profile:
        return False, MANDATORY_FIELDS[:]
    missing = [k for k in MANDATORY_FIELDS if profile.get(k) is None]
    return (len(missing) == 0), missing


def merge_profile(
    existing: dict, new_data: dict, raw_text: Optional[str] = None
) -> dict:
    """
    Merge a newly parsed profile_data into a stored profile.
    - Scalar fields: only overwrite if new value is non-null.
    - If gender is already set, do not overwrite unless raw_text explicitly has a gender keyword.
    - List fields: union (no duplicates).
    """
    merged = existing.copy()
    list_fields = {
        "dietary_restrictions",
        "forbidden_foods",
        "preferred_foods",
        "allergies",
    }
    for key, value in new_data.items():
        if key in list_fields:
            if value:  # non-empty list → union
                merged[key] = list(set(merged.get(key, []) + value))
        elif key == "gender":
            if existing.get("gender") is None and value is not None:
                merged["gender"] = value
            elif raw_text and _extract_gender_from_text(raw_text) is not None:
                merged["gender"] = _extract_gender_from_text(raw_text)
        else:
            if value is not None:
                merged[key] = value
    return merged
