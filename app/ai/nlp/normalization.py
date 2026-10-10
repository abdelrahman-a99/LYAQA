"""Normalize parser outputs and sanitize extracted profile values."""

from __future__ import annotations

import re
from typing import List, Optional

from app.ai.nlp.language import detect_language

UNIT_ALIASES: dict = {
    # English plurals / abbreviations
    "kg": "kilogram",
    "kilograms": "kilogram",
    "kgs": "kilogram",
    "grams": "gram",
    "g": "gram",
    "cups": "cup",
    "pieces": "piece",
    "slices": "slice",
    "tablespoons": "tablespoon",
    "tbsp": "tablespoon",
    "teaspoons": "teaspoon",
    "tsp": "teaspoon",
    "bowls": "bowl",
    "cans": "can",
    "bottles": "bottle",
    "loaves": "loaf",
    "ml": "milliliter",
    "mls": "milliliter",
    "liters": "liter",
    "l": "liter",
    # Arabic units
    "غم": "gram",
    "غرام": "gram",
    "جرام": "gram",
    "كوب": "cup",
    "أكواب": "cup",
    "قطعة": "piece",
    "قطع": "piece",
    "شريحة": "slice",
    "شرايح": "slice",
    "ملعقة كبيرة": "tablespoon",
    "ملعقة صغيرة": "teaspoon",
    "طبق": "bowl",
    "أطباق": "bowl",
    "علبة": "can",
    "زجاجة": "bottle",
    "رغيف": "loaf",
}


_GENDER_FEMALE_KEYWORDS: List[str] = [
    # Arabic
    "أنثى",
    "انثى",
    "بنت",
    "امرأة",
    "فتاة",
    "ست",
    "أنا بنت",
    "أنا أنثى",
    # English
    "female",
    "woman",
    "girl",
]


_GENDER_MALE_KEYWORDS: List[str] = [
    # Arabic
    "ذكر",
    "ولد",
    "راجل",
    "رجل",
    "صبي",
    "أنا ذكر",
    "أنا ولد",
    # English whole words (checked with surrounding whitespace)
    "male",
    "man",
    "boy",
]


def _extract_gender_from_text(text: str) -> Optional[str]:
    """
    Deterministic gender extraction from Arabic/English keywords.

    Uses TOKEN matching only — no substring matching — so "مولود" does NOT
    match "ولد" and "بنطلون" does NOT match "بنت".
    """
    cleaned = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", text.lower())
    tokens = set(cleaned.split())
    padded = f" {cleaned} "

    for kw in _GENDER_FEMALE_KEYWORDS:
        kw_norm = kw.strip().lower()
        if " " in kw_norm:
            if kw_norm in padded:
                return "female"
        elif kw_norm in tokens:
            return "female"

    for kw in _GENDER_MALE_KEYWORDS:
        kw_norm = kw.strip().lower()
        if " " in kw_norm:
            if kw_norm in padded:
                return "male"
        elif kw_norm in tokens:
            return "male"

    return None


def sanitize_extracted_values(parsed: dict) -> dict:
    """
    Sanity checks for biometric & metric values:
    - Height: fix mm (1750 → 175), clamp 50–260 cm
    - Weight: clamp 20–350 kg
    - Target calories: clamp 50–15000 kcal
    Anything out of range is set to None.
    """
    profile = parsed.get("profile_data")
    if profile:
        if profile.get("height") is not None:
            try:
                h = float(profile["height"])
                if h >= 500:
                    h = h / 10.0
                profile["height"] = round(h, 1) if 50.0 <= h <= 260.0 else None
            except (ValueError, TypeError):
                profile["height"] = None

        if profile.get("weight") is not None:
            try:
                w = float(profile["weight"])
                profile["weight"] = round(w, 1) if 20.0 <= w <= 350.0 else None
            except (ValueError, TypeError):
                profile["weight"] = None

    if parsed.get("target_calories") is not None:
        try:
            tc = float(parsed["target_calories"])
            parsed["target_calories"] = round(tc, 1) if 50.0 <= tc <= 15000.0 else None
        except (ValueError, TypeError):
            parsed["target_calories"] = None

    return parsed


NON_FOOD_STOP_WORDS = {
    "before workout",
    "after workout",
    "without eggs",
    "breakfast meal",
    "lunch meal",
    "dinner meal",
    "snack",
    "item",
    "meal",
    "substitute",
    "food",
    "diet",
    "وجبة فطور",
    "وجبة غداء",
    "وجبة عشاء",
    "بدون بيض",
    "قبل التمرين",
    "بعد التمرين",
}


EMPTY_FOODS_INTENTS = {
    "meal_recommendation",
    "fitness_nutrition_question",
    "daily_plan",
    "food_substitution",
    "update_profile",
    "general_chat",
}


def normalize_output(parsed: dict) -> dict:
    """
    Post-process LLM output:
    - Force foods=[] for non-logging intents.
    - Remove non-food artifacts and duplicates.
    - Normalize units + lowercase food_name_en.
    - Fix food_name_original if it hallucinated Arabic on English input.
    - Deterministic gender fallback.
    - Sanity checks on biometric & metric values.
    """
    intent = parsed.get("intent")
    raw_text = parsed.get("raw_text", "")
    lang = detect_language(raw_text)

    # ── Foods: force empty for non-logging intents ──────────────────────────
    if intent in EMPTY_FOODS_INTENTS:
        parsed["foods"] = []
    else:
        seen: set = set()
        clean_foods: list = []
        for food in parsed.get("foods", []):
            name_en = food.get("food_name_en", "").strip().lower()
            name_orig = food.get("food_name_original", "").strip()

            # Filter non-food artifacts
            if (
                name_en in NON_FOOD_STOP_WORDS
                or name_orig.lower() in NON_FOOD_STOP_WORDS
            ):
                continue

            # If input is English, don't keep an Arabic hallucination in original
            if lang == "english" and re.search(r"[\u0600-\u06FF]", name_orig):
                name_orig = name_en
            food["food_name_original"] = name_orig

            raw_unit = str(food.get("unit", "")).strip().lower()
            food["unit"] = UNIT_ALIASES.get(raw_unit, raw_unit)
            if food.get("quantity", 0) <= 0:
                food["quantity"] = 1.0
            food["food_name_en"] = name_en

            key = (food["food_name_en"], food["quantity"], food["unit"])
            if key not in seen and name_en:
                seen.add(key)
                clean_foods.append(food)
        parsed["foods"] = clean_foods

    # ── Deterministic gender fallback ───────────────────────────────────────
    profile = parsed.get("profile_data")
    if profile and profile.get("gender") is None:
        detected_gender = _extract_gender_from_text(raw_text)
        if detected_gender:
            profile["gender"] = detected_gender

    # ── Sanity checks ──────────────────────────────────────────────────────
    return sanitize_extracted_values(parsed)
