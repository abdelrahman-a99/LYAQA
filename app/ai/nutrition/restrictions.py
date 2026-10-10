"""Dietary restrictions and food alias matching."""

from __future__ import annotations

import re
from typing import Dict

from app.ai.common.text import normalize_arabic_text

_RESTRICTION_SYNONYMS: Dict[str, set] = {
    "eggs": {
        "بيض",
        "بيضة",
        "بيضه",
        "بيض مسلوق",
        "egg",
        "eggs",
        "boiled egg",
        "boiled eggs",
    },
    "dairy": {
        "ألبان",
        "البان",
        "حليب",
        "لبن",
        "زبادي",
        "جبنة",
        "جبن",
        "قشطة",
        "كريمة",
        "dairy",
        "milk",
        "yogurt",
        "yoghurt",
        "cheese",
        "cream",
        "butter",
    },
    "nuts": {
        "مكسرات",
        "فول سوداني",
        "لوز",
        "كاجو",
        "بندق",
        "بستم",
        "فستق",
        "nuts",
        "nut",
        "peanut",
        "peanuts",
        "almond",
        "almonds",
        "cashew",
        "hazelnut",
        "pistachio",
        "walnut",
    },
    "fish": {
        "سمك",
        "سلمون",
        "تونة",
        "سردين",
        "ماكريل",
        "fish",
        "salmon",
        "tuna",
        "sardine",
        "mackerel",
    },
    "shellfish": {
        "جمبري",
        "جمبري",
        "كابوريا",
        "استاكوزا",
        "محار",
        "shellfish",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "oyster",
    },
    "gluten": {
        "جلوتين",
        "قمح",
        "خبز",
        "عيش",
        "مكرونة",
        "معكرونة",
        "برغل",
        "كسكس",
        "gluten",
        "wheat",
        "bread",
        "pasta",
        "couscous",
        "bulgur",
    },
    "soy": {
        "صويا",
        "توفو",
        "ادامامي",
        "soy",
        "soya",
        "tofu",
        "edamame",
    },
    "chicken": {
        "دجاج",
        "فراخ",
        "فرخة",
        "chicken",
    },
    "meat": {
        "لحمة",
        "لحم",
        "لحم بقري",
        "ضاني",
        "ضأن",
        "كندوز",
        "meat",
        "beef",
        "lamb",
        "veal",
        "steak",
    },
    "pork": {
        "خنزير",
        "لحم خنزير",
        "بيكون",
        "هام",
        "pork",
        "bacon",
        "ham",
        "prosciutto",
    },
    "honey": {
        "عسل",
        "honey",
    },
}


_DIET_PATTERN_RULES: Dict[str, set] = {
    "vegan": {"eggs", "dairy", "meat", "chicken", "fish", "shellfish", "honey"},
    "vegetarian": {"meat", "chicken", "fish", "shellfish"},
    "pescatarian": {"meat", "chicken"},
    "halal": {"pork"},
    "keto": set(),
}


def _normalize_food_text(text: str) -> str:
    """Normalize Arabic and punctuation for restriction matching."""
    if not text:
        return ""

    text = normalize_arabic_text(text.lower().strip())
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _restriction_keys(text: str) -> set:
    """
    Return the canonical restriction keys a given text matches.

    Single-word aliases are matched as whole tokens (so "أرز أبيض" does NOT
    match the "بيض" alias). Multi-word aliases are matched as substrings.
    """
    normalized = _normalize_food_text(text)
    if not normalized:
        return set()

    tokens = set(normalized.split())
    matched = set()

    for canonical, aliases in _RESTRICTION_SYNONYMS.items():
        for alias in aliases:
            alias_norm = _normalize_food_text(alias)
            if not alias_norm:
                continue
            if " " in alias_norm:
                if alias_norm in normalized:
                    matched.add(canonical)
                    break
            elif alias_norm in tokens:
                matched.add(canonical)
                break

    return matched


def _collect_user_restrictions(profile: dict) -> set:
    """
    Combine forbidden_foods + allergies + dietary_restrictions into a single
    set of canonical restriction keys.
    """
    keys = set()

    for name in profile.get("forbidden_foods", []) or []:
        keys |= _restriction_keys(name)
    for name in profile.get("allergies", []) or []:
        keys |= _restriction_keys(name)

    for pattern in profile.get("dietary_restrictions", []) or []:
        pattern_key = _normalize_food_text(pattern)
        keys |= _DIET_PATTERN_RULES.get(pattern_key, set())

    return keys


def _check_food_against_restrictions(
    food: dict,
    user_restrictions: set,
) -> set:
    """
    Check a single food item against the user's restriction set.
    Looks at food_name_original, food_name (USDA name), and food_name_en.
    """
    if not user_restrictions:
        return set()

    food_keys = _restriction_keys(food.get("food_name_original", ""))
    food_keys |= _restriction_keys(food.get("food_name", ""))
    food_keys |= _restriction_keys(food.get("food_name_en", ""))

    return food_keys & user_restrictions
