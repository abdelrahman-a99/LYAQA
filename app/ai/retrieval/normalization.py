"""USDA query normalization, dialect mapping and optional translation."""

from __future__ import annotations

import re

from app.ai.common.text import normalize_arabic_text
from app.ai.retrieval.cache import TRANSLATION_CACHE
from app.ai.inference import run_inference


def clean_food_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def contains_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _normalize_lookup_key(text):
    text = clean_food_text(text).lower()

    text = normalize_arabic_text(text)

    text = re.sub(r"[^\w\s-]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


FOOD_QUERY_MAP_RAW = {
    # Arabic / Egyptian
    "فراخ مشوية": "grilled chicken",
    "فراخ مشوي": "grilled chicken",
    "دجاج مشوي": "grilled chicken",
    "دجاج مشوية": "grilled chicken",
    "رز أبيض": "white rice",
    "رز ابيض": "white rice",
    "أرز أبيض": "white rice",
    "ارز ابيض": "white rice",
    "عيش بلدي": "pita bread",
    "خبز بلدي": "pita bread",
    "خبز عربي": "pita bread",
    "بطاطس مسلوقة": "boiled potatoes",
    "بطاطا مسلوقة": "boiled potatoes",
    "بيض": "whole eggs",
    "زبادي": "plain yogurt",
    # English
    "egg": "whole eggs",
    "eggs": "whole eggs",
    "banana": "raw banana",
    "bananas": "raw banana",
    "salmon": "raw salmon",
    "yogurt": "plain yogurt",
    "yoghurt": "plain yogurt",
    "oatmeal": "cooked oatmeal",
    # Common translation outputs
    "arab bread": "pita bread",
    "arabic bread": "pita bread",
    "boiled potato": "boiled potatoes",
    "potato boiled": "boiled potatoes",
    "plain yoghurt": "plain yogurt",
}


FOOD_QUERY_MAP = {_normalize_lookup_key(k): v for k, v in FOOD_QUERY_MAP_RAW.items()}


def apply_food_mapping(text):
    return FOOD_QUERY_MAP.get(_normalize_lookup_key(text))


DIALECT_ALIASES = {
    "عيش بلدي": "خبز عربي",
    "فراخ": "دجاج",
    "رز": "أرز",
    "بطاطس": "بطاطا",
}


def normalize_dialect(text):
    for source, target in DIALECT_ALIASES.items():
        text = text.replace(source, target)
    return text


def translate_arabic_to_english(text: str) -> str:
    if not text:
        return ""

    cache_key = text.strip()
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    translation = run_inference("translate", {"text": text})
    translation = clean_food_text(translation).lower()

    TRANSLATION_CACHE[cache_key] = translation
    return translation


def normalize_food_query(food_text):
    """Returns {original_food, normalized_food, method}."""
    original_text = food_text
    cleaned_text = clean_food_text(food_text)

    if not cleaned_text:
        return {
            "original_food": original_text,
            "normalized_food": "",
            "method": "empty",
        }

    # 1) Direct mapping
    mapped_food = apply_food_mapping(cleaned_text)
    if mapped_food:
        english_food = mapped_food
        method = "direct_food_mapping"

    # 2) Arabic path
    elif contains_arabic(cleaned_text):
        dialect_normalized = normalize_dialect(cleaned_text)
        mapped_after_dialect = apply_food_mapping(dialect_normalized)

        if mapped_after_dialect:
            english_food = mapped_after_dialect
            method = "dialect_normalization+food_mapping"
        else:
            english_food = translate_arabic_to_english(dialect_normalized)
            method = "dialect_normalization+transformer_translation"

    # 3) English path
    else:
        english_food = cleaned_text.lower()
        method = "already_english"

    english_food = re.sub(r"[^\w\s-]", " ", english_food)
    english_food = re.sub(r"\s+", " ", english_food).strip().lower()

    post_mapping = apply_food_mapping(english_food)
    if post_mapping and post_mapping != english_food:
        english_food = post_mapping
        method = method + "+post_mapping"

    return {
        "original_food": original_text,
        "normalized_food": english_food,
        "method": method,
    }
