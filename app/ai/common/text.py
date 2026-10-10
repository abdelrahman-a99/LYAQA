"""Shared Arabic Unicode normalization; domain punctuation rules remain separate."""

from __future__ import annotations

import re

_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_ALEF_RE = re.compile(r"[إأآٱ]")


def normalize_arabic_text(text: str) -> str:
    """
    Deterministic Arabic normalization applied to the WORKING COPY sent
    to the LLM for parsing. Does NOT touch `raw_text` in the final output
    — raw_text must stay a verbatim copy of what the user typed (Rule #2).

    Mirrors the rules used in Task 2's `_normalize_lookup_key`, so both
    modules treat Arabic text the same way and don't drift apart:
    - Strips tashkeel (diacritics).
    - Strips tatweel (ـ).
    - Normalizes أ / إ / آ / ٱ → ا
    - Normalizes ى → ي
    - Collapses repeated whitespace.
    """
    if not text:
        return text

    normalized = _ARABIC_DIACRITICS_RE.sub("", text)
    normalized = normalized.replace("ـ", "")
    normalized = _ARABIC_ALEF_RE.sub("ا", normalized)
    normalized = normalized.replace("ى", "ي")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized
