"""Deterministic Arabic, English and mixed-language detection."""

from __future__ import annotations

import re
from typing import Literal


def detect_language(text: str) -> Literal["arabic", "english", "mixed"]:
    """
    Classify input language by Unicode character counts (Arabic U+0600–U+06FF).

    Uses a RATIO-based check so long messages with a few borrowed/foreign
    words (e.g. a long Arabic paragraph containing one English brand name)
    aren't misclassified as "mixed".

    Returns 'arabic', 'english', or 'mixed'.
    """
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    total = arabic_chars + latin_chars

    if total == 0:
        return "english"
    if arabic_chars == 0:
        return "english"
    if latin_chars == 0:
        return "arabic"

    # Common short Latin unit abbreviations in Arabic (e.g. "200g", "175cm", "90kg")
    # and the symmetric case (a couple of Arabic words dropped into an
    # otherwise English sentence) shouldn't flip the whole message to "mixed".
    if latin_chars <= 2 and arabic_chars >= 6:
        return "arabic"
    if arabic_chars <= 2 and latin_chars >= 6:
        return "english"

    arabic_ratio = arabic_chars / total
    if arabic_ratio >= 0.85:
        return "arabic"
    if arabic_ratio <= 0.15:
        return "english"

    return "mixed"
