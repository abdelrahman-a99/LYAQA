"""Process-local USDA and translation caches."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

SEARCH_CACHE: Dict[Tuple[str, int], List[dict]] = {}
DETAILS_CACHE: Dict[int, dict] = {}
TRANSLATION_CACHE: Dict[str, str] = {}


def clear_caches() -> None:
    """Explicit cache reset — call only from tests or a /reset endpoint."""
    SEARCH_CACHE.clear()
    DETAILS_CACHE.clear()
    TRANSLATION_CACHE.clear()
    logger.info("Caches cleared.")
