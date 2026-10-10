"""On-demand USDA HTTP search and nutrient detail extraction."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from app.ai.retrieval.cache import DETAILS_CACHE, SEARCH_CACHE
from app.ai.retrieval.http import invalid_response, request_json

logger = logging.getLogger(__name__)


def search_food_usda(food_name, max_results=50):
    """Return search candidates; propagate USDA operational failures."""
    if not food_name or not food_name.strip():
        return []

    cache_key = (food_name.strip().lower(), max_results)
    if cache_key in SEARCH_CACHE:
        logger.debug("Search cache hit: %s", food_name)
        return SEARCH_CACHE[cache_key]

    data = request_json(
        "foods/search", params={"query": food_name, "pageSize": max_results}
    )
    if not isinstance(data.get("foods"), list):
        raise invalid_response()

    results = []
    for food in data["foods"]:
        _validate_food(food)
        results.append(
            {
                "fdc_id": food["fdcId"],
                "description": food["description"],
                "data_type": food.get("dataType"),
            }
        )

    SEARCH_CACHE[cache_key] = results
    return results


def _validate_food(data: object, expected_id: int | None = None) -> None:
    """Reject malformed records before caching or numerical calculation."""
    if not isinstance(data, dict):
        raise invalid_response()
    fdc_id = data.get("fdcId")
    description = data.get("description")
    if (
        type(fdc_id) is not int
        or fdc_id <= 0
        or (expected_id is not None and fdc_id != expected_id)
        or not isinstance(description, str)
        or not description.strip()
    ):
        raise invalid_response()


_USDA_NUTRIENT_IDS = {
    "calories": (1008, 2047, 2048),
    "protein": (1003,),
    "fat": (1004,),
    "carbs": (1005,),
}


def _first_present(values: Dict[int, float], ids: Tuple[int, ...]) -> Optional[float]:
    """First non-None value (fixes the `x or y` bug when x == 0)."""
    for nid in ids:
        v = values.get(nid)
        if v is not None:
            return v
    return None


def _extract_portions(data: dict) -> List[dict]:
    """USDA foodPortions → [{amount, unit, modifier, gram_weight}]."""
    portions: List[dict] = []
    for p in data.get("foodPortions") or []:
        unit_obj = p.get("measureUnit") or {}
        unit_name = (
            (unit_obj.get("name") or unit_obj.get("abbreviation") or "").strip().lower()
        )
        gw = p.get("gramWeight")
        amt = p.get("amount")
        if not gw or not amt:
            continue
        portions.append(
            {
                "amount": float(amt),
                "unit": unit_name,
                "modifier": (p.get("modifier") or "").strip().lower() or None,
                "gram_weight": float(gw),
            }
        )
    return portions


def get_food_details(fdc_id: int) -> Optional[dict]:
    """Return per-100g nutrients; a missing USDA record returns None."""
    if fdc_id in DETAILS_CACHE:
        logger.debug("Details cache hit: %s", fdc_id)
        return DETAILS_CACHE[fdc_id]

    data = request_json(f"food/{fdc_id}", allow_not_found=True)
    if data is None:
        return None
    _validate_food(data, expected_id=fdc_id)

    items = data.get("foodNutrients", [])
    if not isinstance(items, list):
        raise invalid_response()

    nutrient_values: Dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict):
            raise invalid_response()
        nutrient = item.get("nutrient") or {}
        if not isinstance(nutrient, dict):
            raise invalid_response()
        nid = nutrient.get("id")
        amount = item.get("amount")
        if nid is None or amount is None:
            continue
        if (
            type(nid) is not int
            or type(amount) not in (int, float)
            or not math.isfinite(amount)
            or amount < 0
        ):
            raise invalid_response()
        nutrient_values[nid] = amount

    try:
        portions = _extract_portions(data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise invalid_response() from exc

    result = {
        "fdc_id": data["fdcId"],
        "food_name": data["description"],
        "calories_per_100g": _first_present(
            nutrient_values, _USDA_NUTRIENT_IDS["calories"]
        ),
        "protein_per_100g": _first_present(
            nutrient_values, _USDA_NUTRIENT_IDS["protein"]
        ),
        "carbs_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["carbs"]),
        "fat_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["fat"]),
        "portions": portions,
    }

    DETAILS_CACHE[fdc_id] = result
    return result


def nutrition_is_complete(details) -> bool:
    if not details:
        return False
    required = (
        details.get("calories_per_100g"),
        details.get("protein_per_100g"),
        details.get("carbs_per_100g"),
        details.get("fat_per_100g"),
    )
    return all(v is not None for v in required)
