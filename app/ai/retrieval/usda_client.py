"""On-demand USDA HTTP search and nutrient detail extraction."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import requests

from app.ai.retrieval.cache import DETAILS_CACHE, SEARCH_CACHE
from app.ai.retrieval.config import get_api_key, get_base_url

logger = logging.getLogger(__name__)


def search_food_usda(food_name, max_results=50):
    """USDA search. Returns [{fdc_id, description, data_type}]."""
    if not food_name:
        return []

    cache_key = (food_name.strip().lower(), max_results)
    if cache_key in SEARCH_CACHE:
        logger.debug("Search cache hit: %s", food_name)
        return SEARCH_CACHE[cache_key]

    url = f"{get_base_url()}/foods/search"
    params = {"api_key": get_api_key(), "query": food_name, "pageSize": max_results}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 403:
            raise RuntimeError("Invalid USDA API key.")
        if response.status_code == 429:
            raise RuntimeError("USDA API rate limit exceeded.")

        response.raise_for_status()
        data = response.json()

        results = []
        for food in data.get("foods", []):
            description = food.get("description")
            if not description:
                continue
            results.append(
                {
                    "fdc_id": food.get("fdcId"),
                    "description": description,
                    "data_type": food.get("dataType"),
                }
            )

        SEARCH_CACHE[cache_key] = results
        return results

    except requests.exceptions.Timeout:
        logger.warning("USDA search timed out for query=%r", food_name)
        return []
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to USDA for query=%r", food_name)
        return []
    except Exception as e:
        logger.error("USDA Search Error for query=%r: %s", food_name, e)
        return []


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
    """Per-100g nutrition + household portions for a USDA food."""
    if fdc_id in DETAILS_CACHE:
        logger.debug("Details cache hit: %s", fdc_id)
        return DETAILS_CACHE[fdc_id]

    url = f"{get_base_url()}/food/{fdc_id}"
    params = {"api_key": get_api_key()}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 404:
            return None
        if response.status_code == 403:
            raise RuntimeError("Invalid USDA API key.")
        if response.status_code == 429:
            raise RuntimeError("USDA API rate limit exceeded.")

        response.raise_for_status()
        data = response.json()

        nutrient_values: Dict[int, float] = {}
        for item in data.get("foodNutrients", []):
            nutrient = item.get("nutrient") or {}
            nid = nutrient.get("id")
            amt = item.get("amount")
            if nid is not None and amt is not None:
                nutrient_values[nid] = amt

        result = {
            "fdc_id": data.get("fdcId"),
            "food_name": data.get("description"),
            "calories_per_100g": _first_present(
                nutrient_values, _USDA_NUTRIENT_IDS["calories"]
            ),
            "protein_per_100g": _first_present(
                nutrient_values, _USDA_NUTRIENT_IDS["protein"]
            ),
            "carbs_per_100g": _first_present(
                nutrient_values, _USDA_NUTRIENT_IDS["carbs"]
            ),
            "fat_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["fat"]),
            "portions": _extract_portions(data),
        }

        DETAILS_CACHE[fdc_id] = result
        return result

    except requests.exceptions.Timeout:
        logger.warning("USDA details timed out for fdc_id=%s", fdc_id)
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to USDA for fdc_id=%s", fdc_id)
        return None
    except Exception as e:
        logger.error("USDA Details Error for fdc_id=%s: %s", fdc_id, e)
        return None


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
