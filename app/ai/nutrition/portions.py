"""Canonical portion conversion with explicit approximation policy."""

from __future__ import annotations

from typing import Dict, List, Optional

_PORTION_UNIT_ALIASES: Dict[str, str] = {
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tablespoon",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "tsp": "teaspoon",
    "teaspoon": "teaspoon",
    "teaspoons": "teaspoon",
    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "small": "piece",
    "medium": "piece",
    "large": "piece",
    "unit": "piece",
    "each": "piece",
    "can": "can",
    "bottle": "bottle",
    "loaf": "loaf",
    "bowl": "bowl",
    "serving": "serving",
}


_HOUSEHOLD_FALLBACK_G: Dict[str, float] = {
    "bowl": 250.0,
    "cup": 240.0,
    "slice": 30.0,
    "piece": 100.0,
    "tablespoon": 15.0,
    "teaspoon": 5.0,
    "loaf": 90.0,
    "can": 330.0,
    "bottle": 500.0,
    "serving": 100.0,
}


def convert_to_grams(
    quantity: float,
    unit: str,
    portions: Optional[List[dict]],
    *,
    allow_approximations: bool = True,
) -> Optional[dict]:
    """
    Convert a quantity and unit to grams.

    With allow_approximations=True, preserve the nutrition engine's
    existing order: exact SI units, approximate liquid density,
    USDA portions, then generic household estimates.

    With allow_approximations=False, use only exact SI conversions
    or matching USDA portions.

    Return grams, source, and matched_portion; or None when conversion
    is unavailable.
    """
    if quantity is None:
        return None

    u = (unit or "").strip().lower()

    # ── 1. Direct SI units ────────────────────────────────────────────────
    if u in ("g", "gram", "grams"):
        return {"grams": float(quantity), "source": "direct", "matched_portion": None}
    if u in ("kg", "kilogram", "kilograms"):
        return {
            "grams": float(quantity) * 1000.0,
            "source": "direct",
            "matched_portion": None,
        }

    # ── 2. Liquid units — approximate density = 1 g/ml ───────────────────
    # Assumption: liquid density ≈ water density (1 g/ml).
    # Accurate for water-like liquids; see docstring for known deviations.
    if allow_approximations and u in ("ml", "milliliter", "milliliters"):
        return {
            "grams": float(quantity) * 1.0,
            "source": "liquid_density_approx",
            "matched_portion": None,
        }
    if allow_approximations and u in ("l", "liter", "liters"):
        return {
            "grams": float(quantity) * 1000.0,
            "source": "liquid_density_approx",
            "matched_portion": None,
        }

    # ── 3. USDA portions ─────────────────────────────────────────────────
    target = _PORTION_UNIT_ALIASES.get(u, u)

    if portions:
        exact_matches = []
        fallback_matches = []
        for p in portions:
            p_unit_raw = (p.get("unit") or "").lower()
            p_unit = _PORTION_UNIT_ALIASES.get(p_unit_raw, p_unit_raw)
            if p_unit != target:
                continue
            gw = p.get("gram_weight")
            amt = p.get("amount")
            if not gw or not amt:
                continue
            grams_per_unit = float(gw) / float(amt)
            if p.get("modifier") is None:
                exact_matches.append((grams_per_unit, p))
            else:
                fallback_matches.append((grams_per_unit, p))

        chosen = (
            exact_matches[0]
            if exact_matches
            else (fallback_matches[0] if fallback_matches else None)
        )
        if chosen:
            grams_per_unit, matched = chosen
            return {
                "grams": round(grams_per_unit * float(quantity), 2),
                "source": "usda_portion",
                "matched_portion": matched,
            }

    # ── 4. Household-measure fallback (last resort) ───────────────────────
    # Reached when: (a) no portions list, or (b) portions exist but none
    # match the requested unit.  Uses generic per-unit gram estimates.
    # Task 4 MUST surface source == "household_fallback" to the user.
    if not allow_approximations:
        return None

    fallback_grams = _HOUSEHOLD_FALLBACK_G.get(target)
    if fallback_grams is not None:
        return {
            "grams": round(fallback_grams * float(quantity), 2),
            "source": "household_fallback",
            "matched_portion": None,
        }

    return None
