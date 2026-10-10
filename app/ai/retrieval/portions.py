"""USDA-only conversion delegates to the shared implementation."""

from typing import List, Optional

from app.ai.nutrition.portions import convert_to_grams as _convert


def convert_to_grams(
    quantity: float,
    unit: str,
    portions: Optional[List[dict]],
) -> Optional[dict]:
    return _convert(
        quantity,
        unit,
        portions,
        allow_approximations=False,
    )
