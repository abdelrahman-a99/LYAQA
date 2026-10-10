import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.ai.nutrition.portions import convert_to_grams as nutrition_convert
from app.ai.retrieval.portions import convert_to_grams as usda_convert


@pytest.mark.parametrize(
    "quantity,unit,portions,strict_expected,nutrition_expected",
    [
        (
            0.5,
            "kg",
            [],
            (500.0, "direct"),
            (500.0, "direct"),
        ),
        (
            2,
            "cup",
            [
                {
                    "amount": 1,
                    "unit": "cup",
                    "gram_weight": 50,
                    "modifier": "chopped",
                },
                {
                    "amount": 1,
                    "unit": "cup",
                    "gram_weight": 120,
                    "modifier": None,
                },
            ],
            (240.0, "usda_portion"),
            (240.0, "usda_portion"),
        ),
        (
            1,
            "cup",
            [],
            None,
            (240.0, "household_fallback"),
        ),
        (
            250,
            "ml",
            [],
            None,
            (250.0, "liquid_density_approx"),
        ),
        (
            1,
            "unknown",
            [],
            None,
            None,
        ),
        (
            2,
            "ml",
            [
                {
                    "amount": 1,
                    "unit": "ml",
                    "gram_weight": 2,
                    "modifier": None,
                },
            ],
            (4.0, "usda_portion"),
            (2.0, "liquid_density_approx"),
        ),
    ],
)
def test_conversion_policies(
    quantity,
    unit,
    portions,
    strict_expected,
    nutrition_expected,
):
    def summary(result):
        if result is None:
            return None
        return result["grams"], result["source"]

    assert summary(usda_convert(quantity, unit, portions)) == strict_expected

    assert summary(nutrition_convert(quantity, unit, portions)) == nutrition_expected


def test_reports_import_without_clients_or_circular_imports():

    env = {
        key: value for key, value in os.environ.items() if "API_KEY" not in key.upper()
    }

    code = """
import sys

from app.ai.presentation.validation import validate_meal_nutrition_report
import app.ai.nutrition as nutrition

profile = {
    "birth_date": "2000-01-01",
    "gender": "male",
    "weight": 80,
    "height": 180,
    "activity_level": "moderate",
    "goal": "weight_loss",
}

assert nutrition.calculate_bmr(profile) > 0

for name in (
    "groq",
    "torch",
    "requests",
    "app.ai.retrieval",
    "app.ai.nlp.parser",
):
    assert name not in sys.modules, name
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout + result.stderr
