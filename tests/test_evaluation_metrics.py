"""
tests/test_evaluation_metrics.py — Task 5 evaluation harness.

Loads evaluation/dataset.json, runs metrics functions from evaluation/metrics.py,
and asserts the harness works end-to-end.  Also exposes pytest functions that
verify each metric individually with hand-crafted inputs.

Run: pytest tests/test_evaluation_metrics.py -v
"""

import json

import pytest

from app.paths import EVALUATION_DATA_DIR
from evaluation.metrics import (
    calorie_error,
    entity_set_f1,
    food_name_exact_match_accuracy,
    intent_accuracy,
    quantity_extraction_accuracy,
    recommendation_restriction_compliance_at_k,
)

DATASET_PATH = EVALUATION_DATA_DIR / "dataset.json"


@pytest.fixture(scope="module")
def dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["examples"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. intent_accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentAccuracy:
    def test_perfect(self):
        preds = ["calculate_calories", "daily_plan", "general_chat"]
        gold = ["calculate_calories", "daily_plan", "general_chat"]
        assert intent_accuracy(preds, gold) == 1.0

    def test_partial(self):
        preds = ["calculate_calories", "daily_plan", "general_chat"]
        gold = ["calculate_calories", "food_nutrition", "general_chat"]
        assert intent_accuracy(preds, gold) == pytest.approx(2 / 3, rel=1e-3)

    def test_empty(self):
        assert intent_accuracy([], []) == 0.0

    def test_dataset_has_valid_intents(self, dataset):
        valid_intents = {
            "calculate_calories",
            "calculate_macros",
            "food_nutrition",
            "meal_recommendation",
            "daily_plan",
            "food_substitution",
            "fitness_nutrition_question",
            "update_profile",
            "general_chat",
        }
        for ex in dataset:
            assert ex["intent"] in valid_intents, f"Bad intent in example {ex['id']}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. entity_set_f1
# ═══════════════════════════════════════════════════════════════════════════


class TestNerF1:
    def test_perfect(self):
        preds = [{("chicken", "FOOD"), ("200", "QUANTITY")}]
        gold = [{("chicken", "FOOD"), ("200", "QUANTITY")}]
        r = entity_set_f1(preds, gold)
        assert r["f1"] == 1.0
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0

    def test_partial(self):
        preds = [{("chicken", "FOOD"), ("200", "QUANTITY")}]
        gold = [{("chicken", "FOOD")}]
        r = entity_set_f1(preds, gold)
        assert r["precision"] == pytest.approx(1 / 2)
        assert r["recall"] == 1.0
        assert r["tp"] == 1
        assert r["fp"] == 1
        assert r["fn"] == 0

    def test_miss(self):
        preds = [set()]
        gold = [{("chicken", "FOOD")}]
        r = entity_set_f1(preds, gold)
        assert r["f1"] == 0.0
        assert r["fn"] == 1

    def test_empty(self):
        r = entity_set_f1([], [])
        assert r["f1"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. food_name_exact_match_accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestFoodMatchingAccuracy:
    def test_case_insensitive_match(self):
        preds = ["Chicken Breast"]
        gold = ["chicken breast"]
        assert food_name_exact_match_accuracy(preds, gold) == 1.0

    def test_whitespace_insensitive(self):
        preds = ["  rice  "]
        gold = ["rice"]
        assert food_name_exact_match_accuracy(preds, gold) == 1.0

    def test_mismatch(self):
        preds = ["chicken breast"]
        gold = ["chicken thigh"]
        assert food_name_exact_match_accuracy(preds, gold) == 0.0

    def test_empty(self):
        assert food_name_exact_match_accuracy([], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. quantity_extraction_accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantityExtractionAccuracy:
    def test_exact(self):
        preds = [(200.0, "gram"), (2.0, "piece")]
        gold = [(200.0, "gram"), (2.0, "piece")]
        assert quantity_extraction_accuracy(preds, gold) == 1.0

    def test_unit_mismatch(self):
        preds = [(200.0, "gram")]
        gold = [(200.0, "cup")]
        assert quantity_extraction_accuracy(preds, gold) == 0.0

    def test_quantity_tolerance(self):
        preds = [(205.0, "gram")]
        gold = [(200.0, "gram")]
        assert quantity_extraction_accuracy(preds, gold, tolerance=0.05) == 1.0
        assert quantity_extraction_accuracy(preds, gold, tolerance=0.01) == 0.0

    def test_empty(self):
        assert quantity_extraction_accuracy([], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. calorie_error
# ═══════════════════════════════════════════════════════════════════════════


class TestCalorieError:
    def test_perfect(self):
        r = calorie_error([400.0, 600.0], [400.0, 600.0])
        assert r["mae"] == 0.0
        assert r["mape"] == 0.0

    def test_mae(self):
        r = calorie_error([410.0, 590.0], [400.0, 600.0])
        assert r["mae"] == pytest.approx(10.0)
        assert r["mape"] == pytest.approx((10 / 400 + 10 / 600) / 2, abs=1e-4)

    def test_empty(self):
        r = calorie_error([], [])
        assert r["mae"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. recommendation_restriction_compliance_at_k
# ═══════════════════════════════════════════════════════════════════════════


class TestRecommendationPrecisionAtK:
    def test_all_safe(self):
        recs = [["chicken", "rice", "oats"]]
        forb = [{"eggs"}]
        assert recommendation_restriction_compliance_at_k(recs, forb, k=3) == 1.0

    def test_with_violation(self):
        recs = [["chicken", "eggs", "oats"]]
        forb = [{"eggs"}]
        assert recommendation_restriction_compliance_at_k(
            recs, forb, k=3
        ) == pytest.approx(2 / 3, abs=1e-4)

    def test_k_cutoff(self):
        recs = [["chicken", "rice", "eggs", "oats"]]
        forb = [{"eggs"}]
        # k=2 → only chicken + rice → 100% safe
        assert recommendation_restriction_compliance_at_k(recs, forb, k=2) == 1.0

    def test_empty(self):
        assert recommendation_restriction_compliance_at_k([], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full-dataset sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestDataset:
    def test_dataset_loaded(self, dataset):
        assert len(dataset) == 30

    def test_every_example_has_required_keys(self, dataset):
        required = {
            "id",
            "input",
            "intent",
            "language",
            "entities",
            "foods",
            "quantities",
            "calories_gold",
        }
        for ex in dataset:
            missing = required - set(ex.keys())
            assert not missing, f"Example {ex['id']} missing keys: {missing}"

    def test_language_distribution(self, dataset):
        langs = [ex["language"] for ex in dataset]
        # Sanity: dataset contains at least 5 examples per language class
        assert langs.count("arabic") >= 5
        assert langs.count("english") >= 5
        assert langs.count("mixed") >= 3

    def test_calorie_examples_have_numeric_values(self, dataset):
        for ex in dataset:
            if ex["calories_gold"] is not None:
                assert isinstance(ex["calories_gold"], (int, float))
                assert ex["calories_gold"] > 0
