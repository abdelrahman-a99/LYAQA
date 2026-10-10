# -*- coding: utf-8 -*-
"""
test_nutrition_engine.py — Task 3: Integration tests for nutrition_engine.py.

Fixtures are based on:
  - Task 1 (nlp_parser.py) few-shot examples (profiles and food items).
  - Task 2 (usda_retrieval.py) real get_food_nutrition() output shapes.

Run with:
    python -m pytest test_nutrition_engine.py -v
"""

import pytest

import app.ai.nutrition as ne
from app.ai.nutrition.portions import convert_to_grams
from app.ai.nutrition.restrictions import (
    _check_food_against_restrictions,
    _collect_user_restrictions,
    _restriction_keys,
)
from app.ai.presentation import validation as nutrition_reports

# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def complete_profile_male():
    """
    Full 10-field profile from nlp_parser.py few-shot example 1.
    Male, born 2004-05-10, 90 kg, 175 cm, moderate activity, weight_loss.
    forbidden_foods = ["eggs"], allergies = ["dairy"].
    """
    return {
        "birth_date": "2004-05-10",
        "gender": "male",
        "weight": 90.0,
        "height": 175.0,
        "activity_level": "moderate",
        "goal": "weight_loss",
        "dietary_restrictions": [],
        "forbidden_foods": ["eggs"],
        "preferred_foods": [],
        "allergies": ["dairy"],
    }


@pytest.fixture
def complete_profile_female():
    """
    Full 10-field profile from nlp_parser.py few-shot example 2.
    Female, born 1999-03-15, 65 kg, 160 cm, moderate activity, weight_loss, vegan.
    """
    return {
        "birth_date": "1999-03-15",
        "gender": "female",
        "weight": 65.0,
        "height": 160.0,
        "activity_level": "moderate",
        "goal": "weight_loss",
        "dietary_restrictions": ["vegan"],
        "forbidden_foods": [],
        "preferred_foods": [],
        "allergies": ["peanuts"],
    }


@pytest.fixture
def chicken_nutrition_success():
    """
    Successful Task 2 output for 'grilled chicken breast'.
    Shape matches usda_retrieval.get_food_nutrition() on success=True.
    """
    return {
        "success": True,
        "original_query": "grilled chicken breast",
        "normalized_query": "grilled chicken breast",
        "normalization_method": "already_english",
        "fdc_id": 331960,
        "food_name": "Chicken breast, grilled",
        "data_type": "SR Legacy",
        "selected_rank": 1,
        "semantic_score": 0.87,
        "cross_encoder_score": 4.21,
        "final_match_score": 0.73,
        "calories_per_100g": 165.0,
        "protein_per_100g": 31.0,
        "carbs_per_100g": 0.0,
        "fat_per_100g": 3.6,
        "portions": [
            {"amount": 1.0, "unit": "piece", "modifier": None, "gram_weight": 120.0},
        ],
        "source": "USDA FoodData Central",
    }


@pytest.fixture
def rice_nutrition_success():
    """
    Successful Task 2 output for 'cooked white rice'.
    """
    return {
        "success": True,
        "original_query": "cooked white rice",
        "normalized_query": "cooked white rice",
        "normalization_method": "already_english",
        "fdc_id": 168880,
        "food_name": "Rice, white, cooked",
        "data_type": "SR Legacy",
        "selected_rank": 1,
        "semantic_score": 0.91,
        "cross_encoder_score": 5.10,
        "final_match_score": 0.82,
        "calories_per_100g": 130.0,
        "protein_per_100g": 2.7,
        "carbs_per_100g": 28.0,
        "fat_per_100g": 0.3,
        "portions": [
            {"amount": 1.0, "unit": "cup", "modifier": None, "gram_weight": 158.0},
        ],
        "source": "USDA FoodData Central",
    }


@pytest.fixture
def milk_nutrition_success():
    """
    Successful Task 2 output for 'whole milk' — used for ml/liter unit tests.
    """
    return {
        "success": True,
        "original_query": "whole milk",
        "normalized_query": "whole milk",
        "normalization_method": "already_english",
        "fdc_id": 170860,
        "food_name": "Milk, whole",
        "data_type": "SR Legacy",
        "selected_rank": 1,
        "semantic_score": 0.89,
        "cross_encoder_score": 4.80,
        "final_match_score": 0.77,
        "calories_per_100g": 61.0,
        "protein_per_100g": 3.2,
        "carbs_per_100g": 4.8,
        "fat_per_100g": 3.3,
        "portions": [],
        "source": "USDA FoodData Central",
    }


@pytest.fixture
def egg_nutrition_success():
    """
    Successful Task 2 output for 'boiled egg' — forbidden for profile_male.
    original_query intentionally set to food_name_en (English path).
    """
    return {
        "success": True,
        "original_query": "boiled egg",
        "normalized_query": "boiled egg",
        "normalization_method": "already_english",
        "fdc_id": 173424,
        "food_name": "Egg, whole, cooked, boiled",
        "data_type": "SR Legacy",
        "selected_rank": 1,
        "semantic_score": 0.93,
        "cross_encoder_score": 5.50,
        "final_match_score": 0.85,
        "calories_per_100g": 155.0,
        "protein_per_100g": 13.0,
        "carbs_per_100g": 1.1,
        "fat_per_100g": 11.0,
        "portions": [
            {"amount": 1.0, "unit": "piece", "modifier": None, "gram_weight": 50.0},
        ],
        "source": "USDA FoodData Central",
    }


# ── Task 2 failure shapes ────────────────────────────────────────────────────


@pytest.fixture
def nutrition_failure_not_found():
    return {
        "success": False,
        "original_query": "dragon fruit powder xyz",
        "normalized_query": "dragon fruit powder xyz",
        "error": "food_not_found",
    }


@pytest.fixture
def nutrition_failure_empty_name():
    return {"success": False, "error": "empty_food_name"}


@pytest.fixture
def nutrition_failure_normalization():
    return {"success": False, "original_query": "???", "error": "normalization_failed"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Profile validation — missing / None mandatory fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestProfileValidation:
    @pytest.mark.parametrize(
        "missing_field",
        [
            "birth_date",
            "gender",
            "weight",
            "height",
            "activity_level",
            "goal",
        ],
    )
    def test_calculate_bmr_raises_on_none_field(
        self, complete_profile_male, missing_field
    ):
        profile = {**complete_profile_male, missing_field: None}
        with pytest.raises(ValueError, match=missing_field):
            ne.calculate_bmr(profile)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "birth_date",
            "gender",
            "weight",
            "height",
            "activity_level",
            "goal",
        ],
    )
    def test_calculate_tdee_raises_on_none_field(
        self, complete_profile_male, missing_field
    ):
        profile = {**complete_profile_male, missing_field: None}
        with pytest.raises(ValueError, match=missing_field):
            ne.calculate_tdee(profile)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "birth_date",
            "gender",
            "weight",
            "height",
            "activity_level",
            "goal",
        ],
    )
    def test_calculate_target_calories_raises_on_none_field(
        self, complete_profile_male, missing_field
    ):
        profile = {**complete_profile_male, missing_field: None}
        with pytest.raises(ValueError, match=missing_field):
            ne.calculate_target_calories(profile)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "birth_date",
            "gender",
            "weight",
            "height",
            "activity_level",
            "goal",
        ],
    )
    def test_calculate_macros_raises_on_none_field(
        self, complete_profile_male, missing_field
    ):
        profile = {**complete_profile_male, missing_field: None}
        with pytest.raises(ValueError, match=missing_field):
            ne.calculate_macros(profile)

    def test_multiple_none_fields_raises(self, complete_profile_male):
        profile = {**complete_profile_male, "birth_date": None, "gender": None}
        with pytest.raises(ValueError) as exc_info:
            ne.calculate_bmr(profile)
        msg = str(exc_info.value)
        assert "birth_date" in msg
        assert "gender" in msg

    def test_non_dict_profile_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            ne.calculate_bmr(None)

    def test_complete_profile_does_not_raise(self, complete_profile_male):
        bmr = ne.calculate_bmr(complete_profile_male)
        assert isinstance(bmr, float)
        assert bmr > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Import safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportSafety:
    def test_does_not_import_usda_retrieval(self):
        import importlib

        importlib.reload(ne)

    def test_does_not_import_nlp_parser(self):
        assert "nlp_parser" not in (getattr(ne, "__dict__", {}) or {})


# ═══════════════════════════════════════════════════════════════════════════════
# 3. P0-1 — Punctuation-safe restriction matching (USDA comma-separated names)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPunctuationSafeRestrictionMatching:
    """
    USDA food names use commas (e.g. "Egg, whole, raw, fresh") and other
    punctuation (%, hyphens).  _restriction_keys must match them correctly.
    Before the fix all these returned an empty set.
    """

    def test_egg_usda_style_name(self):
        assert _restriction_keys("Egg, whole, raw, fresh") == {"eggs"}

    def test_cheese_cheddar(self):
        assert _restriction_keys("Cheese, cheddar") == {"dairy"}

    def test_fish_salmon_comma_separated(self):
        assert _restriction_keys("Fish, salmon, Atlantic, raw") == {"fish"}

    def test_milk_with_percent(self):
        keys = _restriction_keys("Milk, whole, 3.25% milkfat")
        assert "dairy" in keys

    def test_arabic_names_still_match_after_fix(self):
        """Ensure the punctuation-strip does not break Arabic alias matching."""
        assert "eggs" in _restriction_keys("بيض")
        assert "dairy" in _restriction_keys("زبادي")

    def test_rice_does_not_match_eggs(self):
        """'أرز أبيض' must NOT match 'بيض' — token boundary protection."""
        keys = _restriction_keys("أرز أبيض")
        assert "eggs" not in keys

    def test_restriction_check_on_usda_food_name_field(
        self, complete_profile_male, egg_nutrition_success
    ):
        """
        The full pipeline: a food whose food_name is the USDA comma-separated
        string must be flagged by _check_food_against_restrictions.
        """
        food = {
            "food_name_original": "بيض",
            "food_name_en": "boiled egg",
            "food_name": "Egg, whole, cooked, boiled",  # ← USDA style
        }
        user_restrictions = _collect_user_restrictions(complete_profile_male)
        matched = _check_food_against_restrictions(food, user_restrictions)
        assert "eggs" in matched

    def test_dairy_usda_name_flagged_in_meal(
        self, complete_profile_male, milk_nutrition_success
    ):
        """
        Milk, whole (USDA name) must trigger the dairy restriction when
        validate_meal_nutrition is called.
        """
        foods = [
            {
                "food_name_original": "milk",
                "food_name_en": "whole milk",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [milk_nutrition_success])
        result = ne.validate_meal_nutrition(meal, complete_profile_male)
        assert result["valid"] is False
        assert any(
            "dairy" in v["matched_restrictions"]
            for v in result["restriction_violations"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. P0-2 — original_query matching: accepts food_name_original (Arabic) too
# ═══════════════════════════════════════════════════════════════════════════════


class TestOriginalQueryMatching:
    """
    Task 4 may call Task 2 with either food_name_en or food_name_original.
    Both must be accepted; only genuine mismatches (wrong index) should reject.
    """

    def test_arabic_original_query_accepted(self):
        """
        original_query == food_name_original (Arabic) → must succeed.
        This is the exact case described in Task 2's own documentation.
        """
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        nutrition = [
            {
                "success": True,
                "original_query": "فراخ مشوية",
                "food_name": "Chicken, breast, grilled",
                "calories_per_100g": 165,
                "protein_per_100g": 31,
                "carbs_per_100g": 0,
                "fat_per_100g": 3.6,
                "portions": [],
            }
        ]
        r = ne.calculate_meal_total(foods, nutrition)
        assert r["success"] is True, r["unmatched_foods"]
        assert r["total_calories"] == pytest.approx(330.0)

    def test_english_original_query_still_accepted(self, chicken_nutrition_success):
        """food_name_en path (the previous behaviour) must still work."""
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        r = ne.calculate_meal_total(foods, [chicken_nutrition_success])
        assert r["success"] is True

    def test_genuinely_mismatched_query_rejected(self):
        """
        If original_query doesn't match either food_name_en or
        food_name_original, it must be flagged as food_mismatch.
        """
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        wrong_nutrition = [
            {
                "success": True,
                "original_query": "something completely different",
                "food_name": "Some other food",
                "calories_per_100g": 100,
                "protein_per_100g": 5,
                "carbs_per_100g": 10,
                "fat_per_100g": 2,
                "portions": [],
            }
        ]
        r = ne.calculate_meal_total(foods, wrong_nutrition)
        assert r["success"] is False
        assert r["unmatched_foods"][0]["reason"] == "food_mismatch"

    def test_empty_original_query_skips_check(self, chicken_nutrition_success):
        """When original_query is absent, the check is skipped gracefully."""
        no_query = {**chicken_nutrition_success}
        no_query.pop("original_query", None)
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        r = ne.calculate_meal_total(foods, [no_query])
        assert r["success"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. P0-3 — Household fallback units
# ═══════════════════════════════════════════════════════════════════════════════


class TestHouseholdFallbackUnits:
    """
    All 7 previously-broken units must now return a numeric grams value
    with source == "household_fallback" when no USDA portions match.
    """

    @pytest.mark.parametrize(
        "unit,expected_g",
        [
            ("slice", 30.0),
            ("tablespoon", 15.0),
            ("teaspoon", 5.0),
            ("bowl", 250.0),
            ("can", 330.0),
            ("bottle", 500.0),
            ("loaf", 90.0),
        ],
    )
    def test_unit_returns_fallback(self, unit, expected_g):
        result = convert_to_grams(1.0, unit, [])
        assert result is not None, f"{unit} returned None"
        assert result["source"] == "household_fallback"
        assert result["grams"] == pytest.approx(expected_g)

    @pytest.mark.parametrize(
        "unit",
        [
            "slice",
            "tablespoon",
            "teaspoon",
            "bowl",
            "can",
            "bottle",
            "loaf",
        ],
    )
    def test_quantity_multiplied_correctly(self, unit):
        """quantity=2 must double the fallback grams."""
        result_1 = convert_to_grams(1.0, unit, [])
        result_2 = convert_to_grams(2.0, unit, [])
        assert result_2["grams"] == pytest.approx(result_1["grams"] * 2)

    def test_usda_portion_still_preferred_over_fallback(self, rice_nutrition_success):
        """When a USDA portion exists for the unit, it wins over the fallback."""
        result = convert_to_grams(1.0, "cup", rice_nutrition_success["portions"])
        assert result["source"] == "usda_portion"
        assert result["grams"] == pytest.approx(158.0)

    def test_fallback_used_in_meal_total(self, rice_nutrition_success):
        """A bowl of rice with no portions list should use household fallback."""
        rice_no_portions = {
            **rice_nutrition_success,
            "portions": [],
            "original_query": "رز أبيض",
        }
        foods = [
            {
                "food_name_original": "رز أبيض",
                "food_name_en": "cooked white rice",
                "quantity": 1.0,
                "unit": "bowl",
            }
        ]
        result = ne.calculate_meal_total(foods, [rice_no_portions])
        assert result["success"] is True
        assert result["foods"][0]["conversion_source"] == "household_fallback"
        # 1 bowl = 250g; 130 kcal/100g → 325 kcal
        assert result["total_calories"] == pytest.approx(325.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. P0-4 — validate_meal_nutrition works with incomplete / empty profile
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateMealNutritionPartialProfile:
    """
    validate_meal_nutrition must NOT require the 6 mandatory profile fields.
    It only needs restriction-related keys, which are optional.
    """

    def test_completely_empty_profile_dict(self):
        """validate_meal_nutrition({}, meal) must not raise."""
        meal = {
            "success": True,
            "foods": [],
            "total_calories": 400,
            "total_protein_g": 20,
            "total_carbs_g": 40,
            "total_fat_g": 10,
        }
        result = ne.validate_meal_nutrition(meal, {}, target_calories=400)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_none_profile_treated_as_empty_dict(self):
        """None profile must be coerced to {} without raising."""
        meal = {
            "success": True,
            "foods": [],
            "total_calories": 300,
            "total_protein_g": 15,
            "total_carbs_g": 30,
            "total_fat_g": 8,
        }
        result = ne.validate_meal_nutrition(meal, None, target_calories=300)
        assert result["valid"] is True

    def test_all_mandatory_fields_none(self):
        """All 6 mandatory fields set to None must still work in meal validation."""
        partial = {
            "birth_date": None,
            "gender": None,
            "weight": None,
            "height": None,
            "activity_level": None,
            "goal": None,
            "dietary_restrictions": [],
            "forbidden_foods": ["eggs"],
            "preferred_foods": [],
            "allergies": [],
        }
        meal = {
            "success": True,
            "foods": [],
            "total_calories": 400,
            "total_protein_g": 20,
            "total_carbs_g": 40,
            "total_fat_g": 10,
        }
        result = ne.validate_meal_nutrition(meal, partial, target_calories=400)
        assert result["valid"] is True

    def test_restriction_still_checked_with_partial_profile(
        self, egg_nutrition_success
    ):
        """Restrictions from a partial profile must still be enforced."""
        partial = {
            "forbidden_foods": ["eggs"],
            "allergies": [],
            "dietary_restrictions": [],
        }
        foods = [
            {
                "food_name_original": "بيضة",
                "food_name_en": "boiled egg",
                "quantity": 50.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [egg_nutrition_success])
        result = ne.validate_meal_nutrition(meal, partial)
        assert result["valid"] is False
        assert len(result["restriction_violations"]) > 0

    def test_validate_daily_nutrition_still_requires_full_profile(
        self, complete_profile_male
    ):
        """validate_daily_nutrition MUST still raise on incomplete profiles."""
        incomplete = {**complete_profile_male, "birth_date": None}
        with pytest.raises(ValueError, match="birth_date"):
            ne.validate_daily_nutrition([], incomplete)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. P0-5 — invalid_quantity never crashes calculate_meal_total
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvalidQuantityHandling:
    """
    If unit conversion returns quantity_g <= 0 (edge case), the item must land
    in unmatched_foods with reason "invalid_quantity" — not raise an exception.
    """

    def test_zero_quantity_is_recorded_not_raised(self, chicken_nutrition_success):
        """Force quantity_g = 0 by passing quantity=0 with gram unit."""
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 0.0,
                "unit": "gram",
            }
        ]
        result = ne.calculate_meal_total(foods, [chicken_nutrition_success])
        assert result["success"] is False
        assert result["unmatched_foods"][0]["reason"] == "invalid_quantity"

    def test_negative_quantity_is_recorded_not_raised(self, chicken_nutrition_success):
        """Negative quantity must also land in unmatched_foods."""
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": -50.0,
                "unit": "gram",
            }
        ]
        result = ne.calculate_meal_total(foods, [chicken_nutrition_success])
        assert result["success"] is False
        assert result["unmatched_foods"][0]["reason"] == "invalid_quantity"

    def test_rest_of_meal_succeeds_despite_invalid_item(
        self, chicken_nutrition_success, rice_nutrition_success
    ):
        """One invalid item must not prevent the rest of the meal from computing."""
        foods = [
            {
                "food_name_original": "فراخ",
                "food_name_en": "grilled chicken breast",
                "quantity": -1.0,
                "unit": "gram",
            },
            {
                "food_name_original": "رز",
                "food_name_en": "cooked white rice",
                "quantity": 200.0,
                "unit": "gram",
            },
        ]
        result = ne.calculate_meal_total(
            foods, [chicken_nutrition_success, rice_nutrition_success]
        )
        assert result["success"] is False
        assert len(result["foods"]) == 1  # rice succeeds
        assert len(result["unmatched_foods"]) == 1  # chicken fails
        assert result["total_calories"] == pytest.approx(260.0)  # 130*2 = 260


# ═══════════════════════════════════════════════════════════════════════════════
# 8. P1-6 — scale_meal_to_target
# ═══════════════════════════════════════════════════════════════════════════════


class TestScaleMealToTarget:
    """
    scale_meal_to_target must proportionally rescale food quantities toward
    a calorie target without calling the LLM or USDA.
    """

    @pytest.fixture
    def gram_foods(self):
        return [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            },
            {
                "food_name_original": "رز",
                "food_name_en": "cooked white rice",
                "quantity": 150.0,
                "unit": "gram",
            },
        ]

    def test_scale_down(self, gram_foods):
        """current > target: quantities must decrease."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=460, target_calories=400
        )
        assert result["achievable"] is True
        assert 0.5 < result["scale_factor"] < 1.0
        for orig, scaled in zip(gram_foods, result["scaled_foods"]):
            assert scaled["quantity"] < orig["quantity"]

    def test_scale_up(self, gram_foods):
        """current < target: quantities must increase."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=300, target_calories=450
        )
        assert result["achievable"] is True
        assert result["scale_factor"] > 1.0
        for orig, scaled in zip(gram_foods, result["scaled_foods"]):
            assert scaled["quantity"] > orig["quantity"]

    def test_max_scale_factor_clamps(self, gram_foods):
        """current far below target → max_scale_factor kicks in, clamped=True."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=100, target_calories=2000, max_scale_factor=2.0
        )
        assert result["clamped"] is True
        assert result["achievable"] is False
        assert result["scale_factor"] == pytest.approx(2.0)

    def test_min_scale_factor_clamps(self, gram_foods):
        """current far above target → 1/max_scale_factor clamps, clamped=True."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=2000, target_calories=100, max_scale_factor=2.0
        )
        assert result["clamped"] is True
        assert result["scale_factor"] == pytest.approx(0.5)

    def test_achievable_false_when_current_zero(self, gram_foods):
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=0, target_calories=400
        )
        assert result["achievable"] is False
        assert result["scale_factor"] == pytest.approx(1.0)
        # Quantities must be unchanged
        for orig, scaled in zip(gram_foods, result["scaled_foods"]):
            assert scaled["quantity"] == orig["quantity"]

    def test_non_gram_unit_scaled_and_floored(self):
        """Piece/cup units scaled but never below 0.25."""
        foods = [
            {
                "food_name_original": "رز",
                "food_name_en": "cooked white rice",
                "quantity": 1.0,
                "unit": "cup",
            },
        ]
        result = ne.scale_meal_to_target(
            foods, current_calories=200, target_calories=50
        )
        # factor = 50/200 = 0.25 → 1 cup * 0.25 = 0.25 — exactly the floor
        scaled_qty = result["scaled_foods"][0]["quantity"]
        assert scaled_qty >= 0.25

    def test_non_gram_unit_scaled_up(self):
        foods = [
            {
                "food_name_original": "رز",
                "food_name_en": "cooked white rice",
                "quantity": 1.0,
                "unit": "cup",
            },
        ]
        result = ne.scale_meal_to_target(
            foods, current_calories=200, target_calories=300
        )
        assert result["scaled_foods"][0]["quantity"] > 1.0

    def test_gram_quantities_rounded_to_5(self, gram_foods):
        """Gram quantities must be rounded to nearest 5g."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=460, target_calories=400
        )
        for food in result["scaled_foods"]:
            if food["unit"] == "gram":
                assert food["quantity"] % 5 == pytest.approx(0, abs=0.01)

    def test_scaled_foods_preserve_all_other_keys(self, gram_foods):
        """scale_meal_to_target must not drop any food dict keys."""
        result = ne.scale_meal_to_target(
            gram_foods, current_calories=460, target_calories=400
        )
        for orig, scaled in zip(gram_foods, result["scaled_foods"]):
            for key in orig:
                assert key in scaled


# ═══════════════════════════════════════════════════════════════════════════════
# 9. P1-7 — min_protein_g parameter in validate_meal_nutrition
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinProteinG:
    """
    The optional min_protein_g floor must be checked independently of
    calorie and restriction checks.
    """

    @pytest.fixture
    def low_protein_meal(self):
        return {
            "success": True,
            "foods": [],
            "total_calories": 400,
            "total_protein_g": 5.0,
            "total_carbs_g": 75.0,
            "total_fat_g": 8.0,
        }

    def test_meal_fails_when_protein_below_floor(self, low_protein_meal):
        result = ne.validate_meal_nutrition(low_protein_meal, {}, min_protein_g=30.0)
        assert result["valid"] is False
        assert result["checks"]["protein"] is False
        assert any("protein" in issue for issue in result["issues"])

    def test_meal_passes_when_protein_meets_floor(self, low_protein_meal):
        high = {**low_protein_meal, "total_protein_g": 35.0}
        result = ne.validate_meal_nutrition(high, {}, min_protein_g=30.0)
        assert result["checks"]["protein"] is True

    def test_protein_check_skipped_when_not_given(self, low_protein_meal):
        """Without min_protein_g, protein_valid is always True."""
        result = ne.validate_meal_nutrition(low_protein_meal, {})
        assert result["checks"]["protein"] is True

    def test_protein_check_in_report_wrapper(self, low_protein_meal):
        """validate_meal_nutrition_report must pass min_protein_g through."""
        report = nutrition_reports.validate_meal_nutrition_report(
            low_protein_meal, {}, min_protein_g=30.0
        )
        assert report["valid"] is False
        assert any("protein" in v for v in report["violations"])

    def test_protein_key_in_checks_dict(self, low_protein_meal):
        result = ne.validate_meal_nutrition(low_protein_meal, {})
        assert "protein" in result["checks"]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. P1-8 — tolerance constants are the actual defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestToleranceConstants:
    """
    The four function signatures must use the module-level constants as
    their default values, not hardcoded literals.
    """

    def test_meal_calorie_tolerance_constant_exists(self):
        assert hasattr(ne, "MEAL_CALORIE_TOLERANCE")
        assert ne.MEAL_CALORIE_TOLERANCE == pytest.approx(0.15)

    def test_daily_calorie_tolerance_constant_exists(self):
        assert hasattr(ne, "DAILY_CALORIE_TOLERANCE")
        assert ne.DAILY_CALORIE_TOLERANCE == pytest.approx(0.20)

    def test_daily_macro_tolerance_constant_exists(self):
        assert hasattr(ne, "DAILY_MACRO_TOLERANCE")
        assert ne.DAILY_MACRO_TOLERANCE == pytest.approx(0.20)

    def test_validate_meal_nutrition_uses_constant_as_default(self):
        """Default calorie_tolerance must be MEAL_CALORIE_TOLERANCE."""
        import inspect

        sig = inspect.signature(ne.validate_meal_nutrition)
        default = sig.parameters["calorie_tolerance"].default
        assert default == pytest.approx(ne.MEAL_CALORIE_TOLERANCE)

    def test_validate_daily_nutrition_uses_constant_as_default(self):
        import inspect

        sig = inspect.signature(ne.validate_daily_nutrition)
        default = sig.parameters["calorie_tolerance"].default
        assert default == pytest.approx(ne.DAILY_CALORIE_TOLERANCE)

    def test_report_wrappers_use_same_constants(self):
        import inspect

        meal_sig = inspect.signature(nutrition_reports.validate_meal_nutrition_report)
        daily_sig = inspect.signature(nutrition_reports.validate_daily_nutrition_report)
        assert meal_sig.parameters["calorie_tolerance"].default == pytest.approx(
            ne.MEAL_CALORIE_TOLERANCE
        )
        assert daily_sig.parameters["calorie_tolerance"].default == pytest.approx(
            ne.DAILY_CALORIE_TOLERANCE
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Milliliter and liter conversion (regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMilliliterAndLiterConversion:
    def test_milliliter_converts_1_to_1(self):
        result = convert_to_grams(250.0, "milliliter", [])
        assert result is not None
        assert result["grams"] == pytest.approx(250.0)
        assert result["source"] == "liquid_density_approx"

    def test_milliliter_alias_ml(self):
        result = convert_to_grams(100.0, "ml", [])
        assert result["grams"] == pytest.approx(100.0)

    def test_liter_converts_to_1000_grams(self):
        result = convert_to_grams(1.0, "liter", [])
        assert result["grams"] == pytest.approx(1000.0)
        assert result["source"] == "liquid_density_approx"

    def test_liter_alias_l(self):
        result = convert_to_grams(0.5, "l", [])
        assert result["grams"] == pytest.approx(500.0)

    def test_milliliter_in_meal_total(self, milk_nutrition_success):
        foods = [
            {
                "food_name_original": "milk",
                "food_name_en": "whole milk",
                "quantity": 250.0,
                "unit": "milliliter",
            }
        ]
        result = ne.calculate_meal_total(foods, [milk_nutrition_success])
        assert result["success"] is True
        assert result["total_calories"] == pytest.approx(152.5)

    def test_gram_still_works(self):
        result = convert_to_grams(200.0, "gram", [])
        assert result["grams"] == pytest.approx(200.0)
        assert result["source"] == "direct"

    def test_kilogram_still_works(self):
        result = convert_to_grams(1.5, "kilogram", [])
        assert result["grams"] == pytest.approx(1500.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. calculate_meal_total — Task 2 failure items (regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMealTotalWithFailures:
    def test_single_failure_item_is_dropped(self, nutrition_failure_not_found):
        foods = [
            {
                "food_name_original": "dragon fruit powder xyz",
                "food_name_en": "dragon fruit powder xyz",
                "quantity": 100.0,
                "unit": "gram",
            }
        ]
        result = ne.calculate_meal_total(foods, [nutrition_failure_not_found])
        assert result["success"] is False
        assert result["total_calories"] == 0.0
        assert result["unmatched_foods"][0]["reason"] == "food_not_found"

    def test_partial_failure_returns_partial_total(
        self, chicken_nutrition_success, nutrition_failure_not_found
    ):
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            },
            {
                "food_name_original": "mystery food",
                "food_name_en": "dragon fruit powder xyz",
                "quantity": 100.0,
                "unit": "gram",
            },
        ]
        fail = {
            **nutrition_failure_not_found,
            "original_query": "dragon fruit powder xyz",
        }
        result = ne.calculate_meal_total(foods, [chicken_nutrition_success, fail])
        assert result["success"] is False
        assert len(result["foods"]) == 1
        assert result["total_calories"] == pytest.approx(330.0)

    def test_unit_conversion_failure_is_reported(self, chicken_nutrition_success):
        no_portions = {**chicken_nutrition_success, "portions": []}
        # Use an unknown unit that has no fallback
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 2.0,
                "unit": "unknownunit_xyz",
            }
        ]
        result = ne.calculate_meal_total(foods, [no_portions])
        assert result["success"] is False
        assert result["unmatched_foods"][0]["reason"] == "unit_conversion_failed"

    def test_length_mismatch_raises(self, chicken_nutrition_success):
        foods = [
            {
                "food_name_original": "a",
                "food_name_en": "a",
                "quantity": 1,
                "unit": "gram",
            },
            {
                "food_name_original": "b",
                "food_name_en": "b",
                "quantity": 1,
                "unit": "gram",
            },
        ]
        with pytest.raises(ValueError, match="must match"):
            ne.calculate_meal_total(foods, [chicken_nutrition_success])


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Forbidden food detection (regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenFoodDetection:
    def test_forbidden_food_detected_in_meal(
        self, complete_profile_male, egg_nutrition_success
    ):
        foods = [
            {
                "food_name_original": "boiled eggs",
                "food_name_en": "boiled egg",
                "quantity": 100.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [egg_nutrition_success])
        result = ne.validate_meal_nutrition(meal, complete_profile_male)
        assert result["valid"] is False
        assert "eggs" in result["restriction_violations"][0]["matched_restrictions"]

    def test_clean_meal_passes_restriction_check(
        self, complete_profile_male, rice_nutrition_success
    ):
        foods = [
            {
                "food_name_original": "رز",
                "food_name_en": "cooked white rice",
                "quantity": 1.0,
                "unit": "cup",
            }
        ]
        meal = ne.calculate_meal_total(foods, [rice_nutrition_success])
        result = ne.validate_meal_nutrition(meal, complete_profile_male)
        assert result["restriction_violations"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Validator output format {"valid", "violations"} (regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatorOutputFormat:
    def test_meal_report_format_keys(
        self, complete_profile_male, egg_nutrition_success
    ):
        foods = [
            {
                "food_name_original": "boiled eggs",
                "food_name_en": "boiled egg",
                "quantity": 100.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [egg_nutrition_success])
        report = nutrition_reports.validate_meal_nutrition_report(
            meal, complete_profile_male
        )
        assert set(report.keys()) == {"valid", "violations"}

    def test_meal_report_violations_is_list_of_strings(
        self, complete_profile_male, egg_nutrition_success
    ):
        foods = [
            {
                "food_name_original": "boiled eggs",
                "food_name_en": "boiled egg",
                "quantity": 100.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [egg_nutrition_success])
        report = nutrition_reports.validate_meal_nutrition_report(
            meal, complete_profile_male
        )
        assert isinstance(report["violations"], list)
        assert all(isinstance(v, str) for v in report["violations"])

    def test_daily_reportkeys(self, complete_profile_male, chicken_nutrition_success):
        foods = [
            {
                "food_name_original": "فراخ مشوية",
                "food_name_en": "grilled chicken breast",
                "quantity": 200.0,
                "unit": "gram",
            }
        ]
        meal = ne.calculate_meal_total(foods, [chicken_nutrition_success])
        report = nutrition_reports.validate_daily_nutrition_report(
            [meal], complete_profile_male
        )
        assert set(report.keys()) == {"valid", "violations"}
        assert isinstance(report["violations"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Calculation formula preservation (regression smoke tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculationFormulas:
    def test_bmr_male_known_range(self, complete_profile_male):
        bmr = ne.calculate_bmr(complete_profile_male)
        assert 1800 < bmr < 2000

    def test_tdee_moderate_activity(self, complete_profile_male):
        bmr = ne.calculate_bmr(complete_profile_male)
        tdee = ne.calculate_tdee(complete_profile_male)
        assert tdee == pytest.approx(bmr * 1.55, rel=1e-3)

    def test_weight_loss_applies_400_kcal_deficit(self, complete_profile_male):
        tdee = ne.calculate_tdee(complete_profile_male)
        target = ne.calculate_target_calories(complete_profile_male)
        assert target == pytest.approx(tdee - 400, rel=1e-3)

    def test_muscle_gain_applies_400_kcal_surplus(self, complete_profile_male):
        profile = {**complete_profile_male, "goal": "muscle_gain"}
        tdee = ne.calculate_tdee(profile)
        target = ne.calculate_target_calories(profile)
        assert target == pytest.approx(tdee + 400, rel=1e-3)

    def test_macros_protein_factor_weight_loss(self, complete_profile_male):
        macros = ne.calculate_macros(complete_profile_male)
        assert macros["protein_g"] == pytest.approx(90.0 * 2.0, rel=1e-3)

    def test_macros_fat_is_25_percent_of_calories(self, complete_profile_male):
        macros = ne.calculate_macros(complete_profile_male)
        fat_from_calories = macros["calories"] * 0.25
        assert macros["fat_g"] == pytest.approx(fat_from_calories / 9, rel=1e-3)

    def test_food_nutrition_scaling(self, chicken_nutrition_success):
        result = ne.calculate_food_nutrition(chicken_nutrition_success, 200.0)
        assert result["calories"] == pytest.approx(330.0)
        assert result["protein_g"] == pytest.approx(62.0)

    def test_min_safe_calories_floor_male(self):
        profile = {
            "birth_date": "2004-01-01",
            "gender": "male",
            "weight": 10.0,
            "height": 50.0,
            "activity_level": "sedentary",
            "goal": "weight_loss",
            "dietary_restrictions": [],
            "forbidden_foods": [],
            "preferred_foods": [],
            "allergies": [],
        }
        assert ne.calculate_target_calories(profile) >= 1500
