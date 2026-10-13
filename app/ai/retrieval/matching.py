"""Semantic retrieval and deterministic food match reranking."""

from __future__ import annotations

import re
from typing import List

import numpy as np
from nltk.stem import SnowballStemmer
from rapidfuzz import fuzz

from app.ai.inference import run_inference


def semantic_retrieve(
    query: str,
    results: List[dict],
    top_k: int = 20,
    prefer_generic: bool = True,
) -> List[dict]:
    if not results:
        return []

    candidates = results
    if prefer_generic:
        generic = [r for r in results if r.get("data_type") != "Branded"]
        if generic:
            candidates = generic

    descriptions = [item["description"] for item in candidates]
    similarities = run_inference(
        "semantic",
        {"text": query, "descriptions": descriptions},
    )

    ranked_candidates = []
    for result, score in zip(candidates, similarities):
        item = result.copy()
        item["semantic_score"] = float(score)
        ranked_candidates.append(item)

    ranked_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    return ranked_candidates[:top_k]


stemmer = SnowballStemmer("english")


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "with",
    "as",
    "to",
    "in",
    "for",
    "from",
    "by",
    "grade",
    "nfs",
    "ns",
    "ingredient",
}


def normalize_rank_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [stemmer.stem(t) for t in text.split() if t not in STOP_WORDS]


def normalize_match_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


PREPARATION_TERMS = {
    "raw": {"raw", "uncooked"},
    "grilled": {"grilled", "grill", "broiled", "broil"},
    "boiled": {"boiled", "boil"},
    "baked": {"baked", "bake"},
    "fried": {"fried", "fry"},
    "roasted": {"roasted", "roast"},
    "sauteed": {"sauteed", "saute"},
    "stewed": {"stewed", "stew"},
    "steamed": {"steamed", "steam"},
    "cooked": {"cooked", "cook"},
}


def extract_preparations(text):
    words = set(normalize_match_text(text).split())
    detected = set()
    for preparation, variants in PREPARATION_TERMS.items():
        if words & variants:
            detected.add(preparation)
    return detected


def preparation_features(query, description):
    query_preps = extract_preparations(query)
    desc_preps = extract_preparations(description)

    if not query_preps or not desc_preps:
        return 0.0, 0.0

    if "cooked" in query_preps:
        cooked_methods = {
            "cooked",
            "grilled",
            "boiled",
            "baked",
            "fried",
            "roasted",
            "sauteed",
            "stewed",
            "steamed",
        }
        if desc_preps & cooked_methods:
            return 1.0, 0.0
        if "raw" in desc_preps:
            return 0.0, 1.0

    if query_preps & desc_preps:
        return 1.0, 0.0

    return 0.0, 1.0


COMPOSITE_DISH_WORDS = {
    "sandwich",
    "salad",
    "soup",
    "casserole",
    "pizza",
    "burrito",
    "wrap",
    "mixture",
    "meal",
    "dish",
    "lomi",
}


def composite_food_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0
    if "and" in d_words and "and" not in q_words:
        penalty = 1.0
    if "with" in d_words and "with" not in q_words:
        penalty = 1.0
    if (d_words & COMPOSITE_DISH_WORDS) - (q_words & COMPOSITE_DISH_WORDS):
        penalty = 1.0
    return penalty


def specific_part_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0

    # whole egg vs egg white / yolk
    if (
        ({"egg", "eggs"} & q_words)
        and ({"white", "yolk"} & d_words)
        and not ({"white", "yolk"} & q_words)
    ):
        penalty = 1.0

    if "skin" in d_words and "skin" not in q_words:
        penalty = max(penalty, 0.75)

    return penalty


def modifier_conflict_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0

    if "plain" in q_words:
        flavored_words = {
            "flavored",
            "flavoured",
            "vanilla",
            "strawberry",
            "chocolate",
            "fruit",
            "sweetened",
        }
        if d_words & flavored_words:
            penalty = 1.0

    if "rice" in q_words and "white" in q_words:
        if {"brown", "wild", "black"} & d_words:
            penalty = 1.0

    return penalty


PROCESSED_FOOD_WORDS = {
    "roll",
    "patty",
    "patties",
    "nugget",
    "nuggets",
    "luncheon",
    "deli",
    "breaded",
    "battered",
    "processed",
    "sausage",
    "frankfurter",
}


def processed_food_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())
    return 1.0 if (d_words & PROCESSED_FOOD_WORDS) - q_words else 0.0


def unrequested_modifier_penalty(query, description):
    q_text = normalize_match_text(query)
    d_text = normalize_match_text(description)
    q_words = set(q_text.split())

    modifier_phrases = [
        "nonfat",
        "non fat",
        "lowfat",
        "low fat",
        "fat free",
        "reduced fat",
        "skim",
        "sweetened",
        "flavored",
        "flavoured",
    ]
    for modifier in modifier_phrases:
        if modifier in d_text and modifier not in q_text:
            return 1.0

    if "yogurt" in q_words and "plain" in q_words:
        special_yogurt_terms = [
            "nonfat",
            "non fat",
            "lowfat",
            "low fat",
            "fat free",
            "reduced fat",
            "skim",
        ]
        if any(term in d_text for term in special_yogurt_terms):
            return 1.0

    return 0.0


def generic_preparation_bonus(query, description):
    q = normalize_match_text(query)
    d = normalize_match_text(description)

    if "oatmeal" in q and "oatmeal" in d:
        return 1.0
    if "whole eggs" in q and "egg" in d and "whole" in d:
        return 1.0
    if "salmon" in q and "salmon" in d:
        return 1.0
    return 0.0


def dynamic_rerank(query: str, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return []

    query_token_list = normalize_rank_tokens(query)
    query_tokens = set(query_token_list)
    normalized_query = normalize_match_text(query)

    cross_raw_scores = np.asarray(
        run_inference(
            "rerank",
            {
                "text": query,
                "descriptions": [c["description"] for c in candidates],
            },
        ),
        dtype=float,
    ).reshape(-1)
    cross_scores = 1.0 / (1.0 + np.exp(-np.clip(cross_raw_scores, -20, 20)))

    semantic_raw_scores = np.asarray(
        [c["semantic_score"] for c in candidates], dtype=float
    )
    semantic_scores = np.clip((semantic_raw_scores + 1.0) / 2.0, 0.0, 1.0)

    query_coverages, candidate_precisions, fuzzy_scores = [], [], []
    leading_matches, phrase_matches = [], []
    preparation_matches, preparation_conflicts = [], []
    composite_penalties, part_penalties, modifier_penalties = [], [], []
    processed_penalties, unrequested_modifier_penalties = [], []
    generic_preparation_bonuses = []

    for candidate in candidates:
        description = candidate["description"]
        description_token_list = normalize_rank_tokens(description)
        description_tokens = set(description_token_list)
        common_tokens = query_tokens & description_tokens

        query_coverages.append(len(common_tokens) / max(len(query_tokens), 1))
        candidate_precisions.append(
            len(common_tokens) / max(len(description_tokens), 1)
        )
        fuzzy_scores.append(
            fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0
        )

        leading_matches.append(
            float(description_token_list[0] in query_tokens)
            if description_token_list
            else 0.0
        )
        phrase_matches.append(
            float(normalized_query in normalize_match_text(description))
        )

        prep_match, prep_conflict = preparation_features(query, description)
        preparation_matches.append(prep_match)
        preparation_conflicts.append(prep_conflict)

        composite_penalties.append(composite_food_penalty(query, description))
        part_penalties.append(specific_part_penalty(query, description))
        modifier_penalties.append(modifier_conflict_penalty(query, description))
        processed_penalties.append(processed_food_penalty(query, description))
        unrequested_modifier_penalties.append(
            unrequested_modifier_penalty(query, description)
        )
        generic_preparation_bonuses.append(
            generic_preparation_bonus(query, description)
        )

    final_results = []
    for i, candidate in enumerate(candidates):
        final_score = (
            0.32 * cross_scores[i]
            + 0.22 * semantic_scores[i]
            + 0.16 * query_coverages[i]
            + 0.10 * candidate_precisions[i]
            + 0.08 * fuzzy_scores[i]
            + 0.07 * leading_matches[i]
            + 0.05 * phrase_matches[i]
        )
        final_score += 0.10 * preparation_matches[i]
        final_score += 0.06 * generic_preparation_bonuses[i]

        final_score -= 0.22 * preparation_conflicts[i]
        final_score -= 0.20 * composite_penalties[i]
        final_score -= 0.18 * part_penalties[i]
        final_score -= 0.15 * modifier_penalties[i]
        final_score -= 0.20 * processed_penalties[i]
        final_score -= 0.10 * unrequested_modifier_penalties[i]

        final_score = float(np.clip(final_score, 0.0, 1.0))

        result = candidate.copy()
        result.update(
            {
                "semantic_score": round(float(semantic_raw_scores[i]), 4),
                "cross_encoder_score": round(float(cross_raw_scores[i]), 4),
                "query_coverage": round(float(query_coverages[i]), 4),
                "candidate_precision": round(float(candidate_precisions[i]), 4),
                "lexical_score": round(float(fuzzy_scores[i]), 4),
                "leading_match": round(float(leading_matches[i]), 4),
                "phrase_match": round(float(phrase_matches[i]), 4),
                "preparation_match": round(float(preparation_matches[i]), 4),
                "preparation_conflict": round(float(preparation_conflicts[i]), 4),
                "composite_penalty": round(float(composite_penalties[i]), 4),
                "part_penalty": round(float(part_penalties[i]), 4),
                "modifier_penalty": round(float(modifier_penalties[i]), 4),
                "processed_penalty": round(float(processed_penalties[i]), 4),
                "unrequested_modifier_penalty": round(
                    float(unrequested_modifier_penalties[i]), 4
                ),
                "generic_preparation_bonus": round(
                    float(generic_preparation_bonuses[i]), 4
                ),
                "final_score": round(float(final_score), 4),
            }
        )
        final_results.append(result)

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results
