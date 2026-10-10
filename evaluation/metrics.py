"""Compute evaluation metrics from aligned predictions and reference values.

The functions perform no API calls or file I/O.

Metrics:
    intent_accuracy: Exact intent-label accuracy.
    entity_set_f1: Micro precision, recall, and F1 over (text, label) sets.
    food_name_exact_match_accuracy: Exact normalized food-name matching.
    quantity_extraction_accuracy: Unit match with relative quantity tolerance.
    calorie_error: MAE and MAPE; MAPE is returned as a fraction.
    recommendation_restriction_compliance_at_k: Top-k forbidden-name compliance.

The evaluation runner currently compares the first food and quantity per example.
"""

from typing import List, Set, Tuple

# ─── 1. Intent classification ───────────────────────────────────────────────


def intent_accuracy(predictions: List[str], gold: List[str]) -> float:
    """Fraction of predictions equal to gold. Returns 0.0 on empty input."""
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    return round(correct / len(gold), 4)


# ─── 2. Entity-set precision, recall, and F1. ─────────────────────────


def entity_set_f1(
    predictions: List[Set[Tuple[str, str]]],
    gold: List[Set[Tuple[str, str]]],
) -> dict:
    """Compute micro precision, recall, and F1 from entity-set overlaps.

    Each example contains a set of (text, label) tuples normalized by the caller.
    Counts are summed across examples before computing the scores.
    Token positions and character offsets are not used.
    """
    tp = fp = fn = 0
    for p, g in zip(predictions, gold):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ─── 3. Food matching accuracy ──────────────────────────────────────────────


def food_name_exact_match_accuracy(
    predictions: List[str],
    gold: List[str],
) -> float:
    """
    Exact-match accuracy on normalized (lowercased, stripped) food names.
    """
    if not gold:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, gold) if p.strip().lower() == g.strip().lower()
    )
    return round(correct / len(gold), 4)


# ─── 4. Quantity extraction accuracy ────────────────────────────────────────


def quantity_extraction_accuracy(
    predictions: List[Tuple[float, str]],
    gold: List[Tuple[float, str]],
    tolerance: float = 0.0,
) -> float:
    """Measure unit agreement and relative quantity error against the reference.

    Units are trimmed and lowercased. For a nonzero reference quantity,
    accept abs(predicted - reference) / abs(reference) <= tolerance.
    A zero reference quantity requires a zero prediction.
    """
    if not gold:
        return 0.0
    correct = 0
    for (pq, pu), (gq, gu) in zip(predictions, gold):
        if pu.strip().lower() != gu.strip().lower():
            continue
        if gq == 0:
            if pq == 0:
                correct += 1
        elif abs(pq - gq) / abs(gq) <= tolerance:
            correct += 1
    return round(correct / len(gold), 4)


# ─── 5. Calorie error ───────────────────────────────────────────────────────


def calorie_error(
    predictions: List[float],
    gold: List[float],
) -> dict:
    """
    Mean Absolute Error and Mean Absolute Percentage Error.
    """
    if not gold:
        return {"mae": 0.0, "mape": 0.0}
    abs_errors = [abs(p - g) for p, g in zip(predictions, gold)]
    pct_errors = [abs(p - g) / abs(g) for p, g in zip(predictions, gold) if g != 0]
    mae = sum(abs_errors) / len(abs_errors)
    mape = (sum(pct_errors) / len(pct_errors)) if pct_errors else 0.0
    return {"mae": round(mae, 4), "mape": round(mape, 4)}


# ─── 6. Recommendation restriction compliance@K ──────────────────────────────────────────


def recommendation_restriction_compliance_at_k(
    recommendations: List[List[str]],
    forbidden_sets: List[Set[str]],
    k: int = 5,
) -> float:
    """Average forbidden-name compliance over the first k foods per request.

    Names are compared after trimming and lowercasing. Empty recommendation
    lists score zero. This metric checks literal name membership only;
    it does not evaluate calorie targets, food relevance, or ingredient allergens.
    """
    if not recommendations:
        return 0.0
    scores = []
    for recs, forbidden in zip(recommendations, forbidden_sets):
        top_k = recs[:k]
        if not top_k:
            scores.append(0.0)
            continue
        safe = sum(
            1
            for r in top_k
            if r.strip().lower() not in {f.strip().lower() for f in forbidden}
        )
        scores.append(safe / len(top_k))
    return round(sum(scores) / len(scores), 4)
