"""Evaluate predictions aligned by index with the bundled dataset.

The default NLP scope evaluates intents, entity sets, first food names,
and first quantities. Full scope also evaluates calorie and recommendation
fields; use it with predictions from those stages. Files marked with
evaluation_scope="nlp" are rejected in full scope.
Write artifacts/evaluation/evaluation_results.json under the repository root.
Without --predictions, use the bundled sample predictions for a smoke test.

Run from the nutrition-ai repository root:
    python -m evaluation.run_evaluation --scope nlp
    python -m evaluation.run_evaluation --predictions artifacts/evaluation/predictions_real.json
"""

import argparse
import json
from pathlib import Path

from app.paths import EVALUATION_DATA_DIR, EVALUATION_OUTPUT_DIR
from evaluation.metrics import (
    calorie_error,
    entity_set_f1,
    food_name_exact_match_accuracy,
    intent_accuracy,
    quantity_extraction_accuracy,
    recommendation_restriction_compliance_at_k,
)

DATASET_PATH = EVALUATION_DATA_DIR / "dataset.json"
SAMPLE_PRED_PATH = EVALUATION_DATA_DIR / "predictions_sample.json"
OUT_PATH = EVALUATION_OUTPUT_DIR / "evaluation_results.json"


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_inputs(examples, predictions, scope: str = "nlp"):
    """Build the parallel lists each metric function needs."""
    intents_pred, intents_gold = [], []
    ner_pred, ner_gold = [], []
    food_pred, food_gold = [], []
    qty_pred, qty_gold = [], []
    cal_pred, cal_gold = [], []
    recs_pred, forb_gold = [], []

    for ex, pred in zip(examples, predictions):
        intents_pred.append(pred.get("intent", ""))
        intents_gold.append(ex["intent"])

        ner_pred.append({(text, label) for text, label in pred.get("entities", [])})
        ner_gold.append({(text, label) for text, label in ex["entities"]})

        if ex["foods"]:
            food_pred.append(pred.get("food_name", ""))
            food_gold.append(ex["foods"][0])

        if ex["quantities"]:
            first_q = pred.get("quantity", [0.0, ""])
            qty_pred.append((float(first_q[0]), str(first_q[1])))
            qty_gold.append(
                (float(ex["quantities"][0][0]), str(ex["quantities"][0][1]))
            )

        if scope == "full" and ex["calories_gold"] is not None:
            cal_pred.append(float(pred.get("calories", 0.0)))
            cal_gold.append(float(ex["calories_gold"]))

        if scope == "full" and ex["intent"] == "meal_recommendation":
            recs_pred.append(pred.get("recommended_foods", []))
            # Forbidden foods = anything tagged RESTRICTION in gold entities
            forb_gold.append(
                {text for text, label in ex["entities"] if label == "RESTRICTION"}
            )

    return {
        "intents": (intents_pred, intents_gold),
        "ner": (ner_pred, ner_gold),
        "food": (food_pred, food_gold),
        "qty": (qty_pred, qty_gold),
        "cal": (cal_pred, cal_gold),
        "rec": (recs_pred, forb_gold),
    }


def run(predictions_path: str, scope: str = "nlp") -> dict:
    if scope not in {"nlp", "full"}:
        raise ValueError("scope must be 'nlp' or 'full'.")

    dataset = _load_json(DATASET_PATH)["examples"]
    prediction_document = _load_json(predictions_path)
    if scope == "full" and prediction_document.get("evaluation_scope") == "nlp":
        raise ValueError("NLP-only predictions cannot be evaluated with scope='full'.")
    predictions = prediction_document["predictions"]

    if len(predictions) != len(dataset):
        raise ValueError(
            f"predictions has {len(predictions)} entries but dataset has "
            f"{len(dataset)}. They must be aligned by index."
        )

    inputs = _extract_metric_inputs(dataset, predictions, scope)

    results = {
        "intent_accuracy": intent_accuracy(*inputs["intents"]),
        "ner": entity_set_f1(*inputs["ner"]),
        "food_name_exact_match_accuracy": food_name_exact_match_accuracy(
            *inputs["food"]
        ),
        "quantity_extraction_accuracy": quantity_extraction_accuracy(
            *inputs["qty"], tolerance=0.05
        ),
        "calorie_error": calorie_error(*inputs["cal"]) if scope == "full" else None,
        "recommendation_restriction_compliance_at_k": (
            recommendation_restriction_compliance_at_k(*inputs["rec"], k=5)
            if scope == "full"
            else None
        ),
        "evaluation_scope": scope,
        "n_examples": len(dataset),
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def _print_table(results: dict) -> None:
    print("=" * 70)
    print("Nutrition AI — Evaluation Results")
    print("=" * 70)
    print(f"Examples evaluated: {results['n_examples']}")
    print(f"Evaluation scope: {results['evaluation_scope']}")
    print()
    print(f"  Intent accuracy ................ {results['intent_accuracy']:.4f}")
    print(f"  Entity-set precision ........... {results['ner']['precision']:.4f}")
    print(f"  Entity-set recall .............. {results['ner']['recall']:.4f}")
    print(f"  Entity-set F1 .................. {results['ner']['f1']:.4f}")
    print(
        f"  Food matching accuracy ......... {results['food_name_exact_match_accuracy']:.4f}"
    )
    print(
        f"  Quantity extraction accuracy .. {results['quantity_extraction_accuracy']:.4f}"
    )
    if results["calorie_error"] is None:
        print("  Calorie error ................. N/A (NLP-only evaluation)")
    else:
        print(
            f"  Calorie MAE .................... {results['calorie_error']['mae']:.2f}"
        )
        print(
            f"  Calorie MAPE ................... {results['calorie_error']['mape']:.4f}"
        )

    if results["recommendation_restriction_compliance_at_k"] is None:
        print("  Recommendation compliance@5 ... N/A (NLP-only evaluation)")
    else:
        print(
            "  Recommendation compliance@5 ... "
            f"{results['recommendation_restriction_compliance_at_k']:.4f}"
        )
    print()
    print(f"Results written to: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=SAMPLE_PRED_PATH,
        help="Path to predictions JSON (aligned by index with dataset.json).",
    )
    parser.add_argument(
        "--scope",
        choices=("nlp", "full"),
        default="nlp",
        help="nlp: parser metrics only; full: also calorie and recommendation metrics.",
    )
    args = parser.parse_args()

    res = run(args.predictions, scope=args.scope)
    _print_table(res)
