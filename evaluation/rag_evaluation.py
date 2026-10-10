"""
rag_evaluation.py — BLEU + ROUGE + BERTScore evaluation for RAG answers.

Evaluates the RAG flow (fitness_nutrition_question intent) against a small
reference set of question-answer pairs.

Usage:
    python -m evaluation.rag_evaluation
    python -m evaluation.rag_evaluation --output artifacts/evaluation/rag_results.json
"""

import argparse
import json
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from app.ai.agent.llm import call_rag_answer
from app.ai.rag.retrieve import retrieve_fitness_knowledge
from app.paths import EVALUATION_OUTPUT_DIR

# ─── Reference QA pairs for RAG evaluation ──────────────────────────────────
RAG_DATASET = [
    {
        "question": "What should I eat before a workout?",
        "reference": (
            "Two to three hours before exercise, eat a meal with moderate "
            "carbohydrates, moderate protein, and low fat. A small snack 30 to "
            "60 minutes before training should be easy to digest, such as a "
            "banana, a rice cake with peanut butter, or yogurt with fruit. "
            "Avoid high-fat and high-fiber foods immediately before training."
        ),
        "language": "english",
    },
    {
        "question": "What should I eat after a workout?",
        "reference": (
            "After exercise, aim for 20 to 40 grams of protein and a moderate "
            "amount of carbohydrates. Good choices include grilled chicken "
            "with rice and vegetables, tuna with whole wheat pasta, Greek "
            "yogurt with fruit and granola, or a protein shake with a banana."
        ),
        "language": "english",
    },
    {
        "question": "How much protein do I need per day?",
        "reference": (
            "The general recommendation for active adults is 1.6 to 2.2 grams "
            "of protein per kilogram of body weight per day. Protein is "
            "essential for building and repairing muscles, producing enzymes, "
            "and supporting the immune system."
        ),
        "language": "english",
    },
    {
        "question": "How much water should I drink during exercise?",
        "reference": (
            "Drink 400 to 600 ml of water in the two hours before exercise. "
            "During exercise, drink 150 to 250 ml every 15 to 20 minutes. "
            "After exercise, drink enough water to replace sweat losses."
        ),
        "language": "english",
    },
    {
        "question": "What are the best sources of protein?",
        "reference": (
            "Good sources of protein include chicken, fish, eggs, legumes, and "
            "dairy. Protein is essential for building and repairing muscles, "
            "producing enzymes, and supporting the immune system."
        ),
        "language": "english",
    },
    {
        "question": "أكل إيه قبل التمرين؟",
        "reference": (
            "قبل التمرين بساعتين إلى ثلاث ساعات، تناول وجبة تحتوي على "
            "كربوهيدرات معتدلة وبروتين معتدل ودهون قليلة. وجبة خفيفة قبل "
            "30 إلى 60 دقيقة من التدريب يجب أن تكون سهلة الهضم مثل موزة أو "
            "خبز أرز مع زبدة الفول السوداني أو زبادي مع فاكهة. تجنب الأطعمة "
            "الغنية بالدهون والألياف مباشرة قبل التدريب."
        ),
        "language": "arabic",
    },
    {
        "question": "أكل إيه بعد التمرين؟",
        "reference": (
            "بعد التمرين، استهدف 20 إلى 40 جرامًا من البروتين وكمية معتدلة من "
            "الكربوهيدرات. خيارات جيدة تشمل دجاج مشوي مع الأرز والخضروات، "
            "أو تونة مع مكرونة القمح الكامل، أو زبادي يوناني مع الفاكهة، أو "
            "شيك بروتين مع موزة."
        ),
        "language": "arabic",
    },
    {
        "question": "كم بروتين أحتاج في اليوم؟",
        "reference": (
            "التوصية العامة للبالغين النشطين هي من 1.6 إلى 2.2 جرام من "
            "البروتين لكل كيلوجرام من وزن الجسم يوميًا. البروتين ضروري لبناء "
            "وإصلاح العضلات وإنتاج الإنزيمات ودعم المناعة."
        ),
        "language": "arabic",
    },
    {
        "question": "كم ماء أشرب أثناء التمرين؟",
        "reference": (
            "اشرب 400 إلى 600 مل من الماء خلال الساعتين قبل التمرين. أثناء "
            "التمرين، اشرب 150 إلى 250 مل كل 15 إلى 20 دقيقة. بعد التمرين، "
            "اشرب ما يكفي لتعويض فقدان العرق."
        ),
        "language": "arabic",
    },
    {
        "question": "إيه أفضل مصادر البروتين؟",
        "reference": (
            "مصادر البروتين الجيدة تشمل الدجاج والسمك والبيض والبقوليات "
            "ومنتجات الألبان. البروتين ضروري لبناء وإصلاح العضلات وإنتاج "
            "الإنزيمات ودعم المناعة."
        ),
        "language": "arabic",
    },
]


def _tokenize(text: str) -> list:
    """Preserve Unicode words and separate punctuation."""
    import re

    return re.findall(r"\w+", text.lower())


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU-4 with smoothing (Chen & Cherry 2014)."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = [_tokenize(reference)]
    if not hyp_tokens or not ref_tokens[0]:
        return 0.0
    smoother = SmoothingFunction().method1
    return float(sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother))


class _RougeTokenizer:
    def tokenize(self, text: str) -> list:
        return _tokenize(text)


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F-measures."""
    import re

    from rouge_score import rouge_scorer

    has_arabic = bool(re.search(r"[\u0600-\u06FF]", hypothesis + reference))
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=not has_arabic,
        tokenizer=_RougeTokenizer() if has_arabic else None,
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bertscore(hypotheses: list, references: list) -> list:
    """
    BERTScore (P/R/F1) using a multilingual model.
    Returns list of dicts with keys: precision, recall, f1.
    """
    from app.ai.retrieval.runtime import prepare_transformers

    prepare_transformers()

    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        hypotheses,
        references,
        lang="en",  # ignored when model_type is multilingual
        model_type="bert-base-multilingual-cased",
        verbose=False,
        batch_size=8,
    )
    return [
        {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        }
        for p, r, f1 in zip(P, R, F1)
    ]


def run() -> dict:
    results = []
    hypotheses = []
    references = []

    print("=" * 70)
    print("RAG Evaluation — BLEU + ROUGE + BERTScore")
    print("=" * 70)

    # ── Phase 1: Generate all answers ───────────────────────────────────
    for i, ex in enumerate(RAG_DATASET, 1):
        q = ex["question"]
        ref = ex["reference"]
        print(f"\n[{i}/{len(RAG_DATASET)}] {q[:60]}")

        rag = retrieve_fitness_knowledge(q)
        if not rag["success"]:
            print(f"   Retrieval failed: {rag.get('error_code')}")
            results.append(
                {
                    "question": q,
                    "language": ex["language"],
                    "retrieval_success": False,
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bertscore_f1": 0.0,
                    "answer": None,
                }
            )
            continue

        passages = rag["data"]["passages"]
        answer = call_rag_answer(q, passages, ex["language"])
        if not answer:
            print("   Answer generation failed")
            results.append(
                {
                    "question": q,
                    "language": ex["language"],
                    "retrieval_success": True,
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bertscore_f1": 0.0,
                    "answer": None,
                }
            )
            continue

        bleu = compute_bleu(answer, ref)
        rouge = compute_rouge(answer, ref)
        print(f"   BLEU={bleu:.4f}  ROUGE-L={rouge['rougeL']:.4f}")

        hypotheses.append(answer)
        references.append(ref)
        results.append(
            {
                "question": q,
                "language": ex["language"],
                "retrieval_success": True,
                "bleu": round(bleu, 4),
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bertscore_f1": 0.0,  # filled below
                "answer": answer,
            }
        )

    # ── Phase 2: Batch BERTScore ────────────────────────────────────────
    print("\nComputing BERTScore (multilingual)...")
    if hypotheses:
        bs_scores = compute_bertscore(hypotheses, references)
        idx = 0
        for r in results:
            if r["answer"]:
                r["bertscore_f1"] = bs_scores[idx]["f1"]
                r["bertscore_precision"] = bs_scores[idx]["precision"]
                r["bertscore_recall"] = bs_scores[idx]["recall"]
                idx += 1

    # ── Aggregate ───────────────────────────────────────────────────────
    valid = [r for r in results if r["answer"]]
    n = len(results) or 1

    summary = {
        "n_examples": len(RAG_DATASET),
        "n_successful": len(valid),
        "answer_success_rate": round(len(valid) / n, 4),
        "bleu": round(sum(r["bleu"] for r in valid) / n, 4),
        "rouge1": round(sum(r["rouge1"] for r in valid) / n, 4),
        "rouge2": round(sum(r["rouge2"] for r in valid) / n, 4),
        "rougeL": round(sum(r["rougeL"] for r in valid) / n, 4),
        "bertscore_f1": round(sum(r["bertscore_f1"] for r in valid) / n, 4),
        "bertscore_precision": round(
            sum(r.get("bertscore_precision", 0) for r in valid) / n, 4
        ),
        "bertscore_recall": round(
            sum(r.get("bertscore_recall", 0) for r in valid) / n, 4
        ),
    }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(EVALUATION_OUTPUT_DIR / "rag_results.json"),
    )
    args = parser.parse_args()

    from app.config import load_environment

    load_environment(".env")
    output = run()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {destination.resolve()}")
