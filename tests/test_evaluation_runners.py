import json

import pytest

from app.paths import EVALUATION_DATA_DIR
from evaluation import rag_evaluation, run_evaluation


def test_nlp_scope_does_not_score_unavailable_outputs(tmp_path, monkeypatch, capsys):
    document = json.loads(
        (EVALUATION_DATA_DIR / "predictions_sample.json").read_text(encoding="utf-8")
    )
    document["evaluation_scope"] = "nlp"
    for prediction in document["predictions"]:
        prediction.pop("calories", None)
        prediction.pop("recommended_foods", None)

    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "results.json"
    monkeypatch.setattr(run_evaluation, "OUT_PATH", str(output))

    result = run_evaluation.run(str(predictions))

    assert result["intent_accuracy"] == 1.0
    assert result["calorie_error"] is None
    assert result["recommendation_restriction_compliance_at_k"] is None
    assert json.loads(output.read_text())["calorie_error"] is None

    run_evaluation._print_table(result)
    assert "N/A (NLP-only evaluation)" in capsys.readouterr().out


def test_full_scope_scores_complete_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, "OUT_PATH", str(tmp_path / "results.json"))

    result = run_evaluation.run(run_evaluation.SAMPLE_PRED_PATH, scope="full")

    assert result["calorie_error"] == {"mae": 0.0, "mape": 0.0}
    assert result["recommendation_restriction_compliance_at_k"] == 1.0


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be"):
        run_evaluation.run("unused.json", scope="invalid")


def test_full_scope_rejects_nlp_only_predictions(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps({"evaluation_scope": "nlp", "predictions": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NLP-only"):
        run_evaluation.run(str(predictions), scope="full")


def test_rag_summary_counts_failed_requests(monkeypatch):
    dataset = [
        {"question": "first", "reference": "answer", "language": "english"},
        {"question": "second", "reference": "answer", "language": "english"},
    ]
    monkeypatch.setattr(rag_evaluation, "RAG_DATASET", dataset)
    monkeypatch.setattr(
        rag_evaluation,
        "retrieve_fitness_knowledge",
        lambda q: (
            {"success": True, "data": {"passages": ["text"]}}
            if q == "first"
            else {"success": False, "error_code": "low_confidence_retrieval"}
        ),
    )
    monkeypatch.setattr(rag_evaluation, "call_rag_answer", lambda *args: "answer")
    monkeypatch.setattr(rag_evaluation, "compute_bleu", lambda *args: 0.8)
    monkeypatch.setattr(
        rag_evaluation,
        "compute_rouge",
        lambda *args: {"rouge1": 0.6, "rouge2": 0.4, "rougeL": 0.5},
    )
    monkeypatch.setattr(
        rag_evaluation,
        "compute_bertscore",
        lambda *args: [{"precision": 0.8, "recall": 0.7, "f1": 0.9}],
    )

    summary = rag_evaluation.run()["summary"]

    assert summary["n_examples"] == 2
    assert summary["n_successful"] == 1
    assert summary["answer_success_rate"] == 0.5
    assert summary["bleu"] == 0.4
    assert summary["rougeL"] == 0.25
    assert summary["bertscore_f1"] == 0.45


@pytest.mark.parametrize("text", ["وجبة متوازنة", "balanced meal"])
def test_rouge_identical_text_scores_one(text):
    pytest.importorskip("rouge_score")

    result = rag_evaluation.compute_rouge(text + "!", text)

    assert result == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}


def test_rouge_unrelated_arabic_scores_zero():
    pytest.importorskip("rouge_score")

    result = rag_evaluation.compute_rouge("تمرين قوي", "وجبة متوازنة")

    assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
