> Historical report from an earlier implementation.
> These results have notbeen rerun for the current service.
> Similarity scores alone do not establish factual accuracy.

# Task 5 — Evaluation Report

**Project:** AI Calorie & Nutrition Assistant (Agentic RAG)
**Evaluation date:** 2026-09-17
**Methodology:** Live NLP parser run on a 30-example labeled dataset + full pytest suite.

---

## 1. Test Suite Results
pytest tests/ test_nutrition_engine.py -v

============= 156 passed, 1 warning in 7.27s=================

| File | Coverage |
|---|---|
| `test_nutrition_engine.py` | Task 3 unit tests (~60) |
| `test_evaluation_metrics.py` | Metric functions + dataset sanity (~30) |
| `test_integration.py` | 10 end-to-end scenarios (mocked LLM/USDA) |
| `test_orchestrator.py` | Intent routing + status gate |
| `test_tools.py` | 7 tool envelopes |
| `test_rag.py` | RAG quality gate |

**Result:** 156 / 156 passing (100%).

---

## 2. Live Evaluation on 30 Examples

The NLP parser was run on the 30-example labeled dataset
(`evaluation/dataset.json`) and its predictions compared against the ground truth.
python -m evaluation.run_real_predictions
python -m evaluation.run_evaluation --predictions evaluation/predictions_real.json

| Metric | Value | Interpretation |
|---|---|---|
| **Intent accuracy** | **1.0000** | Perfect classification across EN / AR / mixed |
| **NER precision** | 0.6706 | Entity extraction precision |
| **NER recall** | 0.7403 | Entity extraction recall |
| **NER F1** | **0.7037** | Solid for multilingual food NER |
| **Food matching accuracy** | 0.5833 | Arabic normalization variance (بيضة vs بيض) |
| **Quantity extraction** | **1.0000** | All quantities correctly parsed |
| Calorie MAE | N/A | Computed by Task 2+3, not Task 1 |
| Recommendation P@5 | N/A | Computed by Task 4, not Task 1 |

---

## 3. Discussion Points

**Strengths:**
1. Intent classification is perfect on the test set — the extended prompt with
   standalone-goal few-shot examples solved the earlier misclassification.
2. Quantity extraction is perfect — numbers + units are consistently parsed
   across all three languages.
3. NER F1 of 0.70 is competitive for a multilingual parser where code-switching
   and dialect variation are common.

**Limitations:**
1. **Dataset size (30 examples)** — metrics are indicative, not statistically
   significant.
2. **Food matching (0.58)** — Arabic morphemes (بيضة vs بيض, ال + noun) cause
   variance that a normalization layer could reduce.
3. **LLM non-determinism** — re-running may produce slightly different numbers.

**Mitigations already in place:**
- Deterministic Arabic normalization (`normalize_arabic_text`)
- Token-boundary restriction matching
- Two-signal quality gate on USDA retrieval (final_score + cross_encoder_score)
- `MIN_SAFE_CALORIES` floor on the nutrition engine