> Historical report from an earlier implementation.
> These results have notbeen rerun for the current service.
> Similarity scores alone do not establish factual accuracy.

## 6. RAG Answer Quality — BLEU, ROUGE, BERTScore

Evaluated 10 reference QA pairs (5 English, 5 Arabic) against the live RAG pipeline.
Each generated answer is compared to a human-written reference derived from the
knowledge base.

### 6.1 Aggregate Results

| Metric | Score | Interpretation |
|---|---|---|
| **BLEU-4** | 0.2242 | Moderate — LLM paraphrases rather than copies |
| **ROUGE-1** | 0.4318 | Strong unigram overlap |
| **ROUGE-2** | 0.3687 | Good bigram overlap |
| **ROUGE-L** | 0.4070 | Strong longest-common-subsequence similarity |
| **BERTScore-P** | 0.7913 | Excellent semantic precision |
| **BERTScore-R** | 0.7622 | Excellent semantic recall |
| **BERTScore-F1** | **0.7757** | **Strong semantic similarity** |

### 6.2 Why Three Metrics?

BLEU and ROUGE measure **lexical overlap** — they reward the model for reusing
the same words as the reference. BERTScore measures **semantic similarity** —
it rewards the model for expressing the same *meaning* even with different words.

This distinction matters because RAG answers are **paraphrases** of the source
knowledge, not extracts.

### 6.3 Lexical vs Semantic Divergence

Three examples scored **ROUGE-L = 0.0** but received high BERTScore:

| # | Question | BLEU | ROUGE-L | Reason |
|---|---|---|---|---|
| 6 | أكل إيه قبل التمرين؟ | 0.124 | 0.000 | Arabic tokenization; LLM paraphrases |
| 7 | أكل إيه بعد التمرين؟ | 0.005 | 0.000 | Same |
| 10 | إيه أفضل مصادر البروتين؟ | 0.023 | 0.000 | Same |

**Conclusion:** lexical metrics penalize correct paraphrases. BERTScore
captures the true quality — the LLM answers are semantically correct even when
they use different wording.

### 6.4 Limitations

- **English-centric tokenization:** `rouge_score` uses an English stemmer;
  Arabic morphemes are not normalized (`أكل` vs `الأكل`).
- **Reference bias:** BLEU/ROUGE are sensitive to the exact phrasing of the
  single reference. A better setup would use 2-3 reference answers per question.
- **Model bias:** BERTScore uses `bert-base-multilingual-cased` — solid for
  Arabic and English, but not perfect for dialectal Arabic.
