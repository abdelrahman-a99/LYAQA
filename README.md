# Nutrition AI Service

A stateless FastAPI service for Arabic and English nutrition conversations.

It parses messages, retrieves USDA nutrients on demand, calculates calories
and macros, generates and validates meals, and answers fitness questions
using a RAG knowledge base.

The calling backend owns accounts, conversations, message storage, and
profile persistence. Each request supplies its processing context.

See [the pipeline guide](docs/pipeline.md) for routing, tools, retrieval,
validation, and evaluation boundaries.

## Setup

Use Python 3.10–3.12. These commands use Windows PowerShell and Python 3.12.
Run them from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For development, `requirements-dev.txt` installs runtime dependencies
plus testing, evaluation, and developer tools.

For a runtime-only environment, install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No editable installation or wheel is required.

Keep credentials in a root `.env` file or set process environment variables.

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY_NLP` | Parser key and fallback for the other Groq clients. |
| `GROQ_API_KEY_MEALS` | Optional dedicated meal-generation and revision key. |
| `GROQ_API_KEY_RAG` | Optional dedicated knowledge-answer and general-chat key. |
| `USDA_API_KEY` | Key for USDA requests. |
| `USDA_BASE_URL` | Optional override of `https://api.nal.usda.gov/fdc/v1`. |
| `NUTRITION_AI_RAG_DIR` | Optional directory containing matching index and chunks files. |

The HTTP service loads `.env` at startup. Existing process environment
values win. Retrieval models load on demand and may download weights
on first use.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open [Swagger](http://127.0.0.1:8001/docs).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness; does not check providers or models. |
| POST | `/v1/chat` | Process a message with optional profile and history. |

Example request:

```json
{
  "message": "My weight is 82 kg",
  "profile": {
    "birth_date": "1990-01-01",
    "gender": "male",
    "weight": 80,
    "height": 180,
    "activity_level": "moderate",
    "goal": "weight_loss"
  },
  "history": []
}
```

The response contains `response`, `intent`, `status`, `language`, `profile`,
`revision_count`, and `meals`. The backend persists accepted profile changes
and messages.

`meals` contains up to four final meal candidates, copied from calculation results.
Each snapshot includes an ID, ingredients and quantities, calculated nutrition,
calorie and protein targets, validation violations, and partial lookup failures.
The calculated food details retain quantity-conversion sources.
A generated candidate can have `valid: false`; an unsuccessful generation produces
no snapshot. For daily plans, validity is per meal, not a guarantee of a complete
or valid day. Successful edits return the complete supplied meal group with
one updated meal and its original ID; unrelated turns return `meals: []`.

History contains prior user/assistant messages with `role` and `content`,
excluding the current message. Each request creates a fresh assistant.

The optional `meal_context` field accepts up to four previously returned meal
snapshots with distinct IDs. The backend supplies its conversation's `last_meals`
through this field. Omit it or send `[]` when no meals are available.
Meal context is separate from text history and current-turn `meals` output.
An explicit `meal_edit` request selects one previous meal and applies validated
food changes. It retrieves and recalculates all ingredients of the edited meal;
supplied nutrient values are not used as the source of fresh calculations.
The original calorie target and protein minimum are preserved unless the current
message explicitly supplies a new calorie target. Current profile restrictions
are checked. Other meals are returned unchanged, not revalidated as a full day.

Selection ambiguity returns `needs_clarification`; missing context returns
`needs_meal_context`; lookup or calculation failure returns
`meal_edit_failed`. Model and USDA operational failures return structured HTTP errors. Failed validation returns `meal_edit_rejected` with calculated
values and reasons in the text. These paths return `meals: []`, preserving the
backend's previous group. A valid edit returns the complete updated group with
the edited meal's original ID. Editing is one attempt without automatic scaling
or nutritional revisions, preserving explicitly requested quantities.

Messages are limited to 4000 characters. History accepts 20 entries of up
to 10000 characters each; the parser uses only the last six entries.

General-chat and RAG answer generation use the last six prior messages.
When history is present, RAG first resolves the current request into a standalone
search question using an additional Groq call. Unclear references trigger a
clarification request with `needs_clarification`. Low-confidence retrieval returns
`insufficient_knowledge`. Provider and RAG execution failures return HTTP errors.
Retrieved passages remain the source for factual RAG answers.
The chatbot backend keeps the latest nonempty meal batch per conversation.
An empty batch preserves the previous one. Durable storage and service
authentication remain future work.

Invalid request data returns 422. HTTP 200 does not certify a valid meal
or complete food retrieval; inspect meal validation and failure metadata.
HTTP failures include `detail`, `error_code`, and `retryable`.
LLM authentication, rate limits, timeouts, rejected requests, and malformed
outputs have distinct codes. The backend preserves approved processing codes
without forwarding upstream diagnostics. Failed HTTP turns do not update
existing conversation context. `retryable` does not trigger automatic replay.

USDA operational failures return structured HTTP errors with these codes:
`usda_configuration_error`, `usda_authentication_failed`,
`usda_request_rejected`, `usda_invalid_response`, `usda_timeout`,
`usda_rate_limited`, and `usda_unavailable`.

A successful empty search, a missing detail record, and incomplete nutrient
data remain ordinary food lookup outcomes. Operational failures abort the
turn and are not cached.

The backend preserves recognized USDA error codes and leaves the previous
profile, history, and meals unchanged after a failed HTTP turn.
`retryable` does not trigger automatic replay.

Food retrieval model groups are cached only after successful initialization.
Concurrent first requests share one initialization per model group within
each process.

Expected initialization failures return HTTP 503 with
`error_code: retrieval_model_unavailable` and `retryable: false`.
Check dependencies, model files, download access, and available memory before
retrying. Failed HTTP turns leave the backend's saved context unchanged.

## Repository layout

| Path | Contents |
| --- | --- |
| `app/main.py` | FastAPI application and startup. |
| `app/api/` | HTTP routes and request/response schemas. |
| `app/ai/` | AI processing, orchestration, and nutrition logic. |
| `app/config.py`, `app/paths.py` | Environment loading and resource paths. |
| `evaluation/` | Metrics and evaluation runners. |
| `scripts/` | Developer inspection utilities. |
| `tests/` | Automated tests. |
| `data/` | Knowledge documents and evaluation inputs. |
| `artifacts/rag/` | Tracked runtime index and matching chunks. |
| `artifacts/evaluation/` | Generated evaluation output. |
| `docs/` | Pipeline guide and historical reports. |

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check app evaluation scripts tests
.\.venv\Scripts\python.exe -m ruff format --check app evaluation scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Tests mock model and API boundaries where required. Passing tests does
not establish live provider availability or model accuracy.

## Evaluation

Evaluation commands require the dependencies in `requirements-dev.txt`.

Smoke-test the harness using sample predictions:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --scope full
```

Generate independent NLP predictions, then evaluate the saved output:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run_real_predictions
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --predictions artifacts/evaluation/predictions_real.json --scope nlp
```

Run live RAG evaluation after configuring credentials:

```powershell
.\.venv\Scripts\python.exe -m evaluation.rag_evaluation
```

Outputs default to `artifacts/evaluation/`. Prediction rows must match
dataset rows in order. NLP scope does not evaluate generated meals or
calculated calories.

Sample scores verify the harness, not live assistant quality.
Reference-similarity metrics alone do not establish factual accuracy.
Historical reports are not results for the current service.

## RAG assets

Builder and retriever default to `artifacts/rag/` under the repository root.
Relative `NUTRITION_AI_RAG_DIR` values resolve against that root.

Keep `faiss_index.bin` and `chunks.json` from the same build.

After changing documents in `data/knowledge_base/`, rebuild:

```powershell
.\.venv\Scripts\python.exe -m app.ai.rag.build_index
```

For custom builder or evaluator paths, set `NUTRITION_AI_RAG_DIR` in the
process environment before running the command. Configure the same
directory for the HTTP service.

Restart the service after rebuilding its assets.
