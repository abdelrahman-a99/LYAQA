# Nutrition AI pipeline

This guide describes the current implementation. AI code paths below are
relative to `app/ai/`. HTTP files are in `app/main.py` and `app/api/`.
Evaluation paths are relative to the repository root.

## 1. Entry point and parsing

`service.py: NutritionAssistant.respond(message)` validates the message and calls
`agent/orchestrator.py: handle(message, session, state)`.

`handle()` clears per-turn state through `AgentState.reset_turn()`, then calls
`_route_and_answer()`. The conversation identity and profile survive this reset;
calculated targets and intermediate results do not.

The parsing path is:

1. `agent/tools/parsing.py: parse_query()`.
2. `conversation/session.py: NLPSession.process()`.
3. `nlp/parser.py: parse_with_context()` and the Groq parsing boundary.
4. Output normalization, profile merging, and selective onboarding.

The parser extracts intent, foods, quantities, profile data, language, meal type,
and an explicit calorie target when provided. It attempts parsing twice before
returning a normalized fallback. An LLM proposes the intent; Python code selects
the corresponding workflow. The LLM does not freely choose among all tools.

`daily_plan` and `update_profile` trigger onboarding when the profile is incomplete.
The required fields are `birth_date`, `gender`, `weight`, `height`,
`activity_level`, and `goal`. Other flows can request missing information later.

## 2. Intent routing

All handlers are in `agent/flows/`.

| Intent | Handler | Main operations |
| --- | --- | --- |
| `general_chat` or an unknown intent | `chat.py: handle_general_chat` | General LLM response. |
| `update_profile` | Handled directly by the orchestrator | Return the profile confirmation or onboarding prompt. |
| `calculate_calories`, `calculate_macros`, `food_nutrition` | `calories.py: handle_calorie_macro_flow` | USDA lookup, portion conversion, food totals, response formatting. |
| `fitness_nutrition_question`, `food_substitution` | `knowledge.py: handle_rag_flow` | Knowledge retrieval followed by grounded answer generation. |
| `meal_recommendation` | `meals.py: handle_meal_recommendation` | Determine target, generate foods, retrieve nutrition, validate and revise. |
| `daily_plan` | `daily.py: handle_daily_plan` | Calculate daily targets, generate four meals, validate daily totals. |
| `meal_edit` | `meal_edits.py: handle_meal_edit` | Select a previous meal, apply indexed changes, retrieve, calculate, and validate. |

The `calculate_macros` intent reports macros for the supplied foods. The separate
`nutrition/profile.py: calculate_macros()` function computes profile-based daily
macro targets; these are different operations.

## 3. Seven tool interfaces

`agent/tools/__init__.py` exports:

| Tool | Responsibility |
| --- | --- |
| `parse_query` | Parse the turn and update conversation context. |
| `get_nutrition` | Retrieve USDA nutrition for one food. |
| `calculate_targets` | Calculate BMR, TDEE, target calories, and daily macros. |
| `calculate_meal_nutrition` | Calculate totals from foods and retrieved nutrients. |
| `validate_plan` | Run meal or daily validation. |
| `retrieve_fitness_knowledge` | Retrieve knowledge-base passages. |
| `scale_meal` | Adjust quantities toward a calorie target. |

Tool envelopes contain `success`, `data`, `error`, and `error_code`. A successful
wrapper does not necessarily mean its nested meal result is complete or valid.
Unhandled execution errors reach the fallback in `NutritionAssistant.respond()`.

## 4. USDA retrieval and numerical calculation

`agent/food_lookup.py: collect_nutrition_for_foods()` calls `get_nutrition()` for
each food. The retrieval path in `retrieval/service.py: get_food_nutrition()` is:

1. Normalize the food query, including supported Arabic names.
2. Search the USDA API on demand; the full USDA dataset is not downloaded.
3. Select semantic candidates and rerank them.
4. Reject the query if its top candidate fails the quality gate.
5. Check each remaining candidate against the same gate before fetching details.
6. Select the first eligible candidate with complete calorie and macro fields.

The gate requires `final_score >= 0.35` and `cross_encoder_score >= -3.0`.
These are matching scores, not calibrated correctness probabilities.

`nutrition/calculations.py` matches foods to retrieval results, uses
`nutrition/portions.py: convert_to_grams()`, and scales per-100g nutrients to
the consumed amount. Direct g/kg conversion, USDA portions, approximate liquid
density, and household estimates have distinct source labels.

`presentation/responses.py: format_quantity_approximation_note()` explains
liquid-density and household approximations in the response. Ordinary lookup
failures can produce a partial result; if no foods are matched, the flow stops.

## 5. Meal generation and revision

`handle_meal_recommendation()` prioritizes an explicit calorie target. Otherwise,
it calculates a daily target and takes the meal share; if no target can be
determined, it asks the user. Shares are breakfast 25%, lunch 35%, dinner 30%,
and snack 10%; an unspecified meal uses 30%.

`meal_recommendation_flow()` calls `agent/llm.py: call_meal_generation()`, retrieves
USDA nutrition, then calls `agent/revision.py: run_revision_loop()`.
If every initial food fails lookup, it permits one new generation and lookup
attempt before entering the revision loop.

| Validation outcome | Next action |
| --- | --- |
| Valid meal | Accept it. |
| Revision budget exhausted | Return the current meal as invalid with its validation report. |
| Restriction failure | Ask the LLM for substitutions, retrieve nutrition, and recalculate. |
| Calorie failure with protein and restrictions satisfied | Try deterministic scaling once, then revalidate; use a full LLM revision if necessary. |
| Protein failure or another unresolved combination | Request a full LLM revision, retrieve nutrition, and recalculate. |

The budget is two counted revision actions per meal. Initial generation, the
all-foods-failed generation retry, and the one scaling attempt are outside this
count. A counted action can contain multiple LLM calls and can fail to improve
the meal. `success=True` in a meal flow does not imply `valid=True`.

Protein factors are shared: 2 g/kg for weight loss or muscle gain, and 1.6 g/kg
for weight gain or maintenance. Enforcement depends on the flow: a single meal
using calculated daily targets receives a protein floor for every goal;
explicit-target fallback and daily per-meal floors apply to weight loss and
muscle gain when weight is available.

`handle_daily_plan()` generates four meals sequentially, with separate revision
budgets, then validates the successful meals together. It does not run another
revision loop over the entire day. The response's `revision_count` is the final
meal counter, not the sum for all meals.

## 6. Knowledge RAG

`handle_rag_flow()` receives a snapshot of recent conversation messages.
When history is present, `call_question_resolution()` resolves the current
request into a standalone search question before retrieval. Its validated output
contains `question`: text when resolved, or null when a reference is unclear.
Ambiguity prompts clarification; model or validation failures prompt a retry.
Neither path calls retrieval. Without history, the original search path is used.

The resolved query goes to the retrieval tool, which delegates to
`rag/retrieve.py: retrieve_fitness_knowledge()`.

The retriever embeds the query with multilingual MiniLM, searches a FAISS
inner-product index, and returns up to three passages with scores and source
filenames. Retrieval requires the best score to be at least 0.35; individual
returned passages are not each checked against that threshold.

Successful retrieval and recent history are passed to
`agent/llm.py: call_rag_answer()`. History supplies conversational context, while
retrieved passages remain the factual source. Low-confidence
retrieval, retrieval errors, and answer-generation failure have separate replies.
This flow performs no USDA lookup or meal revision. A substitution answer is
knowledge-based advice; it does not guarantee equivalent calories or macros.

USDA retrieval supplies numerical food data. Knowledge RAG supplies text passages
for answers. They are separate retrieval paths.

## 7. Request context and output

The HTTP layer creates a fresh `NutritionAssistant` for each request.
Its constructor copies the supplied profile and history and derives onboarding
status from the profile. No conversation instance is retained by the HTTP layer.

The parser uses the last six history entries. During processing, `NLPSession`
adds the current user message and serialized parsing result to its local history.
The HTTP layer discards that instance after the request.

The returned `AssistantResponse` contains `response`, `intent`, `status`,
`language`, `profile`, `revision_count`, and `meals`. The profile is a snapshot.

The calling backend owns accounts, conversations, and durable storage.
It saves the user message, final response, and approved profile changes, then
supplies the required context with the next request.

Before parsing, the orchestrator snapshots the last six prior user/assistant
messages for answer generation. This excludes the current turn's parser JSON
and avoids appending the current user message twice.
General-chat generation, RAG question resolution, and RAG answer generation
receive that snapshot.

`conversation/meals.py: build_meal_snapshot()` copies final candidates after
revision and scaling into `state.meals`. `contracts/meal_memory.py` defines the
HTTP schema. A snapshot includes a new UUID, final foods and quantities,
calculated nutrition and conversion sources, meal targets, violations, and
partial lookup failures. It is valid only when the flow, validation report,
and calculation succeed without missing food data. Failed generation yields
no snapshot; an invalid generated candidate retains its failed checks.

A daily response contains at most four snapshots. Missing meals are omitted,
and per-meal validity does not certify day completeness or daily targets.
Per-turn reset and unexpected-error handling clear snapshot output.
The chatbot backend validates and stores the latest nonempty batch as
`last_meals`; empty batches leave it unchanged. This storage is in memory.
The optional request field `meal_context` supplies up to four previous meal
snapshots with unique IDs. The HTTP layer serializes and copies these into
`state.meal_context`. Turn reset preserves this input while clearing current
`state.meals` output. A request without context starts with an empty list.
The `meal_edit` flow selects one supplied meal, applies indexed changes, then
retrieves and recalculates every ingredient in that meal. Supplied nutrition is
not the source of fresh calculations. The original calorie target is retained
unless the current request supplies a new target; the saved protein floor and
current profile restrictions are checked. There is no automatic scaling or
nutritional revision loop in this flow.

On valid edits, the original meal ID is retained and `meals` contains the entire
supplied group with one updated meal. Other snapshots remain unchanged; the
whole day is not revalidated. Missing context, ambiguity, failures, and rejected
edits return empty meal output so the backend retains its previous group.
Rejected edits include calculated values and validation reasons in the text.

Selection and change prompts are in `agent/meal_edit_prompts.py`; provider calls
and Pydantic validation are in `agent/meal_edit_llm.py`.
`conversation/meal_edits.py` applies changes against original food indices,
rejecting duplicates, invalid indices, empty meals, and oversized results.
`presentation/meal_edits.py` formats calculated results and rejection reasons.

## 8. Evaluation boundaries

`evaluation/metrics.py: entity_set_f1()` measures micro overlap of `(text, label)`
sets. It does not check entity offsets or token-level NER spans.
`recommendation_restriction_compliance_at_k()` checks literal forbidden-name
membership among the first k recommendations; it does not measure relevance,
calorie accuracy, or ingredient-level allergen safety.

The NLP evaluation runner compares the first food and quantity per example.
Full scope needs predictions from the calorie and recommendation stages.
Sample predictions verify the harness, not live assistant quality. RAG evaluation
compares generated answers against references; overlap scores alone do not
establish factual accuracy. Passing mocked tests does not verify live services.

## 9. Stateless HTTP interface

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness; no external service checks. |
| POST | `/v1/chat` | Process a message with optional profile and prior history. |
| GET | `/docs` | Interactive Swagger documentation. |

The request accepts `message`, optional `profile`, optional `history`,
and optional `meal_context`.
History contains prior messages with `role` equal to `user` or `assistant`,
and a text `content`. Exclude the current message from history.
Partial profiles are supported.

Messages are limited to 4000 characters. History accepts up to 20 entries,
with at most 10000 characters each; the existing parser uses only the last six.
Invalid request data returns 422 before parsing.

The response contains `response`, `intent`, `status`, `language`, `profile`,
`revision_count`, and `meals`. No conversation ID or transcript is stored or returned.
Requests with no supplied context start fresh.

History reaches NLP parsing, general-chat generation, RAG question resolution,
RAG answer generation, and meal-edit selection/change generation.
Service authentication and replay-safe processing remain future work.

HTTP 200 does not certify a valid meal or a successful model call.
Existing response text and metadata can describe a failure or invalid proposal.
`/health` does not test keys, model downloads, or providers.
