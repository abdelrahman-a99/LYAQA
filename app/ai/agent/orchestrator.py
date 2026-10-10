"""Route parsed requests to the corresponding agent flow.

Keep the profile-status gate and per-turn state handling here.
"""

from app.ai.agent.flows.calories import handle_calorie_macro_flow
from app.ai.agent.flows.chat import handle_general_chat
from app.ai.agent.flows.daily import handle_daily_plan
from app.ai.agent.flows.knowledge import handle_rag_flow
from app.ai.agent.flows.meals import handle_meal_recommendation
from app.ai.agent.state import AgentState
from app.ai.agent.tools import parse_query
from app.ai.conversation.session import NLPSession


def _route_and_answer(message: str, session: NLPSession, state: AgentState) -> str:
    """
    Parse, route by intent, return a user-facing answer string.

    The status gate is the first and only gate — checked explicitly here,
    not buried inside intent-specific code.
    """
    state.raw_query = message

    parse_result = parse_query(message, session, intent="routing")
    if not parse_result["success"]:
        return "Sorry, I could not understand your message. Please try again."

    session_output = parse_result["data"]
    state.status = session_output.get("status", "ready")
    parsed = session_output.get("parsed", {})
    profile = session_output.get("profile", {})
    language = parsed.get("language_detected", "english")

    state.user_profile = profile
    state.intent = parsed.get("intent", "general_chat")

    # Status gate — checked first, explicitly (spec section 3)
    if state.status == "needs_profile":
        return session_output.get("prompt_for_user", "Please complete your profile.")

    intent = state.intent
    foods_from_task1 = parsed.get("foods", [])
    meal_type = parsed.get("meal_type")
    target_calories_from_message = parsed.get("target_calories")
    requested_substitution = parsed.get("requested_substitution")

    # ── general_chat ────────────────────────────────────────────────────────
    if intent == "general_chat":
        return handle_general_chat(message, language)

    # ── update_profile ──────────────────────────────────────────────────────
    if intent == "update_profile":
        # The reply is already ready from parse_query's prompt_for_user
        if state.status == "onboarding_complete":
            return session_output.get("prompt_for_user", "Profile updated.")
        return session_output.get(
            "prompt_for_user", "Profile partially updated. Please continue."
        )

    # ── calculate_calories / calculate_macros / food_nutrition ──────────────
    if intent in ("calculate_calories", "calculate_macros", "food_nutrition"):
        return handle_calorie_macro_flow(foods_from_task1, profile, language, intent)

    # ── food_substitution / fitness_nutrition_question ──────────────────────
    if intent in ("food_substitution", "fitness_nutrition_question"):
        query = message
        if intent == "food_substitution" and requested_substitution:
            query = f"What can I substitute for {requested_substitution}?"
        return handle_rag_flow(query, language, intent)

    # ── meal_recommendation ─────────────────────────────────────────────────
    if intent == "meal_recommendation":
        return handle_meal_recommendation(
            target_calories_from_message, profile, meal_type, language, state
        )

    # ── daily_plan ──────────────────────────────────────────────────────────
    if intent == "daily_plan":
        return handle_daily_plan(profile, language, state)

    # Unknown intent — safe fallback
    return handle_general_chat(message, language)


def handle(message: str, session: NLPSession, state: AgentState) -> str:
    """
    Public entry point.

    Resets per-turn state, routes the message through _route_and_answer,
    and stores the final reply in state.final_answer before returning it.
    """
    state.reset_turn()
    answer = _route_and_answer(message, session, state)
    state.final_answer = answer
    return answer
