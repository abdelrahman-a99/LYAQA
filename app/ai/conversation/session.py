"""Multi-turn profile and parsing history for one user conversation."""

from __future__ import annotations

import json
from typing import List

from app.ai.conversation.onboarding import (
    PROFILE_DEPENDENT_INTENTS,
    get_onboarding_prompt,
    merge_profile,
    validate_mandatory_profile_fields,
)
from app.ai.nlp.language import detect_language
from app.ai.nlp.parser import parse_with_context


class NLPSession:
    """Maintain an accumulated profile and parsing history for one conversation.

    Each turn is parsed with the stored context, then extracted profile fields
    are merged into the accumulated profile. daily_plan and update_profile
    require birth_date, gender, weight, height, activity_level, and goal.
    Other intents proceed without the session's mandatory-profile gate;
    their flows may still ask for missing information.

    History stores the user message and the extracted JSON as an assistant
    entry. The final user-facing chatbot reply is not recorded here.
    State is held in memory on this instance and is not persisted to a database.
    """

    def __init__(self):
        # Stored user profile — starts fully null / empty
        self.profile: dict = {
            "birth_date": None,
            "gender": None,
            "weight": None,
            "height": None,
            "activity_level": None,
            "goal": None,
            "dietary_restrictions": [],
            "forbidden_foods": [],
            "preferred_foods": [],
            "allergies": [],
        }
        self.onboarding_complete: bool = False
        self.history: List[dict] = []
        self.language: str = "english"

    # ------------------------------------------------------------------
    def process(self, user_message: str) -> dict:
        """
        Process one user turn.

        Selective onboarding:
        - Simple queries are NEVER blocked by missing profile fields.
        - Only profile-dependent intents (daily_plan, update_profile) trigger onboarding.
        """
        self.language = detect_language(user_message)
        parsed = parse_with_context(user_message, self.history)

        # Append to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append(
            {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)}
        )

        # Merge any extracted profile info
        if parsed.get("profile_data"):
            self.profile = merge_profile(
                self.profile, parsed["profile_data"], user_message
            )

        # Check mandatory fields
        is_complete, missing = validate_mandatory_profile_fields(self.profile)
        if is_complete:
            self.onboarding_complete = True

        intent = parsed.get("intent")

        # ── Case 1: profile-dependent intent, profile incomplete ───────────
        if intent in PROFILE_DEPENDENT_INTENTS and not self.onboarding_complete:
            prompt = get_onboarding_prompt(missing, self.language)
            return {
                "parsed": parsed,
                "status": "needs_profile",
                "prompt_for_user": prompt,
                "profile": self.profile,
                "missing_fields": missing,
            }

        # Attach the full accumulated profile so Task 3 always has it
        parsed["profile_data"] = self.profile

        # ── Case 2: just finished onboarding via update_profile ────────────
        if intent == "update_profile" and self.onboarding_complete:
            if self.language == "arabic":
                confirm = (
                    "✅ ممتاز! استلمت بياناتك بنجاح.\n"
                    "دلوقتي تقدر:\n"
                    "  • تسجل وجباتك (مثلاً: 'كلت 200g فراخ مشوية')\n"
                    "  • تطلب خطة أكل ('ممكن خطة أكل ليوم؟')\n"
                    "  • تسأل عن سعرات أي أكلة ('كم سعرة في الأفوكادو؟')"
                )
            else:
                confirm = (
                    "✅ Profile complete! Here's what you can do now:\n"
                    "  • Log a meal (e.g. 'I had 200g grilled chicken')\n"
                    "  • Request a meal plan ('Give me a full day meal plan')\n"
                    "  • Ask about food calories ('How many calories in avocado?')"
                )
            return {
                "parsed": parsed,
                "status": "onboarding_complete",
                "prompt_for_user": confirm,
                "profile": self.profile,
                "missing_fields": [],
            }

        # ── Case 3: normal flow — profile already complete ─────────────────
        return {
            "parsed": parsed,
            "status": "ready",
            "prompt_for_user": None,
            "profile": self.profile,
            "missing_fields": [],
        }
