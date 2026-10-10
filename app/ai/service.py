from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AssistantResponse:
    response: str
    intent: str
    status: str
    language: str
    profile: dict
    revision_count: int
    meals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class NutritionAssistant:
    """Run an assistant with caller-supplied profile, history, and meal context.

    Each HTTP request uses a fresh instance initialized with caller-supplied
    context. This class does not persist conversations between requests.
    """

    def __init__(
        self,
        *,
        profile: dict | None = None,
        history: list[dict] | None = None,
        meal_context: list[dict] | None = None,
    ):
        self.reset()

        if profile is not None:
            from app.ai.conversation.onboarding import (
                validate_mandatory_profile_fields,
            )

            self.nlp_session.profile.update(deepcopy(profile))
            complete, _ = validate_mandatory_profile_fields(self.nlp_session.profile)
            self.nlp_session.onboarding_complete = complete
            self.agent_state.user_profile = deepcopy(self.nlp_session.profile)

        if history is not None:
            self.nlp_session.history = deepcopy(history)

        self.agent_state.meal_context = deepcopy(meal_context or [])

    def reset(self) -> None:
        from app.ai.agent.state import AgentState
        from app.ai.conversation.session import NLPSession

        self.nlp_session = NLPSession()
        self.agent_state = AgentState()

    def get_session_info(self) -> dict:
        """Return a snapshot of the current conversation metadata."""
        return {
            "status": self.agent_state.status or "unknown",
            "intent": self.agent_state.intent or "unknown",
            "language": self.nlp_session.language or "english",
            "profile": deepcopy(self.agent_state.user_profile or {}),
            "history_length": len(self.nlp_session.history),
            "onboarding_complete": bool(self.nlp_session.onboarding_complete),
        }

    def respond(self, message: str) -> AssistantResponse:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        from app.ai.agent import orchestrator

        try:
            response = orchestrator.handle(
                message=message,
                session=self.nlp_session,
                state=self.agent_state,
            )
        except Exception:
            self.agent_state.meals = []
            raise

        return AssistantResponse(
            response=response,
            intent=self.agent_state.intent or "unknown",
            status=self.agent_state.status or "unknown",
            language=self.nlp_session.language or "english",
            profile=deepcopy(self.agent_state.user_profile or {}),
            revision_count=int(self.agent_state.revision_count or 0),
            meals=deepcopy(self.agent_state.meals),
        )
