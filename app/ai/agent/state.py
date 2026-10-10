"""Store the conversation identity, profile, and current agent-turn data."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """Mutable state shared by the orchestrator and its tools and flows.

    reset_turn preserves session_id and user_profile while clearing the
    current query, targets, intermediate results, and revision history.
    Targets are populated when calculated for the current turn.
    Revision history records meal-validation attempts during that turn.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_profile: Dict[str, Any] = field(default_factory=dict)
    raw_query: str = ""
    intent: str = ""
    status: str = ""
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    targets: Optional[Dict[str, Any]] = None
    retrieved_foods: List[Dict] = field(default_factory=list)
    candidate_meals: List[Dict] = field(default_factory=list)
    validation_result: Dict[str, Any] = field(default_factory=dict)
    revision_count: int = 0
    revision_history: List[Dict] = field(default_factory=list)
    final_answer: str = ""

    def reset_turn(self) -> None:
        """
        Clear all per-turn fields while keeping session_id and user_profile.

        Call this at the start of each new orchestrator turn so stale
        intermediate data from the previous turn cannot bleed through.
        """
        self.raw_query = ""
        self.intent = ""
        self.status = ""
        self.extracted_entities = {}
        self.constraints = {}
        self.targets = None
        self.retrieved_foods = []
        self.candidate_meals = []
        self.validation_result = {}
        self.revision_count = 0
        self.revision_history = []
        self.final_answer = ""
