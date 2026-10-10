"""Process one request using only its supplied conversation context."""

from fastapi import APIRouter

from app.ai.service import NutritionAssistant

from .schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Report process liveness without checking external APIs or models."""
    return HealthResponse(status="ok")


@router.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat(payload: ChatRequest) -> ChatResponse:
    """Create a request-local assistant; retain no conversation between calls."""
    profile = payload.profile.model_dump() if payload.profile is not None else None
    history = [item.model_dump() for item in payload.history]
    assistant = NutritionAssistant(profile=profile, history=history)
    result = assistant.respond(payload.message)
    return ChatResponse(**result.to_dict())
