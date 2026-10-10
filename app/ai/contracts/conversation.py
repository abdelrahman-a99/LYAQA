"""Structured output for resolving a knowledge question from conversation."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResolvedKnowledgeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(..., min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("question must not be blank")
        return value.strip() if value is not None else None
