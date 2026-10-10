"""Classify Groq failures without exposing provider response bodies."""

from typing import NoReturn

from app.ai.contracts.errors import ProcessingError


def raise_groq_failure(exc: Exception) -> NoReturn:
    if isinstance(exc, ProcessingError):
        raise exc

    import groq

    if isinstance(exc, groq.APITimeoutError):
        raise ProcessingError(
            "The language model request timed out.",
            code="llm_timeout",
            status_code=504,
            retryable=True,
        ) from exc

    if isinstance(exc, groq.RateLimitError):
        raise ProcessingError(
            "The language model is temporarily rate limited.",
            code="llm_rate_limited",
            status_code=503,
            retryable=True,
        ) from exc

    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        raise ProcessingError(
            "Language model credentials or permissions require attention.",
            code="llm_authentication_failed",
            status_code=502,
        ) from exc

    if isinstance(exc, groq.APIConnectionError) or (
        isinstance(exc, groq.APIStatusError) and exc.status_code >= 500
    ):
        raise ProcessingError(
            "The language model is temporarily unavailable.",
            code="llm_unavailable",
            status_code=503,
            retryable=True,
        ) from exc

    if isinstance(exc, groq.APIStatusError):
        raise ProcessingError(
            "The language model rejected the service request.",
            code="llm_request_rejected",
            status_code=502,
        ) from exc

    if isinstance(exc, (ValueError, TypeError)):
        raise ProcessingError(
            "The language model returned an invalid output.",
            code="llm_invalid_output",
            status_code=502,
        ) from exc

    raise exc


def completion_text(response) -> str:
    """Require non-empty completion text before parsing or displaying it."""
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Missing completion text") from exc

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty completion text")

    return text.strip()
