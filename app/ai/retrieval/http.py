"""Translate USDA transport failures into controlled processing errors."""

import requests

from app.ai.contracts.errors import ProcessingError
from app.ai.retrieval.config import get_api_key, get_base_url


def invalid_response() -> ProcessingError:
    return ProcessingError(
        "USDA returned an invalid response.",
        code="usda_invalid_response",
        status_code=502,
    )


def request_json(
    path: str, *, params: dict | None = None, allow_not_found: bool = False
) -> dict | None:
    """Fetch a JSON object; only an allowed detail 404 returns None."""
    url = f"{get_base_url().rstrip('/')}/{path.lstrip('/')}"
    query = {**(params or {}), "api_key": get_api_key()}
    try:
        response = requests.get(url, params=query, timeout=10)
        if response.status_code == 404 and allow_not_found:
            return None
        if response.status_code in {401, 403}:
            raise ProcessingError(
                "USDA authentication failed.",
                code="usda_authentication_failed",
                status_code=502,
            )
        if response.status_code == 429:
            raise ProcessingError(
                "USDA is temporarily rate limited.",
                code="usda_rate_limited",
                status_code=503,
                retryable=True,
            )
        if response.status_code >= 500:
            raise ProcessingError(
                "USDA is temporarily unavailable.",
                code="usda_unavailable",
                status_code=503,
                retryable=True,
            )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise ProcessingError(
            "USDA request timed out.",
            code="usda_timeout",
            status_code=504,
            retryable=True,
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise ProcessingError(
            "USDA rejected the request.",
            code="usda_request_rejected",
            status_code=502,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProcessingError(
            "Could not complete the USDA connection.",
            code="usda_unavailable",
            status_code=503,
            retryable=True,
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise invalid_response() from exc
    if not isinstance(data, dict):
        raise invalid_response()
    return data
