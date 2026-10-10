import os

from app.ai.contracts.errors import ProcessingError


def get_api_key() -> str:
    key = os.environ.get("USDA_API_KEY")
    if not key:
        raise ProcessingError(
            "USDA credentials are not configured.",
            code="usda_configuration_error",
            status_code=500,
        )
    return key


def get_base_url() -> str:
    return os.environ.get("USDA_BASE_URL") or "https://api.nal.usda.gov/fdc/v1"
