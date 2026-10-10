import os


def get_api_key() -> str:
    key = os.environ.get("USDA_API_KEY")
    if not key:
        raise ValueError("USDA_API_KEY is required for USDA requests.")
    return key


def get_base_url() -> str:
    return os.environ.get("USDA_BASE_URL") or "https://api.nal.usda.gov/fdc/v1"
