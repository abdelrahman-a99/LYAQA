from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: str | Path = ".env") -> bool:
    """Load the calling application's environment; existing values win."""
    return load_dotenv(dotenv_path=path, override=False)
