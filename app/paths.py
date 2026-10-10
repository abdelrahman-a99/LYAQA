"""Locate service resources independently of the current working directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
EVALUATION_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
DEFAULT_RAG_DIR = PROJECT_ROOT / "artifacts" / "rag"


def get_rag_dir() -> Path:
    value = os.environ.get("NUTRITION_AI_RAG_DIR")

    if not value:
        return DEFAULT_RAG_DIR

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
