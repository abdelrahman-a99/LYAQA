"""Lazy embedding, reranking and Arabic translation models."""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.ai.retrieval.runtime import get_device, prepare_transformers

logger = logging.getLogger(__name__)


TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


_MODELS: Dict[str, Any] = {}


def _ensure_embedding_models() -> None:
    """Load embedding and reranking models once."""
    if "bi_encoder" in _MODELS:
        return

    prepare_transformers()

    from sentence_transformers import (
        CrossEncoder,
        SentenceTransformer,
        util,
    )

    logger.info(
        "Loading embedding models on device=%s ...",
        get_device(),
    )

    _MODELS["bi_encoder"] = SentenceTransformer(
        BI_ENCODER_MODEL_NAME,
        device=get_device(),
    )

    _MODELS["cross_encoder"] = CrossEncoder(
        RERANKER_MODEL_NAME,
        device=get_device(),
    )

    _MODELS["util"] = util

    logger.info("Embedding models loaded.")


def _ensure_translation_model() -> None:
    """Load MarianMT when an Arabic query actually needs translation."""
    if "translation_model" in _MODELS:
        return

    prepare_transformers()

    from transformers import MarianMTModel, MarianTokenizer

    logger.info(
        "Loading translation model on device=%s ...",
        get_device(),
    )

    _MODELS["translation_tokenizer"] = MarianTokenizer.from_pretrained(
        TRANSLATION_MODEL_NAME
    )

    _MODELS["translation_model"] = MarianMTModel.from_pretrained(
        TRANSLATION_MODEL_NAME
    ).to(get_device())

    logger.info("Translation model loaded.")
