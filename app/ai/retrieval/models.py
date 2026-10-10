"""Lazy embedding, reranking and Arabic translation models."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict

from app.ai.contracts.errors import ProcessingError
from app.ai.retrieval.runtime import get_device, prepare_transformers

logger = logging.getLogger(__name__)


TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


_MODELS: Dict[str, Any] = {}
_EMBEDDING_LOAD_LOCK = Lock()
_TRANSLATION_LOAD_LOCK = Lock()


def _model_loading_error() -> ProcessingError:
    return ProcessingError(
        "Food retrieval models could not be loaded.",
        code="retrieval_model_unavailable",
        status_code=503,
    )


def _ensure_embedding_models() -> None:
    """Publish the complete model group after one successful load per process."""
    required = ("bi_encoder", "cross_encoder", "util")
    if all(name in _MODELS for name in required):
        return

    with _EMBEDDING_LOAD_LOCK:
        if all(name in _MODELS for name in required):
            return

        try:
            prepare_transformers()
            from sentence_transformers import CrossEncoder, SentenceTransformer, util

            device = get_device()
            logger.info("Loading embedding models on device=%s ...", device)
            bi_encoder = SentenceTransformer(BI_ENCODER_MODEL_NAME, device=device)
            cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=device)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise _model_loading_error() from exc

        _MODELS.update(
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
            util=util,
        )
        logger.info("Embedding models loaded.")


def _ensure_translation_model() -> None:
    """Publish the tokenizer and translation model only after both load."""
    required = ("translation_tokenizer", "translation_model")
    if all(name in _MODELS for name in required):
        return

    with _TRANSLATION_LOAD_LOCK:
        if all(name in _MODELS for name in required):
            return

        try:
            prepare_transformers()
            from transformers import MarianMTModel, MarianTokenizer

            device = get_device()
            logger.info("Loading translation model on device=%s ...", device)
            tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
            model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME).to(device)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise _model_loading_error() from exc

        _MODELS.update(
            translation_tokenizer=tokenizer,
            translation_model=model,
        )
        logger.info("Translation model loaded.")
