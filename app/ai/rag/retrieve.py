"""Retrieve knowledge passages with multilingual embeddings and FAISS.

Intent routing selects when retrieval is called. This module embeds the query,
searches the index, and applies a minimum best-score threshold before returning
passages, similarity scores, and source filenames for answer generation.
The threshold applies to the best result, not independently to every passage.
Similarity scores are retrieval scores, not calibrated probabilities.

The embedding model, index, and chunks are loaded on first use.
NUTRITION_AI_RAG_DIR can override artifacts/rag under the repository root.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

from app.paths import get_rag_dir

logger = logging.getLogger(__name__)

RAG_SIMILARITY_THRESHOLD = 0.35  # Minimum best-result similarity for retrieval.

_RAG_DIR = get_rag_dir()
_INDEX_PATH = os.path.join(_RAG_DIR, "faiss_index.bin")
_CHUNKS_PATH = os.path.join(_RAG_DIR, "chunks.json")

# Lazy-loaded so the index is not read at import time.
_index = None
_chunks: Optional[list] = None
_embed_model = None


def _ensure_loaded() -> None:
    """Load the FAISS index, chunks, and embedding model on first call."""
    global _index, _chunks, _embed_model

    if _index is not None:
        return

    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_CHUNKS_PATH):
        raise FileNotFoundError(
            "RAG index not found. Run `python -m app.ai.rag.build_index` first."
        )

    from app.ai.retrieval.runtime import prepare_transformers

    prepare_transformers()

    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(_INDEX_PATH)

    with open(_CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    # Use the same embedding model that generated the RAG index.
    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    _index, _chunks, _embed_model = index, chunks, embed_model

    logger.info("RAG index loaded: %d chunks", len(_chunks))


def retrieve_fitness_knowledge(query: str, top_k: int = 3) -> dict:
    """
    Search the knowledge base and return the top passages or a confidence failure.

    Returns:
        {success: True,  data: {passages: [str], scores: [float], sources: [str]}}
        {success: False, data: None, error_code: "low_confidence_retrieval"}
        {success: False, data: None, error_code: "index_not_found"}
    """
    try:
        _ensure_loaded()
    except FileNotFoundError as exc:
        return {
            "success": False,
            "data": None,
            "error": str(exc),
            "error_code": "index_not_found",
        }

    query_vec = _embed_model.encode([query], normalize_embeddings=True).astype(
        np.float32
    )

    scores, indices = _index.search(query_vec, top_k)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    best_score = max(scores) if scores else 0.0

    if best_score < RAG_SIMILARITY_THRESHOLD:
        logger.info(
            "RAG low confidence: best_score=%.3f < threshold=%.2f for query=%r",
            best_score,
            RAG_SIMILARITY_THRESHOLD,
            query[:80],
        )
        return {
            "success": False,
            "data": None,
            "error": "Retrieval confidence too low to give a reliable answer.",
            "error_code": "low_confidence_retrieval",
        }

    passages = []
    result_scores = []
    sources = []
    for idx, score in zip(indices, scores):
        if 0 <= idx < len(_chunks):
            passages.append(_chunks[idx]["text"])
            result_scores.append(round(float(score), 4))
            sources.append(_chunks[idx]["source_file"])

    return {
        "success": True,
        "data": {"passages": passages, "scores": result_scores, "sources": sources},
        "error": None,
        "error_code": None,
    }
