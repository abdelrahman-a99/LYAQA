"""
tests/test_rag.py — Tests for the RAG component.
"""

from unittest.mock import MagicMock, patch

from app.ai.rag import retrieve


def test_rag_quality_gate_fallback():
    # If the score is below the threshold (0.35), quality_gate should fail
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock()
    mock_index = MagicMock()
    # D is similarity score, e.g. 0.20 (below 0.35)
    import numpy as np

    mock_index.search.return_value = (
        np.array([[0.20]], dtype=np.float32),
        np.array([[0]]),
    )

    with (
        patch("app.ai.rag.retrieve._index", mock_index),
        patch("app.ai.rag.retrieve._embed_model", mock_model),
        patch(
            "app.ai.rag.retrieve._chunks",
            [{"text": "Sample text", "source_file": "doc.md"}],
        ),
    ):
        res = retrieve.retrieve_fitness_knowledge("irrelevant query")
        assert res["success"] is False
        assert res["error_code"] == "low_confidence_retrieval"
        assert res["data"] is None


def test_rag_quality_gate_pass():
    # If score is above 0.35, quality_gate should pass
    mock_model = MagicMock()
    import numpy as np

    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.75]], dtype=np.float32),
        np.array([[0]]),
    )

    with (
        patch("app.ai.rag.retrieve._index", mock_index),
        patch("app.ai.rag.retrieve._embed_model", mock_model),
        patch(
            "app.ai.rag.retrieve._chunks",
            [{"text": "High protein helps recovery.", "source_file": "doc.md"}],
        ),
    ):
        res = retrieve.retrieve_fitness_knowledge("protein recovery", top_k=1)
        assert res["success"] is True
        assert len(res["data"]["passages"]) == 1
        assert res["data"]["passages"][0] == "High protein helps recovery."
