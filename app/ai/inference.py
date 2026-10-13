"""Small model operations; network providers and orchestration stay outside GPU tasks."""

from collections.abc import Callable

from app.ai.contracts.errors import ProcessingError
from app.ai.runtime import get_device, get_runtime

_zero_executor: Callable[[str, dict], object] | None = None


def install_zero_executor(executor: Callable[[str, dict], object]) -> None:
    global _zero_executor
    _zero_executor = executor


def run_inference(operation: str, payload: dict):
    if get_runtime() == "zerogpu":
        if _zero_executor is None:
            raise ProcessingError(
                "ZeroGPU execution has not been initialized.",
                code="runtime_configuration_error",
                status_code=500,
            )
        return _zero_executor(operation, payload)
    return run_local_inference(operation, payload)


def run_local_inference(operation: str, payload: dict):
    if operation == "semantic":
        from app.ai.retrieval import models

        models._ensure_embedding_models()
        encoder = models._MODELS["bi_encoder"]
        query = encoder.encode(
            payload["text"], convert_to_tensor=True, normalize_embeddings=True
        )
        descriptions = encoder.encode(
            payload["descriptions"], convert_to_tensor=True, normalize_embeddings=True
        )
        scores = models._MODELS["util"].cos_sim(query, descriptions)[0]
        return [float(score) for score in scores]
    if operation == "rerank":
        import numpy as np

        from app.ai.retrieval import models

        models._ensure_embedding_models()
        pairs = [
            [payload["text"], description] for description in payload["descriptions"]
        ]
        scores = models._MODELS["cross_encoder"].predict(pairs)
        return np.asarray(scores, dtype=float).reshape(-1).tolist()
    if operation == "translate":
        import torch

        from app.ai.retrieval import models

        models._ensure_translation_model()
        tokenizer = models._MODELS["translation_tokenizer"]
        model = models._MODELS["translation_model"]
        inputs = tokenizer(
            payload["text"], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(get_device()) for key, value in inputs.items()}
        with torch.no_grad():
            tokens = model.generate(**inputs, max_length=100)
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    if operation == "rag_embed":
        from app.ai.rag import retrieve

        retrieve._ensure_loaded()
        vectors = retrieve._embed_model.encode(
            [payload["text"]], normalize_embeddings=True
        )
        return vectors.tolist()
    raise ValueError("Unsupported model operation.")


def warm_zero_models() -> None:
    """Place every model on emulated CUDA before Gradio starts the workers."""
    from app.ai.rag import retrieve
    from app.ai.retrieval import models

    models._ensure_embedding_models()
    models._ensure_translation_model()
    retrieve._ensure_loaded()
    if retrieve._index.ntotal != len(retrieve._chunks):
        raise ProcessingError(
            "RAG index and chunks are inconsistent.",
            code="rag_configuration_error",
            status_code=500,
        )
    if retrieve._index.d != retrieve._embed_model.get_sentence_embedding_dimension():
        raise ProcessingError(
            "RAG index embedding dimension is inconsistent.",
            code="rag_configuration_error",
            status_code=500,
        )
