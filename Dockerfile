# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV NUTRITION_AI_RUNTIME=cpu \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    USE_JAX=0 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser \
    && mkdir -p /opt/nutrition-ai \
    && chown appuser:appuser /opt/nutrition-ai

WORKDIR /opt/nutrition-ai

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir "torch==2.8.0" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data/knowledge_base ./data/knowledge_base
COPY --chown=appuser:appuser artifacts/rag ./artifacts/rag

USER appuser

# Reuse downloads across builds, then copy the Hub cache into the image.
RUN --mount=type=cache,id=nutrition-ai-models,target=/tmp/hf-cache,uid=1000,gid=1000 HF_HOME=/tmp/hf-cache python - <<'PY'
import os
import shutil
from pathlib import Path

import torch

from app.ai.rag import retrieve
from app.ai.retrieval.models import (
    _ensure_embedding_models,
    _ensure_translation_model,
)

assert torch.version.cuda is None, "Expected the CPU-only PyTorch wheel."
_ensure_embedding_models()
_ensure_translation_model()
retrieve._ensure_loaded()

assert retrieve._index.ntotal == len(retrieve._chunks), "RAG row count mismatch."
dimension = retrieve._embed_model.get_sentence_embedding_dimension()
assert retrieve._index.d == dimension, "RAG dimension mismatch."

shutil.copytree(
    Path(os.environ["HF_HOME"]) / "hub",
    Path("/home/appuser/.cache/huggingface/hub"),
    symlinks=True,
    dirs_exist_ok=True,
)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Verify that all model groups load from the image without the download cache.
RUN python - <<'PY'
from app.ai.rag.retrieve import _ensure_loaded
from app.ai.retrieval.models import (
    _ensure_embedding_models,
    _ensure_translation_model,
)

_ensure_embedding_models()
_ensure_translation_model()
_ensure_loaded()
print("Offline model loading verified.")
PY

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
