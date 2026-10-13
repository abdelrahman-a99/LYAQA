"""Select one inference runtime per process; never fall back implicitly."""

import os
from functools import lru_cache

from app.ai.contracts.errors import ProcessingError


def configuration_error(message: str) -> ProcessingError:
    return ProcessingError(message, code="runtime_configuration_error", status_code=500)


@lru_cache(maxsize=1)
def get_runtime() -> str:
    value = os.environ.get("NUTRITION_AI_RUNTIME", "cpu").strip().lower()
    if value not in {"cpu", "cuda", "zerogpu"}:
        raise configuration_error("NUTRITION_AI_RUNTIME must be cpu, cuda, or zerogpu.")
    return value


def initialize_runtime(*, allow_zerogpu: bool = False) -> None:
    mode = get_runtime()
    if mode == "cpu":
        return
    if mode == "cuda" and os.environ.get("SPACES_ZERO_GPU", "").lower() in {
        "1",
        "true",
    }:
        raise configuration_error(
            "Use zerogpu runtime on ZeroGPU hardware, not persistent cuda."
        )
    if mode == "zerogpu":
        if not allow_zerogpu:
            raise configuration_error(
                "Start ZeroGPU with space_app.py, not Uvicorn directly."
            )
        if os.environ.get("SPACES_ZERO_GPU", "").lower() not in {"1", "true"}:
            raise configuration_error(
                "ZeroGPU runtime requires Hugging Face ZeroGPU hardware."
            )
        if not os.environ.get("SPACES_ZERO_DEVICE_API_URL"):
            raise configuration_error(
                "Hugging Face ZeroGPU scheduling is not configured."
            )
        # Import before torch so the hosted CUDA emulation can be initialized.
        try:
            import spaces  # noqa: F401
            from spaces.config import Config
        except ImportError as exc:
            raise configuration_error(
                "Install the pinned Gradio and spaces dependencies."
            ) from exc
        if not Config.zero_gpu:
            raise configuration_error(
                "Restart with ZeroGPU configured before importing spaces."
            )

        if not os.environ.get("SPACE_ID") or not os.environ.get("HF_TOKEN"):
            raise configuration_error(
                "ZeroGPU requires SPACE_ID and an HF_TOKEN for GPU API calls."
            )
    try:
        import torch
    except ImportError as exc:
        raise configuration_error(
            "Install a CUDA-enabled PyTorch build for this runtime."
        ) from exc
    if mode == "cuda" and not torch.cuda.is_available():
        raise configuration_error(
            "CUDA runtime was requested, but CUDA is unavailable."
        )
    if mode == "zerogpu" and torch.version.cuda is None:
        raise configuration_error(
            "ZeroGPU requires CUDA-enabled PyTorch, not a CPU wheel."
        )
    if mode == "zerogpu" and tuple(
        int(p) for p in torch.__version__.split("+")[0].split(".")[:2]
    ) < (2, 8):
        raise configuration_error(
            "ZeroGPU requires a supported PyTorch version (2.8 or newer)."
        )


@lru_cache(maxsize=1)
def get_device() -> str:
    mode = get_runtime()
    if mode == "cpu":
        return "cpu"
    initialize_runtime(allow_zerogpu=mode == "zerogpu")
    # Hosted ZeroGPU uses CUDA emulation during startup and real CUDA in GPU tasks.
    return "cuda"
