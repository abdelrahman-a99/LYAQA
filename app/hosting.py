"""Headless Gradio hosting with the existing FastAPI routes and error contract."""

import logging
import math
import os
from concurrent.futures import TimeoutError
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.contracts.errors import ProcessingError
from app.ai.inference import install_zero_executor, run_local_inference
from app.ai.runtime import get_runtime
from app.main import create_app

logger = logging.getLogger(__name__)


class ModelTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["semantic", "rerank", "translate", "rag_embed"]
    text: str = Field(min_length=1, max_length=4000)
    descriptions: list[str] = Field(default_factory=list, max_length=300)

    @model_validator(mode="after")
    def require_candidates(self):
        if self.operation in {"semantic", "rerank"} and not self.descriptions:
            raise ValueError("Candidates are required for ranking operations.")
        return self


GPU_ERRORS = {
    "zerogpu_quota_exceeded": (
        503,
        False,
        "The caller's ZeroGPU allowance is unavailable.",
    ),
    "zerogpu_unavailable": (
        503,
        True,
        "ZeroGPU could not execute the model operation.",
    ),
    "zerogpu_execution_failed": (503, False, "GPU model execution failed."),
    "zerogpu_invalid_output": (502, False, "ZeroGPU returned an invalid result."),
    "zerogpu_timeout": (504, True, "ZeroGPU execution timed out."),
    "zerogpu_authentication_failed": (
        502,
        False,
        "Hugging Face API authentication failed.",
    ),
}


def gpu_error(code: str) -> ProcessingError:
    status, retryable, message = GPU_ERRORS[code]
    return ProcessingError(message, code=code, status_code=status, retryable=retryable)


def pack_error(exc: ProcessingError) -> dict:
    return {
        "ok": False,
        "error": {
            "detail": str(exc),
            "error_code": exc.code,
            "status_code": exc.status_code,
            "retryable": exc.retryable,
        },
    }


def gpu_failure(exc: Exception) -> ProcessingError:
    text = str(exc).lower()
    if "quota" in text or "runs limit" in text or "pending credits" in text:
        return gpu_error("zerogpu_quota_exceeded")
    return gpu_error("zerogpu_unavailable")


def validate_model_result(task: ModelTask, data):
    def numbers(values):
        return isinstance(values, list) and all(
            type(item) in {int, float} and math.isfinite(item) for item in values
        )

    if task.operation == "translate":
        valid = isinstance(data, str) and bool(data.strip())
    elif task.operation == "rag_embed":
        valid = (
            isinstance(data, list)
            and len(data) == 1
            and numbers(data[0])
            and len(data[0]) == 384
        )
    else:
        valid = numbers(data) and len(data) == len(task.descriptions)
    if not valid:
        raise gpu_error("zerogpu_invalid_output")
    return data


@lru_cache(maxsize=1)
def get_gpu_client():
    from gradio_client import Client

    # Use the official public Space API so Hugging Face authenticates this token.
    # A localhost call alone would not establish the caller's HF quota identity.
    return Client(os.environ["SPACE_ID"], token=os.environ["HF_TOKEN"], verbose=False)


def execute_queued(operation: str, payload: dict):
    import httpx
    from gradio_client.exceptions import AppError, AuthenticationError
    from huggingface_hub.utils import HfHubHTTPError

    task = ModelTask(operation=operation, **payload)
    job = None
    try:
        job = get_gpu_client().submit(task.model_dump(), api_name="/model_compute")
        result = job.result(timeout=120)
    except TimeoutError as exc:
        if job is not None:
            try:
                job.cancel()
            except Exception:
                logger.warning("Could not cancel the expired GPU job")
        raise gpu_error("zerogpu_timeout") from exc
    except AuthenticationError as exc:
        raise gpu_error("zerogpu_authentication_failed") from exc
    except (HfHubHTTPError, httpx.HTTPStatusError) as exc:
        if exc.response.status_code in {401, 403}:
            raise gpu_error("zerogpu_authentication_failed") from exc
        raise gpu_error("zerogpu_unavailable") from exc
    except AppError as exc:
        raise gpu_failure(exc) from exc
    except Exception as exc:
        raise ProcessingError(
            "ZeroGPU API transport is unavailable.",
            code="zerogpu_unavailable",
            status_code=503,
            retryable=True,
        ) from exc
    if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
        raise gpu_error("zerogpu_invalid_output")
    if not result["ok"]:
        error = result.get("error", {})
        if not isinstance(error, dict) or error.get("error_code") not in GPU_ERRORS:
            raise gpu_error("zerogpu_invalid_output")
        raise gpu_error(error["error_code"])
    return validate_model_result(task, result.get("data"))


def create_space_app():
    import gradio as gr

    application = create_app(application_class=gr.Server, allow_zerogpu=True)

    def kernel(payload: dict) -> dict:
        try:
            task = ModelTask.model_validate(payload)
            data = task.model_dump(exclude={"operation"})
            return {"ok": True, "data": run_local_inference(task.operation, data)}
        except Exception:
            logger.exception("Model task failed")
            return pack_error(
                ProcessingError(
                    "GPU model execution failed.",
                    code="zerogpu_execution_failed",
                    status_code=503,
                )
            )

    if get_runtime() == "zerogpu":
        import spaces

        # Only model computation holds a GPU lease; no Groq or USDA calls here.
        kernel = spaces.GPU(duration=20)(kernel)
        install_zero_executor(execute_queued)

    @application.api(
        name="model_compute",
        concurrency_limit=1,
        queue=True,
        api_visibility="undocumented",
    )
    def model_compute(payload: dict) -> dict:
        try:
            return kernel(payload)
        except gr.Error as exc:
            return pack_error(gpu_failure(exc))

    return application
