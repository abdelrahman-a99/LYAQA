"""Public HTTP error contract; keep exception details in server logs."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

from app.ai.contracts.errors import ProcessingError

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    retryable: bool
    issues: list[dict] = Field(default_factory=list)


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int, *, code: str, retryable: bool):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


CHAT_ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (422, 500, 502, 503, 504)
}


def error_response(
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    *,
    issues: list[dict] | None = None,
    headers: dict | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        detail=message,
        error_code=code,
        retryable=retryable,
        issues=issues or [],
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_defaults=True),
        headers=headers,
    )


def install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(ProcessingError)
    @application.exception_handler(ServiceError)
    async def service_error(
        request: Request,
        exc: ServiceError | ProcessingError,
    ) -> JSONResponse:
        logger.warning("Processing failure: %s", exc.code)
        return error_response(exc.status_code, exc.code, str(exc), exc.retryable)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        issues = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return error_response(
            422, "invalid_request", "Request validation failed.", issues=issues
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error_response(
            exc.status_code,
            "not_found" if exc.status_code == 404 else "http_error",
            str(exc.detail),
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled request error", exc_info=exc)
        return error_response(
            500,
            "internal_error",
            "An internal error occurred while processing the request.",
        )
