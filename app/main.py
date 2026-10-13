"""FastAPI entry point for stateless nutrition requests."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.ai.runtime import initialize_runtime
from app.api.access_logging import AccessLoggingMiddleware
from app.api.errors import install_error_handlers
from app.config import load_environment

from .api.routes import router


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_environment(".env")
    initialize_runtime(allow_zerogpu=application.state.allow_zerogpu)
    yield


def create_app(*, application_class=FastAPI, allow_zerogpu=False) -> FastAPI:
    application = application_class(
        title="Nutrition AI Service",
        version="0.1.0",
        description="Process messages with caller-provided profiles and history.",
        lifespan=lifespan,
    )
    application.state.allow_zerogpu = allow_zerogpu
    application.add_middleware(AccessLoggingMiddleware)
    install_error_handlers(application)
    application.include_router(router)
    return application


app = create_app()
