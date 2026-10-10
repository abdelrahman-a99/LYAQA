"""FastAPI entry point for stateless nutrition requests."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import install_error_handlers
from app.config import load_environment

from .api.routes import router


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_environment(".env")
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Nutrition AI Service",
        version="0.1.0",
        description="Process messages with caller-provided profiles and history.",
        lifespan=lifespan,
    )
    install_error_handlers(application)
    application.include_router(router)
    return application


app = create_app()
