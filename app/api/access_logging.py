"""Log HTTP status codes and time to response headers."""

import logging
from time import perf_counter

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class AccessLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = logging.getLogger("nutrition_ai.http")
        self.logger.setLevel(logging.INFO)
        self.logger.disabled = False
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        status_code = None

        def log_status(status: int) -> None:
            elapsed_ms = (perf_counter() - started_at) * 1000
            self.logger.info(
                "%s %s -> %s (%.1f ms)",
                scope["method"],
                scope["path"],
                status,
                elapsed_ms,
            )

        async def send_with_logging(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                log_status(status_code)

            await send(message)

        try:
            await self.app(scope, receive, send_with_logging)
        except Exception:
            if status_code is None:
                log_status(500)
            raise
