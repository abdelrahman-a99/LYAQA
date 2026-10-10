"""Shared result envelopes and logging for agent tools."""

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("app.ai.agent.tools")


def _envelope(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
) -> dict:
    return {
        "success": success,
        "data": data,
        "error": error,
        "error_code": error_code,
    }


def _log(
    tool_name: str, intent: str, input_summary: dict, success: bool, duration_ms: float
) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_name": tool_name,
        "intent": intent,
        "input_summary": input_summary,
        "success": success,
        "duration_ms": round(duration_ms, 1),
    }
    logger.info(json.dumps(record, ensure_ascii=False))
