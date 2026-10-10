import time

from app.ai.agent.tools.common import _envelope, _log
from app.ai.conversation.session import NLPSession


def parse_query(message: str, session: NLPSession, intent: str = "") -> dict:
    """
    Call NLPSession.process() and wrap the result in the unified envelope.

    data is the full NLPSession.process() output as-is:
        {parsed, status, prompt_for_user, profile, missing_fields}
    """
    t0 = time.perf_counter()
    try:
        result = session.process(message)
        duration = (time.perf_counter() - t0) * 1000
        _log("parse_query", intent, {"message_len": len(message)}, True, duration)
        return _envelope(True, result)
    except Exception:
        duration = (time.perf_counter() - t0) * 1000
        _log("parse_query", intent, {"message_len": len(message)}, False, duration)
        raise
