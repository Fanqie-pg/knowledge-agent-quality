"""Request-scoped business traces; opt-in, independent of any tracing vendor."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from time import perf_counter
import re


_events = ContextVar("agent_trace_events", default=None)
_SECRET_KEYS = re.compile(r"authorization|password|api[_-]?key|access[_-]?token|secret|cookie", re.I)
_SECRET_VALUE = re.compile(r"(?i)(bearer\s+)[\w.\-/+=]+|((?:api[_-]?key|password|access[_-]?token|secret)\s*[=:]\s*)[^\s,;\"']+")


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET_KEYS.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub(lambda m: (m.group(1) or m.group(2)) + "[REDACTED]", value)
    return value


def emit(stage, event, **data):
    events = _events.get()
    if events is not None:
        events.append({"sequence": len(events), "time": datetime.now(timezone.utc).isoformat(),
                       "stage": stage, "event": event, "data": redact(data)})


@contextmanager
def capture_trace():
    events = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


@contextmanager
def span(stage, **data):
    start = perf_counter()
    emit(stage, "started", **data)
    try:
        yield
    except Exception as exc:
        emit(stage, "failed", error_type=type(exc).__name__, elapsed_ms=(perf_counter() - start) * 1000)
        raise
    else:
        emit(stage, "completed", elapsed_ms=(perf_counter() - start) * 1000)
