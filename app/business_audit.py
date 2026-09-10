from contextlib import contextmanager
from contextvars import ContextVar

from app.tracing import emit, redact

_audit = ContextVar("business_operation_audit", default=None)


@contextmanager
def collect_business_events():
    events = []
    token = _audit.set(events)
    try:
        yield events
    finally:
        _audit.reset(token)


def business_event(stage, event, **data):
    emit(stage, event, **data)
    events = _audit.get()
    if events is not None:
        events.append({"stage": stage, "event": event, "data": redact(data)})
