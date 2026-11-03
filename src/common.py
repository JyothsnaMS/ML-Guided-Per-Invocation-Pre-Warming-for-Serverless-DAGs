"""Shared helpers for every stage: timing records and cold-start detection."""
import time
import uuid

# Module-level code runs once per container, when Lambda creates it (a cold start).
CONTAINER_ID = uuid.uuid4().hex[:8]
LOADED_AT_MS = int(time.time() * 1000)
_invocations = 0


def now_ms():
    return int(time.time() * 1000)


def begin(stage):
    """Call at the start of a handler. Returns a timing record to fill in."""
    global _invocations
    _invocations += 1
    start = now_ms()
    return {
        "stage": stage,
        "container_id": CONTAINER_ID,
        "cold": _invocations == 1,
        "container_age_ms": start - LOADED_AT_MS,
        "start_ms": start,
    }


def end(record):
    record["end_ms"] = now_ms()
    record["duration_ms"] = record["end_ms"] - record["start_ms"]
    return record


def is_warmup(event):
    return isinstance(event, dict) and event.get("warmup") is True


def warmup_response(stage, event):
    """A pre-warming ping: stay busy briefly so parallel pings land on separate containers."""
    rec = begin(stage)
    time.sleep(event.get("hold_s", 0.3))
    rec["warmup"] = True
    return end(rec)