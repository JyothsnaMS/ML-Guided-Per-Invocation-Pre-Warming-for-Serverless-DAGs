"""Stage 3: classify frames. Synthetic: a slow 'model load' at container start = big cold start."""
import random
import time

# Module-level code runs only when a new container starts. Sleeping here stands in
# for loading an ML model, so this stage has a large, controllable cold start.
MODEL_LOAD_S = 3.0
time.sleep(MODEL_LOAD_S)

from common import begin, end, is_warmup, warmup_response  # noqa: E402


def handler(event, context):
    if is_warmup(event):
        return warmup_response("classify", event)
    rec = begin("classify")
    time.sleep(0.2 + 0.01 * int(event["frames"]) + random.uniform(0, 0.1))
    return {"chunk_id": event["chunk_id"], "label": "person", "timing": end(rec)}