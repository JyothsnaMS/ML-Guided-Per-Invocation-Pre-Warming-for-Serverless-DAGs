"""Stage 2: extract frames from one chunk. Synthetic: time grows with chunk length."""
import random
import time

from common import begin, end, is_warmup, warmup_response


def handler(event, context):
    if is_warmup(event):
        return warmup_response("extract", event)
    rec = begin("extract")
    time.sleep(0.1 + 0.02 * float(event["chunk_seconds"]) + random.uniform(0, 0.1))
    return {"chunk_id": event["chunk_id"], "frames": 10, "timing": end(rec)}