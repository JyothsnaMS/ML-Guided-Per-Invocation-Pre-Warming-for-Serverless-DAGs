"""Stage 1: split a video into chunks. Synthetic: work time grows with video length."""
import random
import time

from common import begin, end, is_warmup, warmup_response


def handler(event, context):
    if is_warmup(event):
        return warmup_response("split", event)
    rec = begin("split")

    video_s = float(event["video_seconds"])
    n = int(event["chunks"])
    # Synthetic work: 0.5 s base + 0.03 s per second of video, plus noise.
    time.sleep(0.5 + 0.03 * video_s + random.uniform(0, 0.2))

    chunks = [{"chunk_id": i, "chunk_seconds": video_s / n} for i in range(n)]
    return {"chunks": chunks, "timing": end(rec)}