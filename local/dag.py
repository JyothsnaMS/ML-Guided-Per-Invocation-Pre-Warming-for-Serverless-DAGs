"""Runs the video pipeline, with optional warm-ups for Extract and Classify."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HOLD_S = 0.05  # a warm-up call stays busy this long, so parallel calls get separate containers


class StageWarmup:
    """Warms one stage: waits for its timer, or until a real call needs it, whichever is first."""

    def __init__(self, rt, function_name, count, delay_s):
        self.rt = rt
        self.function_name = function_name
        self.count = count
        self.delay_s = delay_s
        self.fire_now = threading.Event()
        self.ready = threading.Event()
        self.ready_at = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _ping(self, _):
        return self.rt.invoke(self.function_name, {"warmup": True, "hold_s": HOLD_S})

    def _run(self):
        self.fire_now.wait(self.delay_s)
        try:
            with ThreadPoolExecutor(max_workers=self.count) as pool:
                list(pool.map(self._ping, range(self.count)))
        except Exception as error:
            self.error = error
        finally:
            self.ready_at = time.time()
            self.ready.set()

    def wait_until_ready(self):
        """A real call has arrived. Returns (idle_ms, wait_ms) for that call."""
        arrived_at = time.time()
        self.fire_now.set()
        self.ready.wait()
        if self.error:
            raise self.error
        idle_ms = max(0, arrived_at - self.ready_at) * 1000
        wait_ms = max(0, self.ready_at - arrived_at) * 1000
        return idle_ms, wait_ms


def run_dag(rt, dag_input, policy=None):
    """Returns (output, e2e_ms). With no policy, nothing is warmed up."""
    start = time.time()
    extract_warmup = classify_warmup = None
    if policy is not None:
        delays = policy.plan(dag_input)
        if delays is not None:
            count = int(dag_input["chunks"])
            extract_warmup = StageWarmup(rt, "orion-extract", count, delays[0])
            classify_warmup = StageWarmup(rt, "orion-classify", count, delays[1])

    idle_ms = [0.0]
    wait_ms = [0.0]
    lock = threading.Lock()

    def use_warmup(warmup):
        if warmup is None:
            return
        idle, wait = warmup.wait_until_ready()
        with lock:
            idle_ms[0] += idle
            wait_ms[0] += wait

    def process_chunk(chunk):
        use_warmup(extract_warmup)
        extract = rt.invoke("orion-extract", chunk)
        use_warmup(classify_warmup)
        classify = rt.invoke("orion-classify", {
            "chunk_id": chunk["chunk_id"],
            "frames": extract["frames"],
        })
        return {**chunk, "extract": extract, "classify": classify}

    split = rt.invoke("orion-split", dag_input)
    chunks = split["chunks"]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(pool.map(process_chunk, chunks))
    e2e_ms = (time.time() - start) * 1000

    output = {
        **dag_input,
        "split": split,
        "chunk_results": chunk_results,
        "warmup_idle_ms": idle_ms[0],
        "warmup_wait_ms": wait_ms[0],
    }
    return output, e2e_ms
