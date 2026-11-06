"""Runs the video pipeline: Split, then Extract and Classify for every chunk in parallel."""
import time
from concurrent.futures import ThreadPoolExecutor


def process_chunk(rt, chunk):
    extract = rt.invoke("orion-extract", chunk)
    classify = rt.invoke("orion-classify", {
        "chunk_id": chunk["chunk_id"],
        "frames": extract["frames"],
    })
    return {**chunk, "extract": extract, "classify": classify}


def run_dag(rt, dag_input):
    """Returns (output, e2e_ms). Works with any runtime that has invoke(name, payload)."""
    start = time.time()
    split = rt.invoke("orion-split", dag_input)
    chunks = split["chunks"]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(pool.map(lambda chunk: process_chunk(rt, chunk), chunks))
    e2e_ms = (time.time() - start) * 1000
    output = {**dag_input, "split": split, "chunk_results": chunk_results}
    return output, e2e_ms
