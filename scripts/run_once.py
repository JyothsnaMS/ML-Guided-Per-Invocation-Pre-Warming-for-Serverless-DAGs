"""Quick check: runs the same video twice. The first run is cold and the second is warm."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from local.dag import run_dag  # noqa: E402
from local.runtime import LocalLambda  # noqa: E402

rt = LocalLambda()
try:
    for label in ("First run (cold)", "Second run (warm)"):
        output, e2e_ms = run_dag(rt, {"video_seconds": 60, "chunks": 3})
        cold = sum(chunk["classify"]["timing"]["cold"] for chunk in output["chunk_results"])
        print(f"{label}: {e2e_ms / 1000:.1f} s, Classify cold starts: {cold} of 3")
finally:
    rt.reset()
