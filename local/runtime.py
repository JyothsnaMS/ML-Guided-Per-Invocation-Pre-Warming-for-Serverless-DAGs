"""A pretend AWS Lambda that runs on your laptop.

Each "container" is a separate Python process. It loads a stage's code once when it
starts (that is the cold start) and then stays alive to handle more calls, one at a time.
"""
import json
import os
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
MODULES = {"orion-split": "split", "orion-extract": "extract", "orion-classify": "classify"}

WORKER_CODE = r"""
import json, sys, traceback
sys.path.insert(0, sys.argv[1])
channel = sys.stdout
sys.stdout = sys.stderr
module = __import__(sys.argv[2])
channel.write("ready\n")
channel.flush()
for line in sys.stdin:
    try:
        reply = {"result": module.handler(json.loads(line), None)}
    except Exception:
        reply = {"error": traceback.format_exc()}
    channel.write(json.dumps(reply) + "\n")
    channel.flush()
"""


class Container:
    def __init__(self, module):
        self.process = subprocess.Popen(
            [sys.executable, "-c", WORKER_CODE, SRC_DIR, module],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        if self.process.stdout.readline().strip() != "ready":
            raise RuntimeError(f"Container for {module} failed to start")

    def call(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Container stopped unexpectedly")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def stop(self):
        self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class LocalLambda:
    def __init__(self):
        self.lock = threading.Lock()
        self.idle = {name: [] for name in MODULES}
        self.containers = []

    def invoke(self, function_name, payload):
        """Use an idle container if there is one. Otherwise start a new one (cold start)."""
        with self.lock:
            idle = self.idle[function_name]
            container = idle.pop() if idle else None
        if container is None:
            container = Container(MODULES[function_name])
            with self.lock:
                self.containers.append(container)
        try:
            return container.call(payload)
        finally:
            with self.lock:
                self.idle[function_name].append(container)

    def reset(self):
        """Stop every container, so the next calls are all cold starts."""
        with self.lock:
            containers = self.containers
            self.containers = []
            self.idle = {name: [] for name in MODULES}
        for container in containers:
            container.stop()
