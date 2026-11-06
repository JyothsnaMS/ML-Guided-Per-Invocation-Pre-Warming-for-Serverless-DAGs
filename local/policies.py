"""Decides when to warm up Extract and Classify for one video."""
import csv
import json
import os
import statistics

import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

INIT_EXTRACT_S = 0.35    # matches time.sleep(0.35) in src/extract.py
INIT_CLASSIFY_S = 4.615  # matches MODEL_LOAD_S in src/classify.py


class NoWarmup:
    name = "none"

    def plan(self, dag_input):
        return None


class OrionPolicy:
    """The same two timers for every video."""
    name = "orion"

    def __init__(self):
        with open(os.path.join(RESULTS, "orion_delays.json")) as f:
            delays = json.load(f)
        self.delays = (delays["delay_split_to_extract_s"], delays["delay_extract_to_classify_s"])

    def plan(self, dag_input):
        return self.delays


class MLPolicy:
    """Predicts this video's Split time, then starts warming so containers are ready on time."""

    def __init__(self, name, model_file, offset_ms, extract_ms):
        self.name = name
        self.model = lgb.Booster(model_file=model_file)
        self.offset_ms = offset_ms
        self.extract_ms = extract_ms

    def plan(self, dag_input):
        features = [[float(dag_input["video_seconds"]), float(dag_input["chunks"])]]
        split_ms = float(self.model.predict(features)[0]) + self.offset_ms
        extract_delay = split_ms / 1000 - INIT_EXTRACT_S
        classify_delay = (split_ms + self.extract_ms) / 1000 - INIT_CLASSIFY_S
        return max(0.0, extract_delay), max(0.0, classify_delay)


def typical_extract_ms():
    """The median slowest-Extract time from the profiling runs."""
    with open(os.path.join(RESULTS, "profiles.csv")) as f:
        return statistics.median(float(row["extract_max_ms"]) for row in csv.DictReader(f))


def build_policies(taus=(0.1, 0.3, 0.5, 0.9)):
    extract_ms = typical_extract_ms()
    with open(os.path.join(MODELS, "split_offsets.json")) as f:
        offsets = json.load(f)
    policies = [NoWarmup(), OrionPolicy()]
    policies.append(MLPolicy("ml mean", os.path.join(MODELS, "split_mean.txt"), 0.0, extract_ms))
    for tau in taus:
        policies.append(MLPolicy(f"ml q{tau}", os.path.join(MODELS, f"split_q{tau}.txt"),
                                 offsets[str(tau)], extract_ms))
    return policies
