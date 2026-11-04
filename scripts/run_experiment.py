"""Run the DAG many times with varied inputs and save one CSV row per run.

Local (no AWS):   python scripts/run_experiment.py --mode local --runs 20
Cloud (AWS):      python scripts/run_experiment.py --mode cloud --runs 20
Step Functions:   python scripts/run_experiment.py --mode sfn --arn <StateMachineArn> --runs 20

Options:
  --cold-every K   force a cold start before every K-th run (0 = never)
  --out PATH       CSV file to append to (default results/profiles.csv)
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FUNCTION_NAMES = ["orion-split", "orion-extract", "orion-classify"]

FIELDS = [
    "run_id", "mode", "timestamp", "forced_cold",
    "video_seconds", "chunks",
    "e2e_ms", "split_ms", "split_cold",
    "extract_max_ms", "classify_max_ms",
    "extract_cold_count", "classify_cold_count",
]



class LocalBackend:
    """Jyothsna's local Lambda emulator (local/runtime.py + local/dag.py)."""

    def __init__(self):
        from local.runtime import LocalLambda
        self.rt = LocalLambda()

    def run(self, dag_input):
        from local.dag import run_dag
        return run_dag(self.rt, dag_input)

    def force_cold(self):
        self.rt.reset()

    def close(self):
        self.rt.reset()


class CloudBackend:
    """Our own orchestrator (local/dag.py) calling the real Lambda functions."""

    def __init__(self):
        from cloud.runtime import CloudLambda
        self.rt = CloudLambda()

    def run(self, dag_input):
        from local.dag import run_dag
        return run_dag(self.rt, dag_input)

    def force_cold(self):
        self.rt.reset()

    def close(self):
        pass


class SfnBackend:
    """Step Functions runs the DAG (sanity check only: it cannot pre-warm)."""

    def __init__(self, arn):
        import boto3
        self.arn = arn
        self.sfn = boto3.client("stepfunctions")
        self.lam = boto3.client("lambda")

    def run(self, dag_input):
        resp = self.sfn.start_sync_execution(stateMachineArn=self.arn, input=json.dumps(dag_input))
        if resp["status"] != "SUCCEEDED":
            raise RuntimeError(f"{resp.get('error')}: {resp.get('cause')}")
        e2e_ms = (resp["stopDate"] - resp["startDate"]).total_seconds() * 1000
        return json.loads(resp["output"]), e2e_ms

    def force_cold(self):
        # Changing an environment variable makes Lambda discard existing containers.
        stamp = str(int(time.time()))
        for name in FUNCTION_NAMES:
            self.lam.update_function_configuration(
                FunctionName=name, Environment={"Variables": {"FORCE_COLD": stamp}})
        waiter = self.lam.get_waiter("function_updated")
        for name in FUNCTION_NAMES:
            waiter.wait(FunctionName=name)

    def close(self):
        pass


# ---------------------------------------------------------------- helpers

def random_input(rng):
    """Varied inputs, so stage durations differ from run to run."""
    return {"video_seconds": rng.randint(10, 180), "chunks": rng.choice([2, 3, 4])}


def flatten(run_id, mode, forced_cold, dag_input, output, e2e_ms):
    """Turn one DAG output into one CSV row."""
    split = output["split"]["timing"]
    extracts = [c["extract"]["timing"] for c in output["chunk_results"]]
    classifies = [c["classify"]["timing"] for c in output["chunk_results"]]
    return {
        "run_id": run_id,
        "mode": mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "forced_cold": forced_cold,
        "video_seconds": dag_input["video_seconds"],
        "chunks": dag_input["chunks"],
        "e2e_ms": round(e2e_ms),
        "split_ms": split["duration_ms"],
        "split_cold": split["cold"],
        "extract_max_ms": max(t["duration_ms"] for t in extracts),
        "classify_max_ms": max(t["duration_ms"] for t in classifies),
        "extract_cold_count": sum(t["cold"] for t in extracts),
        "classify_cold_count": sum(t["cold"] for t in classifies),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "cloud", "sfn"], required=True)
    p.add_argument("--arn", help="StateMachineArn (sfn mode only)")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--cold-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "profiles.csv"))
    args = p.parse_args()
    if args.mode == "sfn" and not args.arn:
        p.error("--arn is required in sfn mode")

    if args.mode == "local":
        backend = LocalBackend()
    elif args.mode == "cloud":
        backend = CloudBackend()
    else:
        backend = SfnBackend(args.arn)

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    raw_path = os.path.splitext(args.out)[0] + "_raw.jsonl"

    try:
        with open(args.out, "a", newline="") as f, open(raw_path, "a") as raw:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            for i in range(args.runs):
                forced = args.cold_every > 0 and i % args.cold_every == 0
                if forced:
                    backend.force_cold()
                dag_input = random_input(rng)
                output, e2e_ms = backend.run(dag_input)
                row = flatten(f"{args.mode}-{int(time.time())}-{i}", args.mode, forced,
                              dag_input, output, e2e_ms)
                writer.writerow(row)
                f.flush()
                raw.write(json.dumps({"row": row, "output": output}) + "\n")
                print(f"[{i + 1}/{args.runs}] video={row['video_seconds']:>3}s chunks={row['chunks']} "
                      f"e2e={row['e2e_ms']:>5} ms split={row['split_ms']:>5} ms "
                      f"cold classify={row['classify_cold_count']}{' (forced cold)' if forced else ''}")
    finally:
        backend.close()
    print(f"\nSaved to {args.out} (raw outputs in {raw_path})")


if __name__ == "__main__":
    main()