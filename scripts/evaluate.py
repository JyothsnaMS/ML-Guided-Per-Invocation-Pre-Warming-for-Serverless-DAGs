"""Compares no warm-up, ORION, and ML on the same videos, starting cold every time.

Usage: python scripts/evaluate.py --runs 50
Saves results/evaluation.csv and results/fig_evaluation.png
"""
import argparse
import csv
import os
import random
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from local.dag import run_dag  # noqa: E402
from local.policies import build_policies  # noqa: E402
from local.runtime import LocalLambda  # noqa: E402

FIELDS = ["run", "policy", "video_seconds", "chunks", "e2e_ms",
          "cold_calls", "idle_ms", "wait_ms"]


def run_all(policies, runs, seed):
    rng = random.Random(seed)
    rt = LocalLambda()
    rows = []
    try:
        for run in range(runs):
            dag_input = {"video_seconds": rng.randint(10, 180), "chunks": rng.choice([2, 3, 4])}
            for policy in policies:
                rt.reset()
                output, e2e_ms = run_dag(rt, dag_input, policy)
                timings = [chunk[stage]["timing"] for chunk in output["chunk_results"]
                           for stage in ("extract", "classify")]
                rows.append({
                    "run": run,
                    "policy": policy.name,
                    "video_seconds": dag_input["video_seconds"],
                    "chunks": dag_input["chunks"],
                    "e2e_ms": round(e2e_ms),
                    "cold_calls": sum(t["cold"] for t in timings),
                    "idle_ms": round(output["warmup_idle_ms"]),
                    "wait_ms": round(output["warmup_wait_ms"]),
                })
            print(f"Finished video {run + 1} of {runs}", flush=True)
    finally:
        rt.reset()
    return rows


def summarize(rows, policies):
    summary = []
    for policy in policies:
        mine = [row for row in rows if row["policy"] == policy.name]
        e2e = [row["e2e_ms"] / 1000 for row in mine]
        summary.append({
            "policy": policy.name,
            "p50": float(np.percentile(e2e, 50)),
            "p95": float(np.percentile(e2e, 95)),
            "cold": float(np.mean([row["cold_calls"] for row in mine])),
            "idle": float(np.mean([row["idle_ms"] for row in mine])) / 1000,
            "wait": float(np.mean([row["wait_ms"] for row in mine])) / 1000,
        })
    return summary


def draw_chart(summary, path):
    panels = [("p50", "P50 total time (s)"), ("p95", "P95 total time (s)"),
              ("idle", "Warm containers idle, per video (s)")]
    names = [row["policy"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        values = [row[key] for row in summary]
        bars = ax.barh(names, values, height=0.5, color="#2a78d6")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, color="#52514e")
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.invert_yaxis() if ax is axes[0] else None
        ax.margins(x=0.15)
        ax.tick_params(colors="#52514e", length=0)
        ax.xaxis.set_visible(False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    policies = build_policies()
    rows = run_all(policies, args.runs, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "evaluation.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, policies)
    print(f"\n{'policy':<10}{'P50 (s)':>9}{'P95 (s)':>9}{'cold calls':>12}{'idle (s)':>10}{'wait (s)':>10}")
    for row in summary:
        print(f"{row['policy']:<10}{row['p50']:>9.2f}{row['p95']:>9.2f}{row['cold']:>12.1f}"
              f"{row['idle']:>10.2f}{row['wait']:>10.2f}")

    chart = os.path.join(args.out_dir, "fig_evaluation.png")
    draw_chart(summary, chart)
    print(f"\nSaved {os.path.join(args.out_dir, 'evaluation.csv')} and {chart}")


if __name__ == "__main__":
    main()
