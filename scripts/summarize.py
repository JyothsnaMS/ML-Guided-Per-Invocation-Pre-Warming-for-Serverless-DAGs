"""Summarize a profiles CSV and draw the two skeleton charts.

Usage: python scripts/summarize.py results/profiles.csv
"""
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("video_seconds", "chunks", "e2e_ms", "split_ms", "classify_cold_count"):
            r[k] = float(r[k])
    return rows


def main(path):
    rows = load(path)
    cold = [r for r in rows if r["classify_cold_count"] > 0]
    warm = [r for r in rows if r["classify_cold_count"] == 0]
    print(f"Runs: {len(rows)} (with Classify cold starts: {len(cold)}, fully warm: {len(warm)})")

    # Inputs differ between runs, so compare the time AFTER Split finishes
    # (Extract + Classify + overheads): that is where Classify's cold start lands.
    for r in rows:
        r["after_split_ms"] = r["e2e_ms"] - r["split_ms"]
    if cold and warm:
        c = st.median(r["after_split_ms"] for r in cold)
        w = st.median(r["after_split_ms"] for r in warm)
        print(f"Median time after Split  cold: {c:.0f} ms  warm: {w:.0f} ms  "
              f"cold-start penalty: {c - w:.0f} ms ({c / w:.1f}x)")

    xs, ys = [r["video_seconds"] for r in rows], [r["split_ms"] for r in rows]
    if len(rows) > 2:
        print(f"Correlation(video_seconds, split_ms) = {st.correlation(xs, ys):.2f} "
              f"(close to 1 = Split time is predictable from the input)")

    out_dir = os.path.dirname(path) or "."

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([[r["after_split_ms"] / 1000 for r in warm] or [0],
                [r["after_split_ms"] / 1000 for r in cold] or [0]])
    ax.set_xticks([1, 2], ["warm", "with cold starts"])
    ax.set_ylabel("Time after Split finishes (s)")
    ax.set_title("Cold starts in later stages add latency")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cold_vs_warm.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, [y / 1000 for y in ys])
    ax.set_xlabel("Input video length (s)")
    ax.set_ylabel("Split duration (s)")
    ax.set_title("Upstream (Split) time varies with the input")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_split_vs_input.png"), dpi=150)

    print(f"Charts saved in {out_dir}/: fig_cold_vs_warm.png, fig_split_vs_input.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/profiles.csv")