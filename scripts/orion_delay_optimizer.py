"""ORION's static pre-warming delay optimizer, reimplemented for our project.

Same method as ORION's Prewarming_Optimizer/DAG_Delay_Optimizer/Video_prewarming_optimizer/
three_components_Video_BFS.py:
  1. Take the measured run-time distributions of Split, Extract and Classify.
  2. For a candidate pair of delays (d2 = when to warm Extract, d3 = when to warm Classify,
     both counted from the moment the DAG starts), simulate N random runs and score them
     by the P95 of (utilization / end-to-end latency), as ORION does.
  3. Start both delays at 0 and keep adding STEP seconds to whichever delay improves the
     score more; stop when neither does (greedy search, which ORION calls BFS).

Two changes from ORION: every candidate is scored on the SAME simulated runs (fixed seed),
and we simulate 100,000 runs instead of 10,000, so the result is repeatable. ORION re-samples
on every call with 10,000 runs, so its answer jitters
(e.g. Extract->Classify comes out 0 or 0.1 s, Split->Extract 2.3 or 2.5 s).

Usage:
  python scripts/orion_delay_optimizer.py --orion-dir Orion-OSDI22/Prewarming_Optimizer/DAG_Delay_Optimizer/Video_prewarming_optimizer
  python scripts/orion_delay_optimizer.py --csv results/profiles.csv --init-extract 350 --init-classify 4615
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- inputs

def load_orion(folder):
    """ORION's own distribution files: one duration in ms per line."""
    def read(name):
        with open(os.path.join(folder, name)) as f:
            return np.array([float(x) for x in f if x.strip()])
    return read("dist_Split.txt"), read("dist_Extract.txt"), read("dist_Detect.txt")


def load_profiles(path, warm_only=True):
    """Our profiles.csv. Uses the slowest chunk per run (that is what the DAG waits for).
    Run times in our CSV exclude container start-up, which is what the model needs:
    start-up time is added separately via --init-extract / --init-classify."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if warm_only:
        warm = [r for r in rows if r["forced_cold"] == "False"]
        rows = warm or rows
    get = lambda k: np.array([float(r[k]) for r in rows])
    return get("split_ms"), get("extract_max_ms"), get("classify_max_ms")


# ---------------------------------------------------------------- ORION's model

def simulate(split, extract, classify, d2_s, d3_s, init_extract, init_classify):
    """ORION's per-run formulas (vectorized). All times in ms.
    Returns (score, e2e array, utilization array)."""
    d2, d3 = d2_s * 1000, d3_s * 1000
    d = np.minimum(d2, split)                    # never warm later than Split's end
    # Extract finishes after its run time, starting when Split is done AND its container is up.
    extract_done = extract + np.maximum(split, d + init_extract)
    e2e = classify + np.maximum(extract_done, d3 + init_classify)

    # ORION's utilization: 1 if no idle warm container, else shrinks with idle time
    # (Extract containers waiting for Split, Classify containers waiting for Extract).
    idle_ratio = ((split - (d + init_extract)) / (split + init_extract + extract)
                  + (extract_done - (d3 + init_classify))
                  / (split + extract + classify + init_extract + init_classify))
    util = np.where(split <= d + init_extract, 1.0, 1.0 / (1.0 + idle_ratio))

    util_per_ms = np.sort(util / e2e)
    score = util_per_ms[int(len(util_per_ms) * 0.95)]   # ORION's P95 of util/E2E
    return score, e2e, util


def optimize(split, extract, classify, init_extract, init_classify, n=100000, step=0.1, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(split, n)
    e = rng.choice(extract, n)
    c = rng.choice(classify, n)
    score = lambda a, b: simulate(s, e, c, a, b, init_extract, init_classify)[0]

    d2 = d3 = 0.0
    best = 0.0                                    # ORION starts its comparison at 0 too
    while True:
        v2, v3 = score(d2 + step, d3), score(d2, d3 + step)
        if v2 <= best and v3 <= best:
            break
        if v2 > v3:
            best, d2 = v2, round(d2 + step, 2)
        else:
            best, d3 = v3, round(d3 + step, 2)

    _, e2e, util = simulate(s, e, c, d2, d3, init_extract, init_classify)
    return {
        "delay_split_to_extract_s": d2,
        "delay_extract_to_classify_s": d3,
        "score_p95_util_per_ms": float(best),
        "est_p50_e2e_ms": float(np.percentile(e2e, 50)),
        "est_p95_e2e_ms": float(np.percentile(e2e, 95)),
        "est_mean_util": float(np.mean(util)),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--orion-dir", help="folder with ORION's dist_Split/Extract/Detect.txt")
    src.add_argument("--csv", help="our profiles.csv")
    ap.add_argument("--include-cold", action="store_true", help="CSV: also use forced-cold runs")
    ap.add_argument("--init-extract", type=float, default=350, help="Extract cold-start time, ms (ORION: 350)")
    ap.add_argument("--init-classify", type=float, default=4615, help="Classify cold-start time, ms (ORION: 4615)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "orion_delays.json"))
    args = ap.parse_args()

    if args.orion_dir:
        split, extract, classify = load_orion(args.orion_dir)
        source = f"ORION distributions ({args.orion_dir})"
    else:
        split, extract, classify = load_profiles(args.csv, warm_only=not args.include_cold)
        source = f"our profiles ({args.csv})"

    print(f"Input: {source}")
    print(f"  Split    median {np.median(split):7.0f} ms   ({len(split)} samples)")
    print(f"  Extract  median {np.median(extract):7.0f} ms")
    print(f"  Classify median {np.median(classify):7.0f} ms")
    print(f"  Cold start: Extract {args.init_extract:.0f} ms, Classify {args.init_classify:.0f} ms")

    r = optimize(split, extract, classify, args.init_extract, args.init_classify, seed=args.seed)
    r.update({"source": source, "init_extract_ms": args.init_extract, "init_classify_ms": args.init_classify})

    print(f"\nORION static delays:  Split->Extract {r['delay_split_to_extract_s']} s   "
          f"Extract->Classify {r['delay_extract_to_classify_s']} s   (from DAG start)")
    print(f"Estimated: P50 E2E {r['est_p50_e2e_ms']:.0f} ms, P95 E2E {r['est_p95_e2e_ms']:.0f} ms, "
          f"mean utilization {r['est_mean_util']:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()