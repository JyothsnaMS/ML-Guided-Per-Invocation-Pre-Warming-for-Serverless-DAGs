"""Phase 3: train LightGBM quantile models that predict Split's duration.

For each quantile tau, compares three ways of guessing how long Split will take:
  - static (ORION-style): one number for every run = the tau-quantile of past Split times
  - ML: a LightGBM quantile model that looks at this run's input (video length, chunks)
  - ML + calibration: the same model, shifted on held-out rows so coverage >= tau
plus a mean (L2) model, as the "mean vs. quantile" ablation from the proposal.

Scored with 5-fold cross-validation: MAE, pinball loss, and coverage
(how often the real Split time was at or below the prediction; ideally = tau).

Usage:
  python scripts/train_model.py results/profiles.csv
  python scripts/train_model.py --synthetic 200      # test without real data

Outputs: results/model_eval.csv, results/fig_model_quantiles.png,
         results/fig_model_vs_static.png, models/split_q*.txt + split_offsets.json (trained on all rows)
"""
import argparse
import csv
import json
import os
import random

import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["video_seconds", "chunks"]
TARGET = "split_ms"
TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------- data

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([float(r[TARGET]) for r in rows])
    return X, y


def synthetic(n, seed=0):
    """Fake data shaped like split.py with ORION-calibrated noise (for testing only)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        v, c = rng.randint(10, 180), rng.choice([2, 3, 4])
        work = (0.5 + 0.03 * v) * rng.lognormvariate(0, 0.10)
        if rng.random() < 0.02:
            work *= rng.uniform(1.5, 2.0)
        X.append([v, c])
        y.append(work * 1000)
    return np.array(X, float), np.array(y)


# ---------------------------------------------------------------- models

def params(objective, alpha=None, n_rows=100):
    p = {
        "objective": objective,
        "learning_rate": 0.05,
        "num_leaves": 8,
        "min_data_in_leaf": max(3, n_rows // 20),   # small datasets need small leaves
        "min_data_in_bin": 1,
        "verbose": -1,
        "seed": 42,
    }
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit(X, y, objective, alpha=None, rounds=300):
    ds = lgb.Dataset(X, y, feature_name=FEATURES, params={"min_data_in_bin": 1})
    return lgb.train(params(objective, alpha, len(y)), ds, num_boost_round=rounds)


def fit_calibrated(X, y, tau, seed=0):
    """Conformalized quantile regression: fit on 75% of rows, then measure on the other
    25% how far off the model is, and shift its predictions so that coverage >= tau.
    Returns (model, offset_ms); predict with model.predict(X) + offset_ms."""
    idx = np.random.default_rng(seed).permutation(len(y))
    n_cal = max(5, len(y) // 4)
    cal, fit_idx = idx[:n_cal], idx[n_cal:]
    model = fit(X[fit_idx], y[fit_idx], "quantile", tau)
    resid = np.sort(y[cal] - model.predict(X[cal]))
    k = min(len(resid) - 1, int(np.ceil((len(resid) + 1) * tau)) - 1)
    return model, float(resid[k])


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def kfold(n, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join(ROOT, "results", "profiles.csv"))
    ap.add_argument("--synthetic", type=int, default=0, help="use N fake rows instead of a CSV")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    X, y = synthetic(args.synthetic) if args.synthetic else load_csv(args.csv)
    print(f"Rows: {len(y)}  ({'synthetic' if args.synthetic else args.csv})")
    if len(y) < 20:
        print("Warning: fewer than 20 rows; results will be noisy. Aim for 100+ runs.")

    folds = kfold(len(y))
    preds = {f"ml_q{t}": np.zeros(len(y)) for t in TAUS}
    preds.update({f"static_q{t}": np.zeros(len(y)) for t in TAUS})
    preds.update({f"mlcal_q{t}": np.zeros(len(y)) for t in TAUS})
    preds["ml_mean"] = np.zeros(len(y))

    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        for t in TAUS:
            preds[f"ml_q{t}"][test] = fit(X[train], y[train], "quantile", t).predict(X[test])
            model, offset = fit_calibrated(X[train], y[train], t)
            preds[f"mlcal_q{t}"][test] = model.predict(X[test]) + offset
            preds[f"static_q{t}"][test] = np.quantile(y[train], t)
        preds["ml_mean"][test] = fit(X[train], y[train], "regression").predict(X[test])

    # ---- score
    names = {"static": "static", "ml": "ml", "mlcal": "ml+cal"}
    rows = []
    for t in TAUS:
        for kind in ("static", "ml", "mlcal"):
            q = preds[f"{kind}_q{t}"]
            rows.append({"model": f"{names[kind]} q{t}", "tau": t,
                         "mae_ms": round(float(np.mean(np.abs(y - q)))),
                         "pinball_ms": round(pinball(y, q, t), 1),
                         "coverage": round(float(np.mean(y <= q)), 2)})
    q = preds["ml_mean"]
    rows.append({"model": "ml mean (L2)", "tau": "", "mae_ms": round(float(np.mean(np.abs(y - q)))),
                 "pinball_ms": round(pinball(y, q, 0.9), 1), "coverage": round(float(np.mean(y <= q)), 2)})

    print(f"\n{'model':<16}{'MAE (ms)':>10}{'pinball (ms)':>14}{'coverage':>10}   (5-fold CV)")
    for r in rows:
        print(f"{r['model']:<16}{r['mae_ms']:>10}{r['pinball_ms']:>14}{r['coverage']:>10}")
    print("coverage = share of runs where real Split time <= prediction (target: tau).")
    print("ml+cal = ML shifted on held-out rows so coverage >= tau (use this one in the controller).")
    print("ml mean's pinball is scored at tau=0.9, to show what a mean model loses at the tail.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "model_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- chart 1: data + predicted quantile curves vs video length
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], y / 1000, s=12, alpha=0.5, label="real Split time")
    grid_v = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    grid = np.column_stack([grid_v, np.full_like(grid_v, np.median(X[:, 1]))])
    for t in TAUS:
        ax.plot(grid_v, fit(X, y, "quantile", t).predict(grid) / 1000, label=f"ML q{t}")
    ax.axhline(np.quantile(y, 0.9) / 1000, color="gray", ls="--", label="static q0.9 (ORION-style)")
    ax.set_xlabel("Input video length (s)")
    ax.set_ylabel("Split duration (s)")
    ax.set_title("Per-invocation quantile predictions vs. one fixed value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_model_quantiles.png"), dpi=150)

    # ---- chart 2: pinball loss, static vs ML, per tau
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(TAUS))
    width = 0.27
    for j, (prefix, label) in enumerate([("static", "static (ORION-style)"), ("ml q", "ML"), ("ml+cal", "ML + calibration")]):
        ax.bar(xs + (j - 1) * width, [r["pinball_ms"] for r in rows if r["model"].startswith(prefix)], width, label=label)
    ax.set_xticks(xs, [f"τ = {t}" for t in TAUS])
    ax.set_ylabel("Pinball loss (ms, lower is better)")
    ax.set_title("Predicting Split's duration: ML vs. a fixed value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_model_vs_static.png"), dpi=150)

    # ---- final models on all rows, for the Phase 4 controller
    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    offsets = {}
    for t in TAUS:
        model, offsets[str(t)] = fit_calibrated(X, y, t)
        model.save_model(os.path.join(model_dir, f"split_q{t}.txt"))
    fit(X, y, "regression").save_model(os.path.join(model_dir, "split_mean.txt"))
    with open(os.path.join(model_dir, "split_offsets.json"), "w") as f:
        json.dump(offsets, f, indent=2)   # controller: predict(X) + offsets[tau]

    print(f"\nSaved: {args.out_dir}/model_eval.csv, fig_model_quantiles.png, fig_model_vs_static.png")
    print(f"Models: {model_dir}/split_q0.5.txt, split_q0.7.txt, split_q0.9.txt, split_mean.txt, split_offsets.json")


if __name__ == "__main__":
    main()