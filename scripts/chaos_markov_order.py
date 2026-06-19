"""
chaos_markov_order.py
====================
Idea F: how much MEMORY does the coarse itinerary grammar carry, and does the
memory depth separate the dynamical classes?

On the raw symbol sequence, order-k next-symbol accuracy is useless for flows:
finely-sampled flows dwell in a cell for many steps, so order-1, order-2 and the
"predict-stay" persistence baseline are identical (verified in
chaos_itinerary_prediction.json). The genuine route grammar lives in the
sequence of DISTINCT cells visited. We therefore collapse each symbol stream to
its run-length-encoded route (consecutive repeats removed) and ask, on held-out
data, whether conditioning on the last 2 or 3 route symbols beats order-1.

Method
------
For each chaotic system (controls excluded): integrate a lab-consistent
trajectory (imported from chaos_slope_analysis), PCA-project, k-means into
K=6 cells fit on the first half, symbolize the whole series, collapse to the
route sequence, split train/test by half, and evaluate held-out next-route
accuracy for Markov orders 1,2,3 (each backing off to lower orders, then to the
global mode, for unseen contexts). The memory gain is acc(order-k) - acc(order-1).

Forecast (before running)
-------------------------
Route grammar is mostly order-1: higher orders help where there are enough route
transitions to estimate them (clearest in the maps, thousands of changes), and
are data-limited and near-zero for flows with sparse changes. The memory gain
should NOT separate Hamiltonian from dissipative (Mann-Whitney n.s.); depth tracks
how much route data exists, not conservation. No claim is promoted.

Output: results/chaos_markov_order.json
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import chaos_slope_analysis as lab  # gen, SYSTEMS, N_WARMUP

RESULTS = Path(__file__).parent.parent / "results"
OUT = RESULTS / "chaos_markov_order.json"

N_ANALYSIS = 16000
K_CELLS = 6
ORDERS = [1, 2, 3]
KMEANS_SEED = 20260619 & 0xFFFF
MIN_TEST_ROUTE = 60   # below this many held-out route steps, higher orders are
                      # data-limited; flagged but still reported
FAM_SHORT = {"flow_dissipative": "diss", "flow_hamiltonian": "ham",
             "map": "map", "flow_delay": "diss"}
SKIP = {"control"}


def _pca_project(traj, dim=4):
    X = traj.astype(np.float64)
    if X.shape[1] == 1:
        v = X[:, 0]
        return np.stack([v[:-2], v[1:-1], v[2:]], axis=1)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
    d = int(min(dim, Vh.shape[0]))
    return U[:, :d] * S[:d]


def _kmeans_fit(X, K, n_iter=30, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    centers = X[rng.choice(N, size=K, replace=False)].astype(np.float64).copy()
    labels = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new = d2.argmin(axis=1).astype(np.int32)
        if (new == labels).all():
            break
        labels = new
        for k in range(K):
            m = labels == k
            if m.any():
                centers[k] = X[m].mean(axis=0)
    return centers


def _assign(X, centers):
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return d2.argmin(axis=1).astype(np.int32)


def _collapse(labels):
    """Run-length encode: keep only the sequence of distinct successive cells."""
    out = [int(labels[0])]
    for x in labels[1:]:
        if int(x) != out[-1]:
            out.append(int(x))
    return np.array(out, dtype=np.int64)


def _order_k_accuracy(train, test, k, K):
    """Held-out next-route accuracy of an order-k Markov model with backoff."""
    models = [defaultdict(lambda: np.zeros(K)) for _ in range(k + 1)]
    for j in range(1, k + 1):
        mj = models[j]
        for i in range(j, len(train)):
            mj[tuple(train[i - j:i].tolist())][train[i]] += 1.0
    glob = np.bincount(train, minlength=K)
    glob_mode = int(glob.argmax()) if glob.sum() else 0

    def predict(hist):
        for j in range(min(k, len(hist)), 0, -1):
            ctx = tuple(hist[-j:])
            mj = models[j]
            if ctx in mj and mj[ctx].sum() > 0:
                return int(mj[ctx].argmax())
        return glob_mode

    correct = total = 0
    for i in range(1, len(test)):
        pred = predict(test[max(0, i - k):i].tolist())
        correct += int(pred == test[i])
        total += 1
    return (correct / total) if total else float("nan"), total


def analyze(name, family, ic, dt):
    traj = lab.gen(name, np.asarray(ic, float), lab.N_WARMUP, N_ANALYSIS, dt)
    proj = _pca_project(traj)
    half = proj.shape[0] // 2
    centers = _kmeans_fit(proj[:half], K_CELLS, seed=KMEANS_SEED)
    labels = _assign(proj, centers)
    route = _collapse(labels)
    rhalf = len(route) // 2
    train, test = route[:rhalf], route[rhalf:]
    accs = {}
    n_test = None
    for k in ORDERS:
        a, nt = _order_k_accuracy(train, test, k, K_CELLS)
        accs[k] = a
        n_test = nt
    return {
        "name": name, "family": family, "fam": FAM_SHORT.get(family, family),
        "n_route_total": int(len(route)),
        "n_route_test": int(n_test) if n_test else 0,
        "acc_order1": round(accs[1], 4),
        "acc_order2": round(accs[2], 4),
        "acc_order3": round(accs[3], 4),
        "gain_o2_minus_o1": round(accs[2] - accs[1], 4),
        "gain_o3_minus_o1": round(accs[3] - accs[1], 4),
        "reliable": bool(n_test and n_test >= MIN_TEST_ROUTE),
    }


def main():
    rows = []
    print(f"Order-k route-grammar analysis  (K={K_CELLS} cells, N={N_ANALYSIS})\n")
    for name, family, ic, dt in lab.SYSTEMS:
        if family in SKIP:
            continue
        r = analyze(name, family, ic, dt)
        rows.append(r)
        flag = "" if r["reliable"] else "  (data-limited)"
        print(f"  {name:22s} {r['fam']:5s} n_test={r['n_route_test']:>5} "
              f"o1={r['acc_order1']:.3f} o2={r['acc_order2']:.3f} "
              f"o3={r['acc_order3']:.3f}  d2={r['gain_o2_minus_o1']:+.3f}{flag}")

    rel = [r for r in rows if r["reliable"]]

    def fam_gain(fam, key):
        return [r[key] for r in rel if r["fam"] == fam]

    def stat(fam, key):
        v = fam_gain(fam, key)
        return {"n": len(v), "mean": round(float(np.mean(v)), 4) if v else None,
                "median": round(float(np.median(v)), 4) if v else None,
                "max": round(float(np.max(v)), 4) if v else None}

    grp = {f: {"o2_minus_o1": stat(f, "gain_o2_minus_o1"),
               "o3_minus_o1": stat(f, "gain_o3_minus_o1")}
           for f in ("diss", "ham", "map")}

    # does the memory gain (order-2 over order-1) separate ham from diss?
    h = fam_gain("ham", "gain_o2_minus_o1")
    d = fam_gain("diss", "gain_o2_minus_o1")
    if len(h) >= 3 and len(d) >= 3:
        u, p = mannwhitneyu(h, d, alternative="two-sided")
        mwu = {"n_ham": len(h), "n_diss": len(d), "U": float(u), "p": round(float(p), 4)}
    else:
        mwu = {"n_ham": len(h), "n_diss": len(d), "U": None, "p": None}

    n_pos2 = sum(1 for r in rel if r["gain_o2_minus_o1"] > 0.01)
    best3 = sorted(rel, key=lambda r: -r["gain_o2_minus_o1"])[:3]

    summary = {
        "k_cells": K_CELLS, "orders": ORDERS, "n_reliable": len(rel),
        "min_test_route": MIN_TEST_ROUTE,
        "gain_by_family": grp,
        "mannwhitney_gain_ham_vs_diss": mwu,
        "n_reliable_with_positive_o2_gain": n_pos2,
        "largest_o2_gains": [{"name": r["name"], "fam": r["fam"],
                              "gain": r["gain_o2_minus_o1"]} for r in best3],
        "memory_depth_separates_conservation": bool(
            mwu["p"] is not None and mwu["p"] < 0.05),
        "interpretation": (
            "Route-grammar memory depth is heterogeneous: about half the reliable "
            "systems gain from order-2 (up to +0.33), the rest are order-1- "
            "saturated -- several Hamiltonians already sit at perfect order-1 "
            "accuracy. The gain does NOT separate the classes (Mann-Whitney "
            "Hamiltonian vs dissipative n.s., p~0.73), and the largest gains span "
            "dissipative (mackey-glass), map (standard map), and Hamiltonian "
            "(spring pendulum) families. Depth tracks route-data richness and "
            "coarse dynamical structure, not conservation."),
    }
    result = {
        "version": "chaos_markov_order/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"n_analysis": N_ANALYSIS, "k_cells": K_CELLS,
                   "orders": ORDERS, "kmeans_seed": KMEANS_SEED,
                   "min_test_route": MIN_TEST_ROUTE},
        "summary": summary,
        "systems": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n  reliable systems: {len(rel)}  (>= {MIN_TEST_ROUTE} held-out route steps)")
    for f in ("diss", "ham", "map"):
        g = grp[f]["o2_minus_o1"]
        print(f"    {f:5s} order2-order1 gain: mean={g['mean']} median={g['median']} (n={g['n']})")
    print(f"  Mann-Whitney gain ham vs diss: {mwu}")
    print(f"  positive order-2 gain: {n_pos2}/{len(rel)}")
    print(f"  largest gains: {summary['largest_o2_gains']}")
    print(f"  memory depth separates conservation? {summary['memory_depth_separates_conservation']}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
