"""Chaos Itinerary Prediction — predict the climate, not the weather.

The thesis
----------
Deterministic chaos makes the *trajectory* (the weather) unpredictable past a
few Lyapunov times. But the *itinerary* — which coarse region of the attractor
the system visits next (the climate) — can retain real predictability. This
script tests that rigorously and out-of-sample.

Method (rigorous, unlike the in-sample grammar in the main lab)
---------------------------------------------------------------
For each chaotic system:
  1. Integrate a long trajectory; PCA-project; fit k-means cell centers on the
     FIRST HALF only (train), then symbolize the whole series by nearest center.
  2. Split the symbol sequence: train = first half, test = held-out second half.
  3. Fit three predictors on train, evaluate next-symbol top-1 accuracy on the
     held-out test transitions:
       - order-0  : always predict the most frequent symbol (stationary mode).
                    This is the baseline — the predictability you get for free.
       - order-1  : first-order Markov, predict argmax P(s_{t+1} | s_t).
       - order-2  : second-order Markov with backoff to order-1 for unseen pairs.
  4. Lift = acc(order-1) − acc(order-0): the genuine itinerary predictability
     above the trivial baseline. Repeated for K ∈ {4, 6, 8} cells.

A positive, stable lift on held-out data — while the microstate is
Lyapunov-unpredictable — is the quantitative form of "predict the climate."

Output
------
    results/chaos_itinerary_prediction.json

Run
---
    scientist-env/bin/python3 scripts/liquid/chaos_itinerary_prediction.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # dist/chaos-universality-paper
sys.path.insert(0, str(ROOT / "scripts"))

from chaos_universality_lab import build_zoo, GLOBAL_SEED  # noqa: E402

OUT = ROOT / "results" / "chaos_itinerary_prediction.json"


def _kmeans_fit(X: np.ndarray, K: int, n_iter: int = 30, seed: int = 0):
    """k-means returning final centers (fit on the given data)."""
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


def _assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return d2.argmin(axis=1).astype(np.int32)


def _pca_project(traj: np.ndarray, dim: int = 4) -> np.ndarray:
    X = traj.astype(np.float64)
    if X.shape[1] == 1:
        v = X[:, 0]
        # delay embedding for 1-D systems
        return np.stack([v[:-2], v[1:-1], v[2:]], axis=1)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
    d = int(min(dim, Vh.shape[0]))
    return U[:, :d] * S[:d]


def _order0_accuracy(train: np.ndarray, test_src: np.ndarray,
                     test_dst: np.ndarray, K: int) -> float:
    counts = np.bincount(train, minlength=K)
    mode = int(counts.argmax())
    return float((test_dst == mode).mean()) if test_dst.size else float("nan")


def _persistence_accuracy(test_src: np.ndarray, test_dst: np.ndarray) -> float:
    """Predict next symbol = current symbol. This is the dwell-time baseline:
    finely-sampled flows stay in the same cell for many steps, so this scores
    high for trivial sampling reasons. Genuine itinerary structure must beat it."""
    return float((test_src == test_dst).mean()) if test_dst.size else float("nan")


def _transition_conditioned_accuracy(tr_src, tr_dst, te_src, te_dst, K) -> dict:
    """Among test steps where the cell ACTUALLY CHANGES (true next != current),
    how often does the order-1 Markov predict the correct destination? Compared
    against the per-source out-neighbor mode restricted to changes. This isolates
    the genuine transition grammar from dwell-time persistence."""
    # order-1 transition counts EXCLUDING self-transitions (the grammar of moves)
    P = np.zeros((K, K))
    for s, d in zip(tr_src, tr_dst):
        if s != d:
            P[s, d] += 1.0
    pred_move = P.argmax(axis=1)
    rows = P.sum(axis=1)
    # global most-common destination among moves, as fallback
    move_mask_tr = tr_src != tr_dst
    if move_mask_tr.any():
        global_move_mode = int(np.bincount(tr_dst[move_mask_tr], minlength=K).argmax())
    else:
        global_move_mode = 0
    pred_move = np.where(rows > 0, pred_move, global_move_mode)
    # evaluate only on test steps that are actual changes
    change = te_src != te_dst
    n_change = int(change.sum())
    if n_change == 0:
        return {"transition_accuracy": float("nan"),
                "transition_baseline": float("nan"),
                "n_changes": 0, "change_fraction": 0.0}
    acc = float((pred_move[te_src[change]] == te_dst[change]).mean())
    # baseline: always guess the globally-most-common move destination
    base = float((te_dst[change] == global_move_mode).mean())
    return {"transition_accuracy": acc, "transition_baseline": base,
            "n_changes": n_change,
            "change_fraction": float(n_change / te_dst.size)}


def _order1_accuracy(tr_src, tr_dst, test_src, test_dst, K) -> float:
    P = np.zeros((K, K))
    for s, d in zip(tr_src, tr_dst):
        P[s, d] += 1.0
    pred = P.argmax(axis=1)
    # for unseen source states (all-zero row) fall back to global mode
    row_sums = P.sum(axis=1)
    global_mode = int(np.bincount(tr_dst, minlength=K).argmax())
    pred = np.where(row_sums > 0, pred, global_mode)
    return float((pred[test_src] == test_dst).mean()) if test_dst.size else float("nan")


def _order2_accuracy(labels_train, test_pairs_prev, test_pairs_src, test_dst,
                     tr_src, tr_dst, K) -> float:
    # second-order counts keyed by (prev, cur)
    cnt = defaultdict(lambda: np.zeros(K))
    for a, b, c in zip(labels_train[:-2], labels_train[1:-1], labels_train[2:]):
        cnt[(int(a), int(b))][int(c)] += 1.0
    # order-1 backoff table
    P1 = np.zeros((K, K))
    for s, d in zip(tr_src, tr_dst):
        P1[s, d] += 1.0
    p1_pred = P1.argmax(axis=1)
    p1_rows = P1.sum(axis=1)
    global_mode = int(np.bincount(tr_dst, minlength=K).argmax())
    correct = 0
    total = test_dst.size
    for prev, cur, dst in zip(test_pairs_prev, test_pairs_src, test_dst):
        key = (int(prev), int(cur))
        if key in cnt and cnt[key].sum() > 0:
            pred = int(cnt[key].argmax())
        elif p1_rows[cur] > 0:
            pred = int(p1_pred[cur])
        else:
            pred = global_mode
        if pred == dst:
            correct += 1
    return float(correct / total) if total else float("nan")


def analyze_system(spec, K_values, n_steps_flow, n_steps_map,
                   seed: int) -> dict:
    n = n_steps_map if spec.family == "map" else n_steps_flow
    traj = spec.integrate(spec.base_ic, n, spec.dt)
    proj = _pca_project(traj)
    half = proj.shape[0] // 2
    out_K = {}
    for K in K_values:
        if proj.shape[0] < 4 * K:
            continue
        centers = _kmeans_fit(proj[:half], K, seed=seed)
        labels = _assign(proj, centers)
        labels_train = labels[:half]
        labels_test = labels[half:]
        # transitions
        tr_src, tr_dst = labels_train[:-1], labels_train[1:]
        te_src, te_dst = labels_test[:-1], labels_test[1:]
        acc0 = _order0_accuracy(labels_train, te_src, te_dst, K)
        acc1 = _order1_accuracy(tr_src, tr_dst, te_src, te_dst, K)
        accp = _persistence_accuracy(te_src, te_dst)
        # order-2 needs prev,cur,dst on test
        te_prev = labels_test[:-2]
        te_cur = labels_test[1:-1]
        te_d2 = labels_test[2:]
        acc2 = _order2_accuracy(labels_train, te_prev, te_cur, te_d2,
                                tr_src, tr_dst, K)
        trans = _transition_conditioned_accuracy(tr_src, tr_dst, te_src, te_dst, K)
        out_K[str(K)] = {
            "order0_baseline": acc0,
            "persistence_baseline": accp,
            "order1_markov": acc1,
            "order2_markov": acc2,
            "lift_order1_minus_order0": (acc1 - acc0) if (acc1 == acc1 and acc0 == acc0) else None,
            "lift_order1_minus_persistence": (acc1 - accp) if (acc1 == acc1 and accp == accp) else None,
            "transition_conditioned": trans,
            "n_test_transitions": int(te_dst.size),
        }
    rep = out_K.get("6") or (next(iter(out_K.values())) if out_K else {})
    rep_trans = rep.get("transition_conditioned", {}) or {}
    return {
        "name": spec.name,
        "family": spec.family,
        "by_K": out_K,
        "rep_lift": rep.get("lift_order1_minus_order0"),
        "rep_lift_vs_persistence": rep.get("lift_order1_minus_persistence"),
        "rep_order1": rep.get("order1_markov"),
        "rep_order0": rep.get("order0_baseline"),
        "rep_persistence": rep.get("persistence_baseline"),
        "rep_transition_acc": rep_trans.get("transition_accuracy"),
        "rep_transition_baseline": rep_trans.get("transition_baseline"),
        "rep_change_fraction": rep_trans.get("change_fraction"),
    }


def run(K_values=(4, 6, 8), n_steps_flow=16000, n_steps_map=12000,
        verbose=True) -> dict:
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    specs = build_zoo()
    t0 = time.time()
    per_system = []
    for spec in specs:
        log(f"  · {spec.name} ({spec.family})")
        try:
            entry = analyze_system(spec, K_values, n_steps_flow, n_steps_map,
                                   seed=int(GLOBAL_SEED & 0xffff))
        except Exception as exc:
            entry = {"name": spec.name, "family": spec.family,
                     "error": f"{type(exc).__name__}: {exc}",
                     "rep_lift": None, "rep_order1": None, "rep_order0": None}
        per_system.append(entry)

    chaotic = [s for s in per_system
               if s["family"] in ("flow_dissipative", "flow_hamiltonian", "map")
               and s.get("rep_lift") is not None]
    controls = [s for s in per_system
                if s["family"] == "control" and s.get("rep_lift") is not None]

    def _mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return float(np.mean(xs)) if xs else None

    chaotic_lift = _mean([s["rep_lift"] for s in chaotic])
    chaotic_acc1 = _mean([s["rep_order1"] for s in chaotic])
    chaotic_acc0 = _mean([s["rep_order0"] for s in chaotic])
    chaotic_persist = _mean([s["rep_persistence"] for s in chaotic])
    chaotic_lift_persist = _mean([s["rep_lift_vs_persistence"] for s in chaotic])
    chaotic_trans = _mean([s["rep_transition_acc"] for s in chaotic])
    chaotic_trans_base = _mean([s["rep_transition_baseline"] for s in chaotic])
    chaotic_change_frac = _mean([s["rep_change_fraction"] for s in chaotic])
    n_positive = sum(1 for s in chaotic if s["rep_lift"] is not None and s["rep_lift"] > 0.01)
    # the HONEST test: does the transition grammar beat its own baseline?
    n_trans_beats = sum(1 for s in chaotic
                        if s["rep_transition_acc"] is not None
                        and s["rep_transition_baseline"] is not None
                        and s["rep_transition_acc"] > s["rep_transition_baseline"] + 0.02)

    summary = {
        "n_chaotic_systems": len(chaotic),
        "K_values": list(K_values),
        "thesis": ("Held-out next-symbol (itinerary) prediction beats baselines "
                   "across chaotic systems even though the microstate trajectory "
                   "is Lyapunov-unpredictable. We separate trivial dwell-time "
                   "persistence (predict stay-in-cell) from genuine transition "
                   "grammar (predict the correct destination when the cell changes)."),
        "mean_order1_accuracy_chaotic": chaotic_acc1,
        "mean_order0_baseline_chaotic": chaotic_acc0,
        "mean_persistence_baseline_chaotic": chaotic_persist,
        "mean_lift_vs_order0": chaotic_lift,
        "mean_lift_vs_persistence": chaotic_lift_persist,
        "mean_change_fraction": chaotic_change_frac,
        "mean_transition_conditioned_accuracy": chaotic_trans,
        "mean_transition_baseline": chaotic_trans_base,
        "n_systems_with_positive_lift_vs_order0": n_positive,
        "n_systems_transition_grammar_beats_baseline": n_trans_beats,
        "n_chaotic_evaluated": len(chaotic),
        "weak_thesis_supported_vs_order0": (chaotic_lift is not None and chaotic_lift > 0.02
                             and n_positive >= 0.7 * max(len(chaotic), 1)),
        "strong_thesis_supported_genuine_grammar": (
            chaotic_trans is not None and chaotic_trans_base is not None
            and chaotic_trans > chaotic_trans_base + 0.05
            and n_trans_beats >= 0.6 * max(len(chaotic), 1)),
        "honest_note": ("Much of the raw 0.9+ next-symbol accuracy is dwell-time "
                        "persistence (finely-sampled flows stay in-cell). The "
                        "transition-conditioned numbers isolate the genuine, "
                        "non-trivial itinerary grammar."),
    }
    return {
        "version": "chaos_itinerary_prediction/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": GLOBAL_SEED,
        "config": {"K_values": list(K_values),
                   "n_steps_flow": n_steps_flow, "n_steps_map": n_steps_map,
                   "train_test_split": "first half train, second half held-out test"},
        "summary": summary,
        "systems": per_system,
        "wall_seconds_total": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Out-of-sample itinerary prediction")
    ap.add_argument("--flow-steps", type=int, default=16000)
    ap.add_argument("--map-steps", type=int, default=12000)
    args = ap.parse_args()
    res = run(n_steps_flow=args.flow_steps, n_steps_map=args.map_steps, verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    print(f"\nWrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)  wall {res['wall_seconds_total']}s")
    print(f"Held-out itinerary prediction across {s['n_chaotic_systems']} chaotic systems:")
    print(f"  order-1 Markov acc      : {s['mean_order1_accuracy_chaotic']:.3f}")
    print(f"  order-0 baseline        : {s['mean_order0_baseline_chaotic']:.3f}  (lift {s['mean_lift_vs_order0']:+.3f})")
    print(f"  persistence baseline    : {s['mean_persistence_baseline_chaotic']:.3f}  (lift {s['mean_lift_vs_persistence']:+.3f})")
    print(f"  --- genuine grammar (transition-conditioned, change_frac={s['mean_change_fraction']:.2f}) ---")
    print(f"  transition accuracy     : {s['mean_transition_conditioned_accuracy']:.3f}")
    print(f"  transition baseline     : {s['mean_transition_baseline']:.3f}")
    print(f"  grammar beats baseline  : {s['n_systems_transition_grammar_beats_baseline']}/{s['n_chaotic_evaluated']} systems")
    print(f"  weak thesis (vs order-0): {s['weak_thesis_supported_vs_order0']}")
    print(f"  STRONG thesis (genuine) : {s['strong_thesis_supported_genuine_grammar']}")


if __name__ == "__main__":
    main()
