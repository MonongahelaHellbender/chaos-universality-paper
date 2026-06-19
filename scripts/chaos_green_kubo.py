"""Chaos Green-Kubo Predictor — predict the 0-1 K curve from autocorrelation.

The closed form
---------------
The 0-1 test's translation variable is p_c(n)+i q_c(n) = Σ_{j≤n} φ(j) e^{ijc}.
The mean-square displacement of the increment over a window of n steps is, for
a mean-zero stationary observable with autocovariance C(k) = E[φ(j)φ(j+k)],

    M_c(n) = E[ (Δp_c)² + (Δq_c)² ]
           = Σ_{a=1}^n Σ_{b=1}^n cos((a-b)c) C(a-b)
           = Σ_{k=-(n-1)}^{n-1} (n - |k|) C(k) cos(kc)                         (★)

and the asymptotic diffusion coefficient is the Green-Kubo / Wiener-Khinchin
power spectral density of the observable evaluated at frequency c,

    D_c = lim_{n→∞} M_c(n)/n = Σ_{k=-∞}^{∞} C(k) cos(kc)
        = C(0) + 2 Σ_{k≥1} C(k) cos(kc).                                       (★★)

The 0-1 statistic K_c = corr(log n, log M_c(n)) → 1 once n is large enough that
(★) is dominated by its linear term D_c · n — i.e. once n exceeds the
correlation time. Everything the 0-1 test reports is therefore *determined by
the autocovariance C(k) alone*. This script computes the predicted K-vs-n curve
and the predicted crossover n_½ from (★)/(★★) and validates them against the
measured 0-1 test, turning the earlier Spearman correlation (ρ=0.58) into an
actual prediction.

Output
------
    results/chaos_green_kubo.json

Run
---
    scientist-env/bin/python3 scripts/liquid/chaos_green_kubo.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.liquid.chaos_universality_lab import (  # noqa: E402
    build_zoo, gottwald_melbourne_01_test, _first_decorrelation_lag, GLOBAL_SEED,
)

OUT = ROOT / "results" / "chaos_green_kubo.json"
LENGTH_LADDER = [2000, 4000, 8000, 16000, 32000]
N_C = 12  # number of random frequencies c, matching the GM test


def _decimate(traj_prefix: np.ndarray, target_samples: int = 1500) -> tuple[np.ndarray, int]:
    """Replicate the GM test's decimation: stride = first decorrelation lag,
    floored so we keep enough samples. Returns (mean-subtracted decimated phi,
    stride)."""
    phi_full = traj_prefix[:, 0].astype(np.float64)
    decorr_lag = _first_decorrelation_lag(phi_full)
    min_stride = max(1, phi_full.size // target_samples)
    stride = max(min_stride, min(decorr_lag, phi_full.size // 200))
    stride = max(stride, 1)
    phi = phi_full[::stride]
    phi = phi - phi.mean()
    return phi, stride


def _autocovariance(phi: np.ndarray, k_max: int) -> np.ndarray:
    """C(k) for k=0..k_max (biased estimator, consistent normalization)."""
    n = phi.size
    k_max = min(k_max, n - 1)
    C = np.empty(k_max + 1, dtype=np.float64)
    for k in range(k_max + 1):
        C[k] = float((phi[:n - k] * phi[k:]).mean()) if k < n else 0.0
    return C


def _predicted_M(C: np.ndarray, n: int, c: float) -> float:
    """Closed form (★): M_c(n) = n C(0) + 2 Σ_{k=1}^{n-1} (n-k) C(k) cos(kc)."""
    kmax = min(n - 1, C.size - 1)
    if kmax < 1:
        return n * C[0]
    ks = np.arange(1, kmax + 1)
    terms = (n - ks) * C[ks] * np.cos(ks * c)
    return float(n * C[0] + 2.0 * terms.sum())


def _predicted_K(C: np.ndarray, n_grid: np.ndarray, c_values: np.ndarray) -> float:
    """Predicted 0-1 K (median over c) from the closed-form M_c(n) curve."""
    Ks = []
    log_n = np.log(n_grid.astype(np.float64))
    for c in c_values:
        Ms = np.array([_predicted_M(C, int(n), float(c)) for n in n_grid])
        valid = Ms > 0
        if valid.sum() < 5:
            continue
        ln = log_n[valid] - log_n[valid].mean()
        lM = np.log(Ms[valid]); lM = lM - lM.mean()
        denom = math.sqrt(float((ln * ln).sum() * (lM * lM).sum()))
        if denom <= 0:
            continue
        Ks.append(float((ln * lM).sum() / denom))
    return float(np.median(Ks)) if Ks else float("nan")


def _n_grid_for(n_dec: int) -> np.ndarray:
    n_max = max(10, n_dec // 5)
    return np.unique(np.round(np.geomspace(2, n_max, 30)).astype(int))


def analyze_system(spec, verbose_log) -> dict:
    max_n = max(LENGTH_LADDER)
    n = max_n if spec.family != "map" else min(max_n, 20000)
    traj = spec.integrate(spec.base_ic, n, spec.dt)
    rng = np.random.default_rng(GLOBAL_SEED + 31)  # same seed as GM test's c draw
    c_values = rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0, size=N_C)

    per_length = {}
    pred_K_by_len = {}
    meas_K_by_len = {}
    D_c_repr = None
    for N in LENGTH_LADDER:
        if N > traj.shape[0]:
            continue
        prefix = traj[:N]
        phi, stride = _decimate(prefix)
        n_dec = phi.size
        if n_dec < 20:
            continue
        n_grid = _n_grid_for(n_dec)
        k_max = int(n_grid.max())
        C = _autocovariance(phi, k_max)
        pred_K = _predicted_K(C, n_grid, c_values)
        meas_K = gottwald_melbourne_01_test(prefix).get("K_median", float("nan"))
        # Green-Kubo D_c (spectral density) averaged over c, on this prefix
        D_c = []
        ks = np.arange(1, C.size)
        for c in c_values:
            D_c.append(float(C[0] + 2.0 * (C[ks] * np.cos(ks * c)).sum()))
        D_c_mean = float(np.mean(D_c))
        if N == LENGTH_LADDER[-1] or D_c_repr is None:
            D_c_repr = D_c_mean
        per_length[str(N)] = {
            "predicted_K": pred_K, "measured_K": meas_K,
            "stride": int(stride), "n_decimated": int(n_dec),
            "green_kubo_D_c_mean": D_c_mean,
        }
        pred_K_by_len[N] = pred_K
        meas_K_by_len[N] = meas_K

    def _crossover(K_by_len):
        for N in sorted(K_by_len):
            v = K_by_len[N]
            if v == v and v > 0.5:
                return N
        return None

    return {
        "name": spec.name,
        "family": spec.family,
        "green_kubo_D_c_mean": D_c_repr,
        "per_length": per_length,
        "predicted_n_half": _crossover(pred_K_by_len),
        "measured_n_half": _crossover(meas_K_by_len),
    }


def _pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    x = x - x.mean(); y = y - y.mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def run(verbose=True) -> dict:
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    specs = [s for s in build_zoo() if s.family != "control"]
    t0 = time.time()
    per_system = []
    for spec in specs:
        log(f"  · {spec.name} ({spec.family})")
        try:
            per_system.append(analyze_system(spec, log))
        except Exception as exc:
            per_system.append({"name": spec.name, "family": spec.family,
                               "error": f"{type(exc).__name__}: {exc}"})

    # Validation: predicted K vs measured K across all (system, length) points
    pred_pts, meas_pts = [], []
    for s in per_system:
        for v in s.get("per_length", {}).values():
            pred_pts.append(v["predicted_K"]); meas_pts.append(v["measured_K"])
    K_pearson = _pearson(pred_pts, meas_pts)
    K_mae = float(np.mean([abs(p - m) for p, m in zip(pred_pts, meas_pts)
                           if p == p and m == m])) if pred_pts else float("nan")

    # n_half agreement
    nh_pred = [s.get("predicted_n_half") for s in per_system if "per_length" in s]
    nh_meas = [s.get("measured_n_half") for s in per_system if "per_length" in s]
    cap = float(max(LENGTH_LADDER) * 2)
    nh_pred_c = [x if x is not None else cap for x in nh_pred]
    nh_meas_c = [x if x is not None else cap for x in nh_meas]
    nh_pearson = _pearson(nh_pred_c, nh_meas_c)
    nh_exact = sum(1 for p, m in zip(nh_pred, nh_meas) if p == m)
    nh_within1 = 0
    for p, m in zip(nh_pred, nh_meas):
        if p is None and m is None:
            nh_within1 += 1; continue
        if p is None or m is None:
            continue
        ip = LENGTH_LADDER.index(p) if p in LENGTH_LADDER else -1
        im = LENGTH_LADDER.index(m) if m in LENGTH_LADDER else -1
        if ip >= 0 and im >= 0 and abs(ip - im) <= 1:
            nh_within1 += 1

    summary = {
        "n_chaotic_systems": len([s for s in per_system if "per_length" in s]),
        "closed_form": "M_c(n) = n C(0) + 2 Σ_{k=1}^{n-1} (n-k) C(k) cos(kc); D_c = C(0)+2Σ C(k)cos(kc)",
        "predicted_vs_measured_K_pearson": K_pearson,
        "predicted_vs_measured_K_mae": K_mae,
        "n_half_pearson": nh_pearson,
        "n_half_exact_match": nh_exact,
        "n_half_within_one_ladder_step": nh_within1,
        "n_systems": len(nh_pred),
        "prediction_validated": (K_pearson == K_pearson and K_pearson > 0.8
                                 and K_mae == K_mae and K_mae < 0.2),
    }
    return {
        "version": "chaos_green_kubo/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": GLOBAL_SEED + 31,
        "config": {"length_ladder": LENGTH_LADDER, "n_c": N_C},
        "summary": summary,
        "systems": per_system,
        "wall_seconds_total": round(time.time() - t0, 1),
    }


def main():
    res = run(verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    print(f"\nWrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)  wall {res['wall_seconds_total']}s")
    print(f"Closed-form Green-Kubo prediction vs measured 0-1 K:")
    print(f"  predicted-vs-measured K  Pearson r = {s['predicted_vs_measured_K_pearson']:.3f}, "
          f"MAE = {s['predicted_vs_measured_K_mae']:.3f}")
    print(f"  n_½ Pearson r = {s['n_half_pearson']:.3f}  | "
          f"exact match {s['n_half_exact_match']}/{s['n_systems']}, "
          f"within 1 ladder step {s['n_half_within_one_ladder_step']}/{s['n_systems']}")
    print(f"  prediction validated: {s['prediction_validated']}")


if __name__ == "__main__":
    main()
