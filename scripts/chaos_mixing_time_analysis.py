"""Chaos Mixing-Time Analysis — the analytic-appendix backbone.

The claim (analytic appendix of the chaos paper)
------------------------------------------------
The Gottwald–Melbourne 0-1 statistic K is built from the mean-square
displacement M_c(n) of the translation variables p_c(n) = Σ φ(j) cos(jc),
q_c(n) = Σ φ(j) sin(jc). For a *mixing* observable whose autocorrelation
ρ(k) is summable, the variance of the partial sums grows linearly,
M_c(n) ~ D_c · n, by the same Green–Kubo argument that gives a central-limit
diffusion rate D_c = Σ_k ρ(k) cos(kc)·(…). Linear growth ⇒ slope(log M, log n)
→ 1 ⇒ K → 1. But the linear (diffusive) regime only sets in once the
trajectory is long compared to the integrated autocorrelation (mixing) time
τ of the observable. Below that, the partial sums are still correlated and K
is suppressed.

Concrete, testable prediction
-----------------------------
The trajectory length n_½ at which K first crosses 0.5 should scale with the
observable's mixing time τ. Slowly-mixing systems (large τ — often the
bounded Hamiltonian flows) need long trajectories before K climbs to 1; fast
mixing systems (small τ — strongly dissipative attractors) reach K ≈ 1 almost
immediately. This *quantitatively explains* the trajectory-length artifact
that earlier faked a conservative/dissipative separation.

What this script measures, per chaotic system
---------------------------------------------
  τ        — integrated autocorrelation time of the first observable (steps),
             Sokal-style truncated window.
  K(n)     — 0-1 K at a ladder of trajectory lengths, over several ICs,
             reported as mean ± std (the error bands).
  n_½      — smallest ladder length whose mean K exceeds 0.5.
Then the aggregate test: Spearman ρ(τ, n_½) across systems (prediction: > 0).

Output
------
    results/chaos_mixing_time_analysis.json

Run
---
    scientist-env/bin/python3 scripts/liquid/chaos_mixing_time_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.liquid.chaos_universality_lab import (  # noqa: E402
    build_zoo, gottwald_melbourne_01_test, GLOBAL_SEED,
)

OUT = ROOT / "results" / "chaos_mixing_time_analysis.json"

# Trajectory-length ladder (steps) at which K is evaluated.
LENGTH_LADDER = [2000, 4000, 8000, 16000, 32000]


def integrated_autocorrelation_time(x: np.ndarray, c_max_lag: int = 2000) -> float:
    """Sokal-windowed integrated autocorrelation time (in steps).

    τ = 1 + 2 Σ_{k=1}^{W} ρ(k), with the window W chosen as the first lag where
    ρ(k) drops below 0.05 or goes negative — a standard automatic-windowing
    heuristic. Returns τ in trajectory steps."""
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return float("nan")
    tau = 1.0
    upper = min(c_max_lag, x.size - 2)
    for k in range(1, upper + 1):
        rho = float((x[:-k] * x[k:]).mean()) / var
        if rho < 0.05:
            break
        tau += 2.0 * rho
    return tau


def k_crossover_length(length_to_mean_K: dict) -> int | None:
    """Smallest ladder length whose mean K exceeds 0.5; None if never."""
    for n in sorted(length_to_mean_K):
        m = length_to_mean_K[n]
        if m is not None and m > 0.5:
            return n
    return None


def _perturbed_ic(base_ic: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = 0.1 * (np.abs(base_ic) + 1.0)
    return base_ic + rng.uniform(-delta, delta, size=base_ic.shape)


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation, NaN-safe, no scipy dependency."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)

    def _rank(v):
        order = v.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); ns = np.zeros(len(counts))
        for r, g in zip(ranks, inv):
            sums[g] += r; ns[g] += 1
        avg = sums / ns
        return avg[inv]

    rx, ry = _rank(xs), _rank(ys)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def run_analysis(M: int = 4, verbose: bool = True) -> dict:
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    specs = build_zoo()
    rng = np.random.default_rng(GLOBAL_SEED + 909)
    t0 = time.time()
    per_system = []
    max_n = max(LENGTH_LADDER)

    for spec in specs:
        # Restrict to genuinely chaotic families (skip pure controls; they have
        # no chaos and are not informative for the mixing-time→K-crossover law).
        if spec.family == "control":
            continue
        log(f"  · {spec.name} ({spec.family})")
        # mixing time τ from one base trajectory at max length
        base_traj = spec.integrate(spec.base_ic, max_n, spec.dt)
        tau = integrated_autocorrelation_time(base_traj[:, 0])

        # K(n) ladder over M ICs (slice prefixes of one long traj per IC)
        length_K_samples: dict[int, list[float]] = {n: [] for n in LENGTH_LADDER}
        for j in range(M):
            ic = spec.base_ic if j == 0 else _perturbed_ic(
                spec.base_ic.astype(np.float64), rng)
            traj = spec.integrate(ic, max_n, spec.dt)
            for n in LENGTH_LADDER:
                if n > traj.shape[0]:
                    continue
                K = gottwald_melbourne_01_test(traj[:n]).get("K_median", float("nan"))
                if K == K:
                    length_K_samples[n].append(float(K))

        length_to_stats = {}
        length_to_mean = {}
        for n, vals in length_K_samples.items():
            if vals:
                arr = np.array(vals)
                length_to_stats[str(n)] = {
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "n_ics": int(arr.size)}
                length_to_mean[n] = float(arr.mean())
            else:
                length_to_stats[str(n)] = {"mean": None, "std": None, "n_ics": 0}
                length_to_mean[n] = None

        n_half = k_crossover_length(length_to_mean)
        per_system.append({
            "name": spec.name,
            "family": spec.family,
            "dt": spec.dt,
            "mixing_time_steps": tau,
            "mixing_time_natural": (tau * spec.dt) if tau == tau else None,
            "K_vs_length": length_to_stats,
            "k_crossover_length_n_half": n_half,
        })

    # Aggregate prediction test: does n_½ scale with τ?
    taus = [s["mixing_time_steps"] for s in per_system]
    # use max_n as a censored value when K never crosses 0.5 (slowest mixers)
    n_halfs = [s["k_crossover_length_n_half"] if s["k_crossover_length_n_half"] is not None
               else float(max_n) for s in per_system]
    rho_tau_nhalf = _spearman(taus, n_halfs)

    # Also split by family for the narrative
    diss = [s for s in per_system if s["family"] in ("flow_dissipative", "map")]
    ham = [s for s in per_system if s["family"] == "flow_hamiltonian"]

    def _median(vals):
        v = sorted(x for x in vals if x == x)
        return float(v[len(v) // 2]) if v else None

    summary = {
        "n_chaotic_systems": len(per_system),
        "M_initial_conditions": M,
        "length_ladder": LENGTH_LADDER,
        "spearman_tau_vs_n_half": rho_tau_nhalf,
        "prediction": ("K→1 crossover length n_½ scales with the observable mixing "
                       "time τ; Spearman ρ(τ, n_½) > 0 confirms the analytic "
                       "appendix's explanation of the trajectory-length artifact."),
        "prediction_supported": (rho_tau_nhalf == rho_tau_nhalf and rho_tau_nhalf > 0.3),
        "median_mixing_time_steps_dissipative": _median([s["mixing_time_steps"] for s in diss]),
        "median_mixing_time_steps_hamiltonian": _median([s["mixing_time_steps"] for s in ham]),
        "median_n_half_dissipative": _median(
            [s["k_crossover_length_n_half"] for s in diss
             if s["k_crossover_length_n_half"] is not None]),
        "median_n_half_hamiltonian": _median(
            [s["k_crossover_length_n_half"] for s in ham
             if s["k_crossover_length_n_half"] is not None]),
    }

    return {
        "version": "chaos_mixing_time_analysis/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": GLOBAL_SEED + 909,
        "config": {"M": M, "length_ladder": LENGTH_LADDER},
        "summary": summary,
        "systems": per_system,
        "wall_seconds_total": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Chaos mixing-time / K-crossover analysis")
    ap.add_argument("--m", type=int, default=4, help="ICs per system (default 4)")
    args = ap.parse_args()
    res = run_analysis(M=args.m, verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    print(f"\nWrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)  wall {res['wall_seconds_total']}s")
    print(f"Spearman ρ(mixing time τ, K-crossover n_½) = {s['spearman_tau_vs_n_half']:.3f}")
    print(f"  prediction supported (ρ>0.3): {s['prediction_supported']}")
    print(f"  median τ  — dissipative: {s['median_mixing_time_steps_dissipative']}  "
          f"Hamiltonian: {s['median_mixing_time_steps_hamiltonian']}")
    print(f"  median n_½ — dissipative: {s['median_n_half_dissipative']}  "
          f"Hamiltonian: {s['median_n_half_hamiltonian']}")


if __name__ == "__main__":
    main()
