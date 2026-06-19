"""
chaos_run_length.py
===================
Idea E: an effective run-length calculator for the 0-1 test.

Given an observable's autocovariance C(k) = E[phi(j)phi(j+k)], the closed-form
mean-square displacement (Eq. mc in the paper),

    M_c(n) = n C(0) + 2 sum_{k=1}^{n-1} (n-k) C(k) cos(kc),

determines the whole 0-1 K-vs-length curve. This module turns that into a
practical question: how long must a trajectory be for K to cross a threshold?

It provides two calculators:
  * effective_run_length(C, c_values, threshold) -- EXACT: smallest decimated
    sample count n at which the closed-form predicted K(median over c) crosses
    the threshold. Multiply by the 0-1 decimation stride for raw steps.
  * quick_run_length(tau, A) -- a one-scalar RULE OF THUMB N ~ A * tau, where
    tau is the integrated autocorrelation time (steps) and A is calibrated
    below. Order-of-magnitude only, and for continuous-time FLOWS; maps break
    it (per-step decorrelation makes tau ~ 1 meaningless).

Calibration and validation are read from the existing seeded artifacts
(chaos_green_kubo.json, chaos_mixing_time_analysis.json); the exact calculator
is additionally checked on a controlled AR(1) process with known tau.

Forecast (before running)
-------------------------
The exact closed form already matches the measured ladder crossover in
green_kubo (it should reproduce ~14/17 exact, ~17/17 within one ladder step).
The quick rule N ~ A*tau should hold to order of magnitude for flows
(A ~ 60-80, fold-scatter ~2-3x, with outliers ~10x) and fail on maps. The
AR(1) check should show effective_run_length growing roughly linearly in tau.
This is a practical tool, not a new scientific claim; nothing is promoted.

Output: results/chaos_run_length.json
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
GK = RESULTS / "chaos_green_kubo.json"
MT = RESULTS / "chaos_mixing_time_analysis.json"
OUT = RESULTS / "chaos_run_length.json"

LADDER = [2000, 4000, 8000, 16000, 32000]
THRESHOLD = 0.5
N_C = 12
TAU_FLOOR_STEPS = 2.0   # below this, tau is at the per-step floor (maps): rule N/A


# ── exact closed-form calculator ─────────────────────────────────────────────

def predicted_Mc(C: np.ndarray, n: int, c: float) -> float:
    """Closed form: M_c(n) = n C(0) + 2 sum_{k=1}^{n-1} (n-k) C(k) cos(kc)."""
    kmax = min(n - 1, C.size - 1)
    if kmax < 1:
        return float(n * C[0])
    ks = np.arange(1, kmax + 1)
    return float(n * C[0] + 2.0 * ((n - ks) * C[ks] * np.cos(ks * c)).sum())


def predicted_K(C: np.ndarray, n_grid: np.ndarray, c_values: np.ndarray) -> float:
    """Median over c of corr(log n, log M_c(n)) from the closed form."""
    log_n = np.log(n_grid.astype(float))
    Ks = []
    for c in c_values:
        Ms = np.array([predicted_Mc(C, int(n), float(c)) for n in n_grid])
        ok = Ms > 0
        if ok.sum() < 5:
            continue
        ln = log_n[ok] - log_n[ok].mean()
        lm = np.log(Ms[ok]); lm = lm - lm.mean()
        d = math.sqrt(float((ln * ln).sum() * (lm * lm).sum()))
        if d > 0:
            Ks.append(float((ln * lm).sum() / d))
    return float(np.median(Ks)) if Ks else float("nan")


def effective_run_length(C: np.ndarray, c_values: np.ndarray,
                         threshold: float = THRESHOLD,
                         sample_ladder=(30, 60, 125, 250, 500, 1000, 2000, 4000, 8000),
                         ) -> int | None:
    """Smallest decimated sample count n at which the closed-form predicted K
    exceeds `threshold`. Returns None if not reached on the ladder. Multiply by
    the 0-1 decimation stride to convert to raw integration steps."""
    for n in sample_ladder:
        n_grid = np.unique(np.round(np.geomspace(2, max(10, n // 5), 30)).astype(int))
        if predicted_K(C, n_grid, c_values) > threshold:
            return int(n)
    return None


# ── quick rule of thumb ──────────────────────────────────────────────────────

def quick_run_length(tau_int_steps: float, A: float) -> float:
    """Order-of-magnitude run length N ~ A * tau for continuous-time flows."""
    return float(A * tau_int_steps)


# ── calibration / validation from existing artifacts ─────────────────────────

def main():
    gk = json.loads(GK.read_text())
    mt = json.loads(MT.read_text())
    meas = {s["name"]: s.get("measured_n_half") for s in gk["systems"]}
    pred = {s["name"]: s.get("predicted_n_half") for s in gk["systems"]}
    fam = {s["name"]: s.get("family") for s in gk["systems"]}
    tau = {s["name"]: s.get("mixing_time_steps") for s in mt["systems"]}

    # quick-rule calibration on flows with measurable crossover (exclude maps:
    # tau ~ 1 step is the per-step floor and the rule does not apply).
    rows, ratios_flow = [], []
    for name in tau:
        t, m, p, f = tau[name], meas.get(name), pred.get(name), fam.get(name)
        is_map = (f == "map") or (t is not None and t <= TAU_FLOOR_STEPS)
        ratio = (m / t) if (m and t) else None
        if ratio is not None and not is_map and f != "control":
            ratios_flow.append(ratio)
        rows.append({"name": name, "family": f, "tau_steps": t,
                     "measured_n_half": m, "exact_predicted_n_half": p,
                     "n_half_over_tau": round(ratio, 1) if ratio else None,
                     "is_map_floor": bool(is_map)})

    ratios_flow = np.array(ratios_flow)
    A = float(np.median(ratios_flow))
    # fold-error of the quick rule on the calibration flows
    fold = np.maximum(ratios_flow / A, A / ratios_flow)
    quick_stats = {
        "A_median_n_half_over_tau": round(A, 1),
        "n_flows": int(ratios_flow.size),
        "ratio_iqr": [round(float(np.percentile(ratios_flow, 25)), 1),
                      round(float(np.percentile(ratios_flow, 75)), 1)],
        "ratio_range": [round(float(ratios_flow.min()), 1),
                        round(float(ratios_flow.max()), 1)],
        "median_fold_error": round(float(np.median(fold)), 2),
        "max_fold_error": round(float(fold.max()), 2),
        "frac_within_2x": round(float((fold <= 2).mean()), 2),
        "frac_within_3x": round(float((fold <= 3).mean()), 2),
    }

    # exact-calculator accuracy: closed-form predicted vs measured ladder n_half
    both = [(pred[n], meas[n]) for n in tau if pred.get(n) and meas.get(n)]
    exact_match = sum(1 for p, m in both if p == m)
    within1 = 0
    for p, m in both:
        ip = LADDER.index(p) if p in LADDER else -1
        im = LADDER.index(m) if m in LADDER else -1
        if ip >= 0 and im >= 0 and abs(ip - im) <= 1:
            within1 += 1
    exact_stats = {"n": len(both), "exact_match": exact_match,
                   "within_one_ladder_step": within1}

    # controlled AR(1) check: phi_t = rho phi_{t-1} + noise has C(k)=rho^k,
    # tau_int = (1+rho)/(1-rho). The exact calculator's n should grow with tau.
    rng = np.random.default_rng(20260619)
    c_values = rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0, size=N_C)
    ar1 = []
    for rho in (0.5, 0.8, 0.9, 0.95, 0.98):
        kmax = 4000
        C = rho ** np.arange(kmax + 1, dtype=float)  # normalized C(0)=1
        tau_int = (1.0 + rho) / (1.0 - rho)
        n = effective_run_length(C, c_values, THRESHOLD)
        ar1.append({"rho": rho, "tau_int": round(tau_int, 2),
                    "effective_n_samples": n})
    finite = [(a["tau_int"], a["effective_n_samples"]) for a in ar1
              if a["effective_n_samples"]]
    ar1_monotone = all(finite[i][1] <= finite[i + 1][1]
                       for i in range(len(finite) - 1)) if len(finite) > 1 else None

    summary = {
        "exact_calculator": exact_stats,
        "quick_rule": quick_stats,
        "quick_rule_formula": "N_steps ~ A * tau_int  (A = %.0f; flows only)" % A,
        "ar1_check": {"rows": ar1, "monotone_increasing_in_tau": ar1_monotone},
        "recommendation": (
            "Use effective_run_length(C, c_values) when the autocovariance is "
            "available: it reproduces the measured ladder crossover (%d/%d exact, "
            "%d/%d within one ladder step). The one-scalar rule N ~ %.0f*tau is an "
            "order-of-magnitude guide for flows (median fold-error %.1fx, "
            "%.0f%% within 3x) and does not apply to maps."
            % (exact_match, len(both), within1, len(both), A,
               quick_stats["median_fold_error"], 100 * quick_stats["frac_within_3x"])),
    }
    result = {
        "version": "chaos_run_length/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"green_kubo": GK.name, "mixing_time": MT.name},
        "config": {"ladder": LADDER, "threshold": THRESHOLD, "n_c": N_C,
                   "tau_floor_steps": TAU_FLOOR_STEPS},
        "summary": summary,
        "systems": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("Effective run-length calculator\n")
    print(f"  exact closed form vs measured: {exact_match}/{len(both)} exact, "
          f"{within1}/{len(both)} within one ladder step")
    print(f"  quick rule  N ~ {A:.0f} * tau  (flows, n={ratios_flow.size})")
    print(f"    ratio IQR {quick_stats['ratio_iqr']}, range {quick_stats['ratio_range']}")
    print(f"    median fold-error {quick_stats['median_fold_error']}x, "
          f"within 2x {quick_stats['frac_within_2x']}, within 3x {quick_stats['frac_within_3x']}")
    print("  AR(1) control (rho, tau_int, effective_n_samples):")
    for a in ar1:
        print(f"    rho={a['rho']:.2f}  tau={a['tau_int']:>7.2f}  n={a['effective_n_samples']}")
    print(f"    monotone in tau: {ar1_monotone}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
