"""
chaos_decay_law.py
==================
Idea D: Is the FUNCTIONAL FORM of the observable's autocorrelation decay
(exponential vs power-law) a conservative/dissipative classifier?

The mixing-time analysis already gives a decay *timescale* and correlates it
with the 0-1 crossover n_half. This asks the orthogonal question about decay
*shape*: a short exponential tail signals hyperbolic mixing; a heavy power-law
tail signals sticky / intermittent dynamics (long-time correlations). Sticky
behavior is classically associated with mixed-phase-space Hamiltonians, so a
naive expectation is that the decay law splits the two classes.

Method
------
For each chaotic system (controls excluded, as in the Green-Kubo script) we
ensemble-average the normalized autocorrelation rho(k) = C(k)/C(0) of the first
state coordinate over M=4 slightly-perturbed initial conditions (matching the
mixing-time protocol), using lab-consistent trajectories imported from
chaos_slope_analysis. Over the initial decay window [1, k*] (down to
rho = 0.05) we fit two two-parameter laws by least squares in log space,

    exponential : rho(k) = A exp(-k / tau)     (ln rho linear in k)
    power-law   : rho(k) = A k^{-p}            (ln rho linear in ln k)

and compare R^2. We record the winner, the margin R^2_pow - R^2_exp, the decay
length k* (lags to reach rho=0.05), and join measured n_half (Green-Kubo) and
mixing_time_steps (mixing-time analysis) by name.

Forecast (before running)
-------------------------
Over the resolvable window both 2-parameter laws fit comparably, so the
exp-vs-power label is suggestive, not definitive. Expect NO clean Hamiltonian /
dissipative split: the heavy-tailed (power-favored or very slow) systems should
span classes -- the sticky Hamiltonians (Yang-Mills, coupled Duffing) together
with a map (standard map), all of which have large mixing times -- while many
Hamiltonians mix exponentially. The robust, conservation-blind signal is the
decay length k*, which should track n_half (slow decay -> slow K convergence),
echoing the mixing-time result. No claim is promoted.

Output: results/chaos_decay_law.json
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import chaos_slope_analysis as lab  # gen, SYSTEMS, N_WARMUP (same dist/scripts dir)

RESULTS = Path(__file__).parent.parent / "results"
GK = RESULTS / "chaos_green_kubo.json"
MT = RESULTS / "chaos_mixing_time_analysis.json"
OUT = RESULTS / "chaos_decay_law.json"

N_ANALYSIS = 16000      # steps after warmup, in the Green-Kubo length ladder
M_ENSEMBLE = 4          # initial conditions, matching the mixing-time protocol
PERTURB = 1e-4          # IC perturbation to decorrelate ensemble members
RHO_FLOOR = 0.05        # fit the decay down to this autocorrelation level
MIN_FIT_PTS = 6         # fewer resolvable lags than this -> decay too fast to fit
SEED = 20260619
# delay/control taxonomy: group the delay system with dissipative for the
# two-class contrast, and skip non-mixing controls (as Green-Kubo does).
SKIP_FAMILIES = {"control"}
NAME_MAP = {"mackey_glass": "mackey_glass_tau17"}  # slope name -> lab/json name


def _autocorr(x: np.ndarray, kmax: int) -> np.ndarray:
    """Normalized autocorrelation rho(k)=C(k)/C(0), k=0..kmax, via FFT."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if not np.any(x):
        return np.zeros(kmax + 1)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    f = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(f * np.conj(f), n=nfft)[:kmax + 1]
    if ac[0] <= 0:
        return np.zeros(kmax + 1)
    return ac / ac[0]


def _fit_loglin(xk: np.ndarray, y: np.ndarray):
    """Least-squares y = slope*xk + intercept; return slope, intercept, R^2."""
    A = np.vstack([xk, np.ones_like(xk)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ sol
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(sol[0]), float(sol[1]), r2


def _load_side():
    gk = json.loads(GK.read_text())
    mt = json.loads(MT.read_text())
    nhalf = {s["name"]: s.get("measured_n_half") for s in gk["systems"]}
    mtime = {s["name"]: s.get("mixing_time_steps") for s in mt["systems"]}
    return nhalf, mtime


def analyze(name, family, ic, dt, kmax, idx, seed=SEED):
    rng = np.random.default_rng([seed, idx])  # deterministic per-(system,seed)
    base = np.asarray(ic, float)
    rhos = []
    for _ in range(M_ENSEMBLE):
        ic_m = base + rng.normal(0.0, PERTURB, size=base.shape)
        traj = lab.gen(name, ic_m, lab.N_WARMUP, N_ANALYSIS, dt)
        rhos.append(_autocorr(traj[:, 0], kmax))
    rho = np.mean(rhos, axis=0)

    # decay window: lags 1..k*-1 where rho stays >= RHO_FLOOR
    kstar = kmax
    for k in range(1, kmax + 1):
        if rho[k] < RHO_FLOOR:
            kstar = k
            break
    ks = np.arange(1, kstar)
    win_rho = rho[1:kstar]
    n_pts = int(ks.size)

    rec = {"name": name, "family": family, "rho1": float(rho[1]),
           "k_decay": int(kstar), "n_fit_pts": n_pts}
    if n_pts < MIN_FIT_PTS:
        rec.update({"class": "trivial", "note": "decay too fast to fit",
                    "r2_exp": None, "r2_pow": None, "margin_pow_minus_exp": None,
                    "tau_exp": None, "p_pow": None})
        return rec, rho
    y = np.log(win_rho)
    se, _, r2e = _fit_loglin(ks.astype(float), y)
    sp, _, r2p = _fit_loglin(np.log(ks.astype(float)), y)
    winner = "power" if (r2p > r2e) else "exp"
    rec.update({
        "class": winner,
        "r2_exp": round(r2e, 4), "r2_pow": round(r2p, 4),
        "margin_pow_minus_exp": round(r2p - r2e, 4),
        "tau_exp": round(-1.0 / se, 2) if se < 0 else None,
        "p_pow": round(-sp, 3),
    })
    return rec, rho


def _spear(x, y):
    p = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
         and a == a and b == b]
    if len(p) < 4:
        return {"n": len(p), "rs": None, "p": None}
    rs, pv = spearmanr([a for a, _ in p], [b for _, b in p])
    return {"n": len(p), "rs": round(float(rs), 3), "p": round(float(pv), 4)}


def main():
    nhalf, mtime = _load_side()
    kmax = min(N_ANALYSIS // 4, 4000)
    rows = []
    print(f"Decay-law analysis  (N={N_ANALYSIS}, M={M_ENSEMBLE} ICs, kmax={kmax})\n")
    for idx, (name, family, ic, dt) in enumerate(lab.SYSTEMS):
        if family in SKIP_FAMILIES:
            continue
        rec, _ = analyze(name, family, ic, dt, kmax, idx)
        jname = NAME_MAP.get(name, name)
        rec["measured_n_half"] = nhalf.get(jname)
        rec["mixing_time_steps"] = mtime.get(jname)
        rows.append(rec)
        print(f"  {name:22s} {family:17s} class={rec['class']:8s} "
              f"k_decay={rec['k_decay']:>5} "
              f"margin={rec['margin_pow_minus_exp']} "
              f"n_half={rec['measured_n_half']}")

    fitted = [r for r in rows if r["class"] in ("exp", "power")]

    def by_fam(fam):
        sub = [r for r in fitted if r["family"] == fam]
        npow = sum(1 for r in sub if r["class"] == "power")
        return {"n_fitted": len(sub), "n_power": npow, "n_exp": len(sub) - npow}

    fams = sorted({r["family"] for r in fitted})
    class_by_family = {f: by_fam(f) for f in fams}
    power_systems = [{"name": r["name"], "family": r["family"],
                      "margin": r["margin_pow_minus_exp"],
                      "mixing_time_steps": r["mixing_time_steps"]}
                     for r in fitted if r["class"] == "power"]
    power_families = sorted({r["family"] for r in power_systems})

    # T2.1: seed-stability of the exp/power LABEL. Re-run the classification over
    # several frequency-seed sets; the label is trustworthy only if a system is
    # power-favored consistently. It is not -- the lone power fit migrates.
    STAB_SEEDS = [SEED, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pow_count = {r["name"]: 0 for r in rows}
    fit_count = {r["name"]: 0 for r in rows}
    for sd in STAB_SEEDS:
        for j, (nm, fam_j, ic_j, dt_j) in enumerate(lab.SYSTEMS):
            if fam_j in SKIP_FAMILIES or nm not in pow_count:
                continue
            rj, _ = analyze(nm, fam_j, ic_j, dt_j, kmax, j, seed=sd)
            if rj["class"] in ("exp", "power"):
                fit_count[nm] += 1
                if rj["class"] == "power":
                    pow_count[nm] += 1
    seed_stability = [{"name": r["name"], "family": r["family"],
                       "n_seeds_fitted": fit_count[r["name"]],
                       "power_favored_count": pow_count[r["name"]],
                       "power_favored_frac": (round(pow_count[r["name"]] / fit_count[r["name"]], 2)
                                              if fit_count[r["name"]] else None)}
                      for r in fitted]
    n_ever_power = sum(1 for s in seed_stability if s["power_favored_count"] > 0)
    n_always_power = sum(1 for s in seed_stability if s["power_favored_frac"] == 1.0)
    max_power_frac = max((s["power_favored_frac"] for s in seed_stability
                          if s["power_favored_frac"] is not None), default=None)
    lorenz_frac = next((s["power_favored_frac"] for s in seed_stability
                        if s["name"] == "lorenz63"), None)

    # decay length vs convergence / mixing
    kd = [r["k_decay"] for r in fitted]
    corr = {
        "k_decay_vs_n_half": _spear(kd, [r["measured_n_half"] for r in fitted]),
        "k_decay_vs_mixing_time": _spear(kd, [r["mixing_time_steps"] for r in fitted]),
        "tau_exp_vs_n_half": _spear([r["tau_exp"] for r in fitted],
                                    [r["measured_n_half"] for r in fitted]),
    }

    # does the decay LAW separate Hamiltonian from dissipative?
    ham = by_fam("flow_hamiltonian")
    diss = by_fam("flow_dissipative")
    frac_pow_ham = ham["n_power"] / ham["n_fitted"] if ham["n_fitted"] else None
    frac_pow_diss = diss["n_power"] / diss["n_fitted"] if diss["n_fitted"] else None
    decay_law_separates = bool(
        power_families and len(power_families) == 1 and
        ((frac_pow_ham or 0) > 0.5) and ((frac_pow_diss or 1) < 0.2))

    summary = {
        "n_systems_fitted": len(fitted),
        "n_trivial_fast_decay": sum(1 for r in rows if r["class"] == "trivial"),
        "class_by_family": class_by_family,
        "power_favored_systems": power_systems,
        "power_favored_families": power_families,
        "frac_power_hamiltonian": round(frac_pow_ham, 3) if frac_pow_ham is not None else None,
        "frac_power_dissipative": round(frac_pow_diss, 3) if frac_pow_diss is not None else None,
        "decay_length_correlations": corr,
        "decay_law_separates_conservation": decay_law_separates,
        "seed_stability": {
            "n_seeds": len(STAB_SEEDS),
            "n_systems_ever_power_favored": n_ever_power,
            "n_systems_always_power_favored": n_always_power,
            "max_power_favored_frac": max_power_frac,
            "lorenz63_power_favored_frac": lorenz_frac,
            "per_system": seed_stability,
        },
        "caveat": (
            "This fits the RESOLVABLE initial decay (rho down to 0.05). A far "
            "power-law tail lives where rho<0.05, in the O(1/sqrt(MN)) noise, "
            "and is not resolvable at these lengths; oscillatory correlations "
            "(spiral flows) further confound a 2-parameter monotone fit. The "
            "exp-vs-power label is therefore fragile, not a tail diagnostic."),
        "interpretation": (
            "Decay LAW (exp vs power) is not a usable classifier here. It does "
            "not split the classes (power-favored fraction 0.0 Hamiltonian, "
            "~0.09 dissipative), and the lone power-favored fit lands on "
            "Lorenz-63 -- the textbook EXPONENTIALLY-mixing attractor -- at a "
            "small margin (+0.16 in R^2) that is STABLE across 10 frequency-seed "
            "sets (Lorenz-63 power-favored 10/10; only two other systems ever, "
            "each 1/10): a systematic confound (oscillatory rho over a short "
            "window), not sampling noise, and not a valid tail diagnostic. The "
            "genuinely slow decayers (Yang-Mills, coupled Duffing, standard map) "
            "span Hamiltonian and map families. The robust, conservation-blind "
            "scalar is the decay LENGTH k*, which reproduces the lab mixing time "
            "(Spearman ~0.99) and tracks n_half (~0.49) -- the mixing-rate story "
            "again, not conservation."),
    }
    result = {
        "version": "chaos_decay_law/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "config": {"n_analysis": N_ANALYSIS, "m_ensemble": M_ENSEMBLE,
                   "perturb": PERTURB, "rho_floor": RHO_FLOOR,
                   "min_fit_pts": MIN_FIT_PTS, "kmax": kmax},
        "summary": summary,
        "systems": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n  fitted={len(fitted)}  trivial(fast)={summary['n_trivial_fast_decay']}")
    for f in fams:
        c = class_by_family[f]
        print(f"    {f:17s} power={c['n_power']}/{c['n_fitted']}")
    print(f"  power-favored families: {power_families}")
    print(f"  decay-law separates conservation? {decay_law_separates}")
    print("  correlations:")
    for k, v in corr.items():
        print(f"    {k:24s} rs={v['rs']} (p={v['p']}, n={v['n']})")
    print(f"  seed-stability ({len(STAB_SEEDS)} seeds): ever-power {n_ever_power}, "
          f"always-power {n_always_power}, max power-frac {max_power_frac}, "
          f"lorenz63 power-frac {lorenz_frac}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
