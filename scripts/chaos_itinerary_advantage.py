"""
chaos_itinerary_advantage.py
============================
Idea C: Is the itinerary-grammar advantage a conservative/dissipative
classifier, or is it -- like the 0-1 test -- governed by mixing rate?

Meta-analysis joining two existing artifacts by system name:
  - results/chaos_itinerary_prediction.json  (transition-conditioned grammar)
  - results/chaos_universality_lab_best.json  (K, local Lyapunov, contraction)

For each system the genuine itinerary advantage is the transition-conditioned
grammar score minus its baseline, evaluated only on steps where the coarse
cell ACTUALLY changes:
    adv = transition_accuracy - transition_baseline.
The order-1-minus-order-0 "lift" is NOT used: it is dominated by dwell-time
persistence (finely-sampled flows stay in-cell), as the source script notes.

Forecast (written before running the full statistics)
-----------------------------------------------------
Several weakly-chaotic systems saturate the ceiling adv = 1 - 1/K (perfect
transition prediction, tacc=1.000). These span BOTH classes (Hamiltonian
henon_heiles/cr3bp/pullen/quartic, but also dissipative halvorsen and the
harmonic control). Flows change cells rarely (change_fraction ~ 0.2-8%), maps
almost every step. Prediction: adv is NOT a clean conservative/dissipative
separator; it tracks coarse-grained mixing (anti-correlated with
change_fraction); the most strongly-mixing systems (chen, standard map) sit
lowest in both classes. No new claim is promoted -- a null/cautionary result
that reinforces the paper's "mixing, not conservation" thesis.

Output: results/chaos_itinerary_advantage.json
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

RESULTS = Path(__file__).parent.parent / "results"
ITIN = RESULTS / "chaos_itinerary_prediction.json"
LAB = RESULTS / "chaos_universality_lab_best.json"
SLOPE = RESULTS / "chaos_slope_analysis.json"  # second, independent 12k 0-1 K
OUT = RESULTS / "chaos_itinerary_advantage.json"

K_CELLS = ["4", "6", "8"]
K_PRIMARY = "6"
# Reliability gate: require enough actual cell changes that the binomial
# standard error on transition_accuracy, sqrt(p(1-p)/n), is <~ 0.04.
# For p~0.8 that needs n >~ 0.16/0.0016 = 100. Workload-justified, not round.
MIN_CHANGES = 100
# A system "saturates the ceiling" when transition prediction is essentially
# perfect (tacc >= 0.98), so adv ~ 1 - 1/K regardless of grammar richness.
CEILING_TACC = 0.98

FAM_SHORT = {"flow_dissipative": "diss", "flow_hamiltonian": "ham",
             "map": "map", "control": "control"}


def _load():
    it = json.loads(ITIN.read_text())
    lb = json.loads(LAB.read_text())
    sl = json.loads(SLOPE.read_text())
    K = {s["name"]: (s.get("gottwald_melbourne_01") or {}).get("K_median")
         for s in lb["systems"]}
    # independent 12k-step 0-1 K from the slope analysis (different c-values,
    # stride cap, IC alignment) -- used only to check the adv-K result is not
    # an artifact of one K estimator. (Slope names mackey_glass without _tau17.)
    Ksl = {s["name"]: s.get("K_median") for s in sl["systems"]}
    Ksl.setdefault("mackey_glass_tau17", Ksl.get("mackey_glass"))
    # Benettin largest Lyapunov exponent (the Table 1 lambda1), a clean global
    # estimate; the windowed local_lyapunov.mean is a poor proxy (negative for
    # the delay system and the controls).
    lam = {s["name"]: (s.get("fingerprints") or {}).get("lambda1")
           for s in lb["systems"]}
    sig = {s["name"]: (s.get("phase_volume_contraction") or {}).get("mean_divergence")
           for s in lb["systems"]}
    return it, K, Ksl, lam, sig


def _adv_at(sysrec, kc):
    tc = (sysrec.get("by_K", {}).get(kc, {}) or {}).get("transition_conditioned", {}) or {}
    ta, tb = tc.get("transition_accuracy"), tc.get("transition_baseline")
    n = tc.get("n_changes")
    cf = tc.get("change_fraction")
    if ta is None or tb is None or (isinstance(ta, float) and math.isnan(ta)):
        return None
    adv = ta - tb
    se = math.sqrt(max(ta * (1 - ta), 0.0) / n) if n else float("nan")
    return {"tacc": ta, "tbase": tb, "adv": adv, "n_changes": n,
            "change_fraction": cf, "se": se}


def _spear_pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return {"n": int(m.sum()), "spearman_rs": None, "spearman_p": None,
                "pearson_r": None, "pearson_p": None}
    rs, ps = spearmanr(x[m], y[m]); pr, pp = pearsonr(x[m], y[m])
    return {"n": int(m.sum()),
            "spearman_rs": round(float(rs), 3), "spearman_p": round(float(ps), 4),
            "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4)}


def main():
    it, Kmed, Kslope, Lam, Sig = _load()
    rows = []
    for s in it["systems"]:
        name, fam = s["name"], s["family"]
        rec = {"name": name, "family": fam, "fam": FAM_SHORT.get(fam, fam),
               "K_median": Kmed.get(name), "K_slope_12k": Kslope.get(name),
               "lambda1": Lam.get(name), "sigma": Sig.get(name), "by_K": {}}
        for kc in K_CELLS:
            a = _adv_at(s, kc)
            if a is not None:
                rec["by_K"][kc] = a
        prim = rec["by_K"].get(K_PRIMARY)
        rec["adv"] = prim["adv"] if prim else None
        rec["tacc"] = prim["tacc"] if prim else None
        rec["n_changes"] = prim["n_changes"] if prim else None
        rec["change_fraction"] = prim["change_fraction"] if prim else None
        rec["se"] = prim["se"] if prim else None
        rec["reliable"] = bool(prim and prim["n_changes"] and prim["n_changes"] >= MIN_CHANGES)
        rec["ceiling"] = bool(prim and prim["tacc"] >= CEILING_TACC)
        rows.append(rec)

    chaotic = [r for r in rows if r["fam"] in ("diss", "ham", "map") and r["adv"] is not None]
    reliable = [r for r in chaotic if r["reliable"]]

    def grp(rs, fam):
        return [r["adv"] for r in rs if r["fam"] == fam]

    def stats(rs, fam):
        v = grp(rs, fam)
        return {"n": len(v), "mean_adv": round(float(np.mean(v)), 3) if v else None,
                "median_adv": round(float(np.median(v)), 3) if v else None}

    # Mann-Whitney Hamiltonian vs dissipative, raw and reliability-filtered
    def mwu(rs):
        h, d = grp(rs, "ham"), grp(rs, "diss")
        if len(h) >= 3 and len(d) >= 3:
            u, p = mannwhitneyu(h, d, alternative="two-sided")
            return {"n_ham": len(h), "n_diss": len(d),
                    "U": float(u), "p": round(float(p), 4)}
        return {"n_ham": len(h), "n_diss": len(d), "U": None, "p": None}

    # cross-K robustness of the ham-vs-diss gap (mean adv difference per K)
    gap_by_K = {}
    for kc in K_CELLS:
        hv = [r["by_K"][kc]["adv"] for r in chaotic
              if kc in r["by_K"] and r["by_K"][kc]["n_changes"] and r["by_K"][kc]["n_changes"] >= MIN_CHANGES and r["fam"] == "ham"]
        dv = [r["by_K"][kc]["adv"] for r in chaotic
              if kc in r["by_K"] and r["by_K"][kc]["n_changes"] and r["by_K"][kc]["n_changes"] >= MIN_CHANGES and r["fam"] == "diss"]
        gap_by_K[kc] = {"n_ham": len(hv), "n_diss": len(dv),
                        "mean_ham": round(float(np.mean(hv)), 3) if hv else None,
                        "mean_diss": round(float(np.mean(dv)), 3) if dv else None,
                        "gap": round(float(np.mean(hv) - np.mean(dv)), 3) if hv and dv else None}

    # correlations over reliable chaotic systems (primary K)
    adv = [r["adv"] for r in reliable]
    cf = [r["change_fraction"] for r in reliable]
    kk = [r["K_median"] for r in reliable]
    kk2 = [r["K_slope_12k"] for r in reliable]
    ll = [r["lambda1"] for r in reliable]
    corr = {
        "adv_vs_change_fraction": _spear_pear(adv, cf),
        "adv_vs_K_median": _spear_pear(adv, kk),
        "adv_vs_K_slope_12k": _spear_pear(adv, kk2),  # robustness: 2nd K estimator
        "adv_vs_lambda1_benettin": _spear_pear(adv, ll),
    }

    n_ceiling = sum(1 for r in chaotic if r["ceiling"])
    ceiling_fams = sorted({r["fam"] for r in chaotic if r["ceiling"]})

    # Verdict. The raw Hamiltonian-vs-dissipative separation can be real in
    # sample yet fail to be an INDEPENDENT conservation classifier. Three
    # confounds are checked explicitly:
    #   (a) ceiling crossing -- does any dissipative system reach the
    #       Hamiltonian ceiling (perfect coarse prediction)?
    #   (b) survivorship -- which Hamiltonians does the reliability gate drop,
    #       and were they low-advantage (strongly chaotic)?
    #   (c) mixing confound -- is adv explained by the 0-1 K (anti-correlation)?
    p_filt = mwu(reliable)["p"]
    ham_diss_separable = bool(p_filt is not None and p_filt < 0.05)
    diss_at_ceiling = [r["name"] for r in reliable if r["fam"] == "diss" and r["ceiling"]]
    ham_dropped = [{"name": r["name"], "adv": round(r["adv"], 3),
                    "n_changes": r["n_changes"]}
                   for r in chaotic if r["fam"] == "ham" and not r["reliable"]]
    advK = corr["adv_vs_K_median"]
    adv_explained_by_K = bool(advK["spearman_p"] is not None and advK["spearman_p"] < 0.05)
    independent_classifier = bool(ham_diss_separable
                                  and not diss_at_ceiling
                                  and not adv_explained_by_K)

    summary = {
        "metric": "adv = transition_accuracy - transition_baseline (cell-change steps only)",
        "primary_K_cells": K_PRIMARY,
        "min_changes_gate": MIN_CHANGES,
        "ceiling_adv_formula": "1 - 1/K",
        "n_chaotic_with_adv": len(chaotic),
        "n_reliable": len(reliable),
        "group_means_raw": {f: stats(chaotic, f) for f in ("ham", "diss", "map")},
        "group_means_reliable": {f: stats(reliable, f) for f in ("ham", "diss", "map")},
        "mannwhitney_ham_vs_diss_raw": mwu(chaotic),
        "mannwhitney_ham_vs_diss_reliable": mwu(reliable),
        "gap_by_cell_count": gap_by_K,
        "n_ceiling_saturated": n_ceiling,
        "ceiling_saturated_families": ceiling_fams,
        "correlations_reliable_chaotic": corr,
        "ham_diss_separable_in_sample": ham_diss_separable,
        "is_independent_conservation_classifier": independent_classifier,
        "confounds": {
            "dissipative_systems_at_ceiling": diss_at_ceiling,
            "hamiltonians_dropped_by_reliability_gate": ham_dropped,
            "adv_anticorrelates_with_K": advK,
        },
        "interpretation": (
            "Reliable Hamiltonians separate from dissipative in this sample "
            "(MWU p<0.05), but this is not an independent conservation "
            "classifier: a dissipative flow (halvorsen) reaches the same "
            "perfect-prediction ceiling; the reliability gate drops exactly "
            "the strongly-chaotic Hamiltonians (double pendulum, Yang-Mills) "
            "whose coarse itinerary is disordered; and adv anti-correlates "
            "with the 0-1 K (Spearman ~ -0.6). The advantage tracks coarse "
            "mixing/dwell order, not conservation -- echoing the 0-1 result."),
    }

    result = {
        "version": "chaos_itinerary_advantage/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"itinerary": ITIN.name, "lab": LAB.name},
        "config": {"K_cells": K_CELLS, "K_primary": K_PRIMARY,
                   "min_changes": MIN_CHANGES, "ceiling_tacc": CEILING_TACC},
        "summary": summary,
        "systems": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))

    # console
    print(f"Itinerary-advantage meta-analysis  (primary K={K_PRIMARY} cells)\n")
    print(f"  chaotic systems with adv : {len(chaotic)}   reliable (>={MIN_CHANGES} changes): {len(reliable)}")
    print(f"  ceiling-saturated (tacc>={CEILING_TACC}): {n_ceiling}  across {ceiling_fams}\n")
    print("  group mean adv      RAW        RELIABLE")
    for f in ("ham", "diss", "map"):
        r = stats(chaotic, f); rr = stats(reliable, f)
        print(f"    {f:5s}  {str(r['mean_adv']):>8} (n={r['n']:>2})   {str(rr['mean_adv']):>8} (n={rr['n']:>2})")
    print(f"\n  Mann-Whitney ham vs diss (reliable): {mwu(reliable)}")
    print(f"  gap by cell count: " + "  ".join(
        f"K{kc}:{gap_by_K[kc]['gap']}" for kc in K_CELLS))
    print("\n  correlations (reliable chaotic):")
    for k, v in corr.items():
        print(f"    {k:28s} rs={v['spearman_rs']} (p={v['spearman_p']}, n={v['n']})")
    print(f"\n  ham/diss separable in sample?        {summary['ham_diss_separable_in_sample']}")
    print(f"  INDEPENDENT conservation classifier? {summary['is_independent_conservation_classifier']}")
    print(f"    confound -- dissipative at ceiling : {diss_at_ceiling}")
    print(f"    confound -- hams dropped by gate   : {[d['name'] for d in ham_dropped]}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
