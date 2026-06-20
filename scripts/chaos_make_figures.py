"""Generate publication figures for the chaos universality manuscript.

Reads the result JSONs and writes PDF+PNG figures to manuscripts/figures/chaos/.
Run after the analysis scripts have produced their artifacts:

    scientist-env/bin/python3 scripts/liquid/chaos_make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # dist/chaos-universality-paper
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

HAM = "#7c3aed"   # purple
DISS = "#0891b2"  # cyan
MAP = "#16a34a"   # green


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ── Fig 1: Σλ discriminator ──────────────────────────────────────────────────
def fig_sigma_lambda():
    lab = _load("chaos_universality_lab.json")
    rows = []
    for s in lab.get("systems", []):
        pvc = s.get("phase_volume_contraction", {})
        if not pvc.get("available"):
            continue
        rows.append((s["name"], s["family"], pvc["mean_divergence"]))
    if not rows:
        return
    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    divs = [r[2] for r in rows]
    cols = [HAM if r[1] == "flow_hamiltonian" else DISS for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(rows))
    ax.barh(y, divs, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8, family="monospace")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(r"phase-space volume contraction  $\Sigma\lambda = \langle \nabla\!\cdot f\rangle$")
    ax.set_title("The discriminator: $\\Sigma\\lambda = 0$ (Hamiltonian) vs $<0$ (dissipative)")
    # symlog so -13.7 and 0 both readable
    ax.set_xscale("symlog", linthresh=0.05)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HAM, label="Hamiltonian (conservative)"),
                       Patch(color=DISS, label="dissipative")],
              loc="lower left", fontsize=9)
    _save(fig, "fig1_sigma_lambda_discriminator")


# ── Fig 2: K vs trajectory length (the artifact) + Green-Kubo overlay ────────
def fig_K_vs_length():
    gk = _load("chaos_green_kubo.json")
    if not gk:
        return
    picks = {"double_pendulum": HAM, "coupled_duffing_ham": HAM,
             "lorenz63": DISS, "chen": DISS, "yang_mills_x2y2": HAM}
    by_name = {s["name"]: s for s in gk.get("systems", [])}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, col in picks.items():
        s = by_name.get(name)
        if not s or "per_length" not in s:
            continue
        Ns = sorted(int(k) for k in s["per_length"])
        meas = [s["per_length"][str(N)]["measured_K"] for N in Ns]
        pred = [s["per_length"][str(N)]["predicted_K"] for N in Ns]
        ls = "-" if col == DISS else "--"
        ax.plot(Ns, meas, ls, color=col, marker="o", ms=4,
                label=f"{name} (measured)")
        ax.plot(Ns, pred, ":", color=col, marker="x", ms=5, alpha=0.7)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("trajectory length (steps)")
    ax.set_ylabel("Gottwald–Melbourne $K$")
    ax.set_title("0-1 $K$ is length-dependent for Hamiltonian flows\n"
                 "(dotted = closed-form Green–Kubo prediction)")
    ax.legend(fontsize=7.5, loc="lower right")
    _save(fig, "fig2_K_vs_length")


# ── Fig 3: predicted vs measured K scatter ───────────────────────────────────
def fig_green_kubo_scatter():
    gk = _load("chaos_green_kubo.json")
    if not gk:
        return
    pred, meas, cols = [], [], []
    fam_col = {"flow_hamiltonian": HAM, "flow_dissipative": DISS, "map": MAP}
    for s in gk.get("systems", []):
        for v in s.get("per_length", {}).values():
            if v["predicted_K"] == v["predicted_K"] and v["measured_K"] == v["measured_K"]:
                pred.append(v["predicted_K"]); meas.append(v["measured_K"])
                cols.append(fam_col.get(s["family"], "gray"))
    if not pred:
        return
    r = gk["summary"]["predicted_vs_measured_K_pearson"]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(meas, pred, c=cols, s=22, alpha=0.8, edgecolor="none")
    lims = [min(min(pred), min(meas)) - 0.05, max(max(pred), max(meas)) + 0.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("measured $K$ (run the 0-1 test)")
    ax.set_ylabel("predicted $K$ (closed form from $C(k)$)")
    ax.set_title(f"Green–Kubo closed form predicts $K$\nPearson $r={r:.3f}$")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HAM, label="Hamiltonian"),
                       Patch(color=DISS, label="dissipative"),
                       Patch(color=MAP, label="map")], fontsize=8, loc="upper left")
    _save(fig, "fig3_green_kubo_scatter")


# ── Fig 4: mixing time τ vs crossover n_½, and Σλ-vs-parameter inset ─────────
def fig_mixing_and_sweep():
    mt = _load("chaos_mixing_time_analysis.json")
    ps = _load("chaos_parameter_sweep.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    if mt:
        fam_col = {"flow_hamiltonian": HAM, "flow_dissipative": DISS, "map": MAP}
        for s in mt.get("systems", []):
            tau = s.get("mixing_time_steps")
            nh = s.get("k_crossover_length_n_half")
            if tau is None or tau != tau:
                continue
            nh = nh if nh is not None else 64000  # censored marker
            ax.scatter(tau, nh, c=fam_col.get(s["family"], "gray"), s=28, alpha=0.8)
        rho = mt["summary"].get("spearman_tau_vs_n_half")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"mixing time $\tau$ (steps)")
        ax.set_ylabel(r"$K\!\to\!1$ crossover $n_{1/2}$ (steps)")
        ax.set_title(f"Mixing-time law  (Spearman $\\rho={rho:.2f}$)")

    ax = axes[1]
    if ps:
        lor = next((s for s in ps["sweeps"] if s["system"] == "lorenz63"), None)
        if lor:
            rhos = [r["param"] for r in lor["sweep"]]
            divs = [r["sigma_lambda_divergence"] for r in lor["sweep"]]
            lams = [r["lambda1"] for r in lor["sweep"]]
            ax.plot(rhos, divs, "o-", color=DISS, label=r"$\Sigma\lambda$ (pinned)")
            ax.axhline(lor["analytic_divergence_constant"], color="black", ls=":",
                       lw=1, label=r"analytic $-\sigma-1-\beta$")
            ax2 = ax.twinx()
            ax2.plot(rhos, lams, "s--", color=HAM, label=r"$\lambda_1$ (varies 30×)")
            ax2.set_ylabel(r"$\lambda_1$", color=HAM)
            ax.set_xlabel(r"Lorenz $\rho$")
            ax.set_ylabel(r"$\Sigma\lambda$", color=DISS)
            ax.set_title(r"$\Sigma\lambda$ is chaos-strength-independent")
            ax.legend(loc="center left", fontsize=8)
            ax2.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_mixing_time_and_parameter_sweep")


# ── Fig 5: itinerary transition-conditioned accuracy vs baseline ─────────────
def fig_itinerary():
    it = _load("chaos_itinerary_prediction.json")
    if not it:
        return
    rows = []
    for s in it.get("systems", []):
        ta = s.get("rep_transition_acc"); tb = s.get("rep_transition_baseline")
        if ta is None or tb is None:
            continue
        rows.append((s["name"], ta, tb))
    if not rows:
        return
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    names = [r[0] for r in rows]
    ta = [r[1] for r in rows]; tb = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh(y + 0.2, ta, height=0.4, color="#0891b2", label="transition grammar")
    ax.barh(y - 0.2, tb, height=0.4, color="#cbd5e1", label="baseline")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5, family="monospace")
    ax.set_xlabel("next-cell accuracy when the cell changes (held-out)")
    ax.set_title("Predict the climate: itinerary transition grammar vs baseline")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    _save(fig, "fig5_itinerary_prediction")


def main():
    print("Generating chaos manuscript figures →", FIGDIR)
    fig_sigma_lambda()
    fig_K_vs_length()
    fig_green_kubo_scatter()
    fig_mixing_and_sweep()
    fig_itinerary()
    print("done.")


if __name__ == "__main__":
    main()
