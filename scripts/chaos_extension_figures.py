"""
chaos_extension_figures.py
=========================
Figures for the A–F diagnostic battery (manuscript sections 3.5–3.9), matching
the style of chaos_make_figures.py. Each reads the corresponding extension JSON
from results/ and writes PDF+PNG to BOTH the main manuscript figure directory
(manuscripts/figures/chaos/, where the .tex \\graphicspath points) and the
self-contained dist package figures/ directory.

    fig6  alpha--K decoupling + c-sampling fragility   (sec 3.5)
    fig7  observable selection: Rossler K vs length    (sec 3.6)
    fig8  itinerary advantage vs mixing proxies        (sec 3.7)
    fig9  decay law (exp vs power) + decay length       (sec 3.8)
    fig10 order-k route-grammar memory gain            (sec 3.9)

Run: scientist-env/bin/python3 scripts/chaos_extension_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PKG = Path(__file__).resolve().parent.parent       # dist/chaos-universality-paper
RESULTS = PKG / "results"
FIGDIR = PKG / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

HAM = "#7c3aed"; DISS = "#0891b2"; MAP = "#16a34a"; CTRL = "#9ca3af"; DELAY = "#d97706"
FAMC = {"flow_hamiltonian": HAM, "flow_dissipative": DISS, "map": MAP,
        "control": CTRL, "flow_delay": DELAY}
FAMC_SHORT = {"ham": HAM, "diss": DISS, "map": MAP, "control": CTRL, "delay": DELAY}


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig, stem):
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def _legend(ax, fams, **kw):
    ax.legend(handles=[Patch(color=FAMC[f], label=l) for f, l in fams], **kw)


# ── Fig 6: alpha–K decoupling and the c-sampling fragility (sec 3.5) ──────────
def fig_alpha_K():
    sl = _load("chaos_slope_analysis.json")
    if not sl:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    label_pts = {"cr3bp_earth_moon": "CR3BP", "quartic_coupled": "quartic",
                 "spring_pendulum": "spring", "lorenz63": "Lorenz 63",
                 "henon_heiles": "Hénon–Heiles"}
    seen = set()
    for s in sl.get("systems", []):
        a = s.get("alpha_full"); km = s.get("K_seed_median")
        lo = s.get("K_seed_lo"); hi = s.get("K_seed_hi")
        if a is None or km is None:
            continue
        c = FAMC.get(s["family"], "gray")
        yerr = [[km - lo], [hi - km]] if lo is not None and hi is not None else None
        ax.errorbar(a, km, yerr=yerr, fmt="o", ms=6, color=c, ecolor=c,
                    elinewidth=1, capsize=2, alpha=0.85, zorder=3)
        seen.add(s["family"])
        if s["name"] in label_pts:
            ax.annotate(label_pts[s["name"]], (a, km), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
    # decoupling guide box: alpha ~ 1 yet K ~ 0
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvspan(0.7, 1.15, color="#fde68a", alpha=0.25, zorder=0)
    ax.text(1.11, 0.46, "diffusive\ngrowth\n($\\alpha\\!\\approx\\!1$)",
            fontsize=8, ha="center", va="center", color="#92400e", zorder=1)
    ax.set_xlabel(r"growth exponent  $\alpha = \mathrm{d}\log M_c/\mathrm{d}\log n$")
    ax.set_ylabel(r"0–1 $K$ (median over 12 frequency seeds; bars span the seeds)")
    ax.set_title("$\\alpha$–$K$ decoupling and the c-sampling fragility of $K$")
    fams = [(f, {"flow_hamiltonian": "Hamiltonian", "flow_dissipative": "dissipative",
                 "map": "map", "control": "control", "flow_delay": "delay"}[f])
            for f in ["flow_hamiltonian", "flow_dissipative", "map", "control", "flow_delay"]
            if f in seen]
    _legend(ax, fams, loc="lower right", fontsize=8)
    _save(fig, "fig6_alpha_K_decoupling")


# ── Fig 7: observable selection — Rossler K vs length per coordinate (sec 3.6) ─
def fig_observable():
    ob = _load("chaos_observable_selection.json")
    if not ob:
        return
    ross = next((s for s in ob.get("systems", []) if s["name"] == "rossler"), None)
    if not ross:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    coord_col = {"x": "#64748b", "y": DISS, "z": HAM}
    for c in ross["coords"]:
        ns = [pt["n"] for pt in c["K_curve"]]
        ks = [pt["K"] for pt in c["K_curve"]]
        nm = c["name"]; tau = c["tau_int"]
        ax.plot(ns, ks, "o-", color=coord_col.get(nm, "gray"), ms=5,
                label=f"{nm}  ($\\tau_{{\\mathrm{{int}}}}={tau:.1f}$)")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("trajectory length $N$ (steps)")
    ax.set_ylabel("0–1 $K$")
    ax.set_title("Observable choice sets convergence speed (Rössler)\n"
                 "$z$ ($\\tau_{\\mathrm{int}}{=}7.5$) crosses $K{=}0.5$ at $N{=}1$k; "
                 "$y$ not until $N{=}8$k")
    ax.legend(title="coordinate", fontsize=9, loc="lower right")
    _save(fig, "fig7_observable_selection")


# ── Fig 8: itinerary advantage vs mixing proxies (sec 3.7) ───────────────────
def fig_itinerary_advantage():
    ad = _load("chaos_itinerary_advantage.json")
    if not ad:
        return
    rows = [s for s in ad.get("systems", [])
            if s.get("adv") is not None and s.get("reliable")
            and s["fam"] in ("ham", "diss", "map")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    note = {"halvorsen": "Halvorsen", "chen": "Chen", "cr3bp_earth_moon": "CR3BP",
            "standard_map_K1p2": "std map"}
    for ax, key, xlab in ((axes[0], "K_median", "0–1 $K_{\\mathrm{median}}$ (12k)"),
                          (axes[1], "lambda1", "Benettin $\\lambda_1$")):
        xs, ys, cs = [], [], []
        for s in rows:
            x = s.get(key)
            if x is None or x != x:
                continue
            xs.append(x); ys.append(s["adv"]); cs.append(FAMC_SHORT.get(s["fam"], "gray"))
            if s["name"] in note:
                ax.annotate(note[s["name"]], (x, s["adv"]), fontsize=7.5,
                            xytext=(4, -2), textcoords="offset points")
        ax.scatter(xs, ys, c=cs, s=34, alpha=0.85, edgecolor="none", zorder=3)
        # robust Spearman from the artifact
        corr = ad["summary"]["correlations_reliable_chaotic"]
        rk = corr["adv_vs_K_median"] if key == "K_median" else corr["adv_vs_lambda1_benettin"]
        ax.set_xlabel(xlab); ax.set_ylabel("transition-grammar advantage")
        ax.set_title(f"Spearman $r_s={rk['spearman_rs']}$ ($p={rk['spearman_p']}$)")
    axes[0].legend(handles=[Patch(color=HAM, label="Hamiltonian"),
                            Patch(color=DISS, label="dissipative"),
                            Patch(color=MAP, label="map")], fontsize=8, loc="lower left")
    fig.suptitle("Itinerary advantage falls as the dynamics mix faster "
                 "(higher $K$, higher $\\lambda_1$) — not a conservation split", y=1.02)
    fig.tight_layout()
    _save(fig, "fig8_itinerary_advantage")


# ── Fig 9: decay law (exp vs power) + decay length (sec 3.8) ──────────────────
def fig_decay_law():
    dl = _load("chaos_decay_law.json")
    if not dl:
        return
    fitted = [s for s in dl.get("systems", []) if s.get("class") in ("exp", "power")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    ax = axes[0]
    for s in fitted:
        c = FAMC.get(s["family"], "gray")
        ax.scatter(s["r2_exp"], s["r2_pow"], c=c, s=34, alpha=0.85, edgecolor="none", zorder=3)
        if s["name"] == "lorenz63":
            ax.annotate("Lorenz 63\n(power-favored: artifact)", (s["r2_exp"], s["r2_pow"]),
                        fontsize=7.5, xytext=(-10, -34), textcoords="offset points",
                        ha="center", arrowprops=dict(arrowstyle="->", lw=0.7))
    lims = [0.0, 1.02]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.fill_between(lims, lims, [lims[0], lims[0]], color=DISS, alpha=0.06)
    ax.text(0.55, 0.18, "exponential\nfits better", fontsize=8, color=DISS, ha="center")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("$R^2$ exponential fit"); ax.set_ylabel("$R^2$ power-law fit")
    ax.set_title("Decay law: exponential wins for 18/19 flows")

    ax = axes[1]
    for s in fitted:
        nh = s.get("measured_n_half")
        if nh is None:
            continue
        ax.scatter(s["k_decay"], nh, c=FAMC.get(s["family"], "gray"),
                   s=34, alpha=0.85, edgecolor="none", zorder=3)
    corr = dl["summary"]["decay_length_correlations"]["k_decay_vs_n_half"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"decay length $k^\ast$ (lags to $\rho=0.05$)")
    ax.set_ylabel(r"0–1 crossover $n_{1/2}$ (steps)")
    ax.set_title(f"Decay length tracks $n_{{1/2}}$ (Spearman $r_s={corr['rs']}$)")
    axes[0].legend(handles=[Patch(color=HAM, label="Hamiltonian"),
                            Patch(color=DISS, label="dissipative"),
                            Patch(color=MAP, label="map"),
                            Patch(color=DELAY, label="delay")], fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig9_decay_law")


# ── Fig 10: order-k route-grammar memory gain (sec 3.9) ──────────────────────
def fig_markov_order():
    mk = _load("chaos_markov_order.json")
    if not mk:
        return
    rows = [s for s in mk.get("systems", []) if s.get("reliable")
            and s.get("gain_o2_minus_o1") is not None]
    rows.sort(key=lambda s: s["gain_o2_minus_o1"])
    names = [s["name"] for s in rows]
    gains = [s["gain_o2_minus_o1"] for s in rows]
    cols = [FAMC_SHORT.get(s["fam"], "gray") for s in rows]
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    y = np.arange(len(rows))
    ax.barh(y, gains, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5, family="monospace")
    ax.axvline(0, color="black", lw=0.8)
    mwu = mk["summary"]["mannwhitney_gain_ham_vs_diss"]
    ax.set_xlabel("order-2 minus order-1 next-route accuracy (held-out)")
    ax.set_title("Higher-order memory gain is heterogeneous and class-blind\n"
                 f"(Hamiltonian vs dissipative Mann–Whitney $p={mwu.get('p')}$)")
    ax.legend(handles=[Patch(color=HAM, label="Hamiltonian"),
                       Patch(color=DISS, label="dissipative"),
                       Patch(color=MAP, label="map")], fontsize=8, loc="lower right")
    _save(fig, "fig10_markov_order")


def main():
    print("Generating extension figures →", FIGDIR)
    fig_alpha_K()
    fig_observable()
    fig_itinerary_advantage()
    fig_decay_law()
    fig_markov_order()
    print("done.")


if __name__ == "__main__":
    main()
