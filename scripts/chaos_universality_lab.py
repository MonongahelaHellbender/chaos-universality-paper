"""Chaos Universality Lab — cross-system structure of deterministic chaos.

Why this exists
---------------
The exponential Lyapunov wall makes long-term trajectory prediction impossible
for chaotic systems. But the *structure* of chaos — the attractor, the Koopman
spectrum, the symbolic-dynamics grammar, the surviving quasi-invariants — is
stable. This experiment characterizes that structure across many systems at
once and asks: do different chaotic systems share a low-dimensional grammar of
chaos that classifies them into universality classes?

What it does
------------
For a zoo of chaotic + control systems it computes:

  (A) Dynamical fingerprints
       λ₁  — largest Lyapunov exponent (Benettin two-trajectory method)
       D₂  — correlation dimension (Grassberger–Procaccia)
       H_PSD — spectral entropy of the power spectrum
       |λ_K|  — DMD Koopman spectral radius
       Δ_K   — Koopman spectral gap (|λ₁| − |λ₂|)
       1−|μ₂| — transfer-operator (Ulam) spectral gap, decorrelation rate
       MSE_F  — Foundation LiquidPredictor next-step MSE (relative to persistence)

  (B) Quasi-invariant hunter (AI-Poincaré style)
       Builds a monomial basis up to degree 2 over the state variables.
       Scores each candidate Q by autocorrelation persistence across long lag.
       Reports its "survival time" in Lyapunov units — how long Q stays
       essentially constant before chaos scrambles it.

  (C) Symbolic-dynamics itinerary predictor
       k-means partitions the attractor into K cells. Trajectory → symbol
       sequence. Learns first-order Markov transitions. Compares next-symbol
       top-1 accuracy of the itinerary predictor against the per-step MSE of
       the continuous predictor — the "predict the climate, not the weather"
       quantification.

  (D) Universality clustering
       Stacks fingerprints into a feature matrix, z-scores, runs PCA + complete-
       linkage agglomerative clustering. Reports which systems land together.

Output
------
    results/chaos_universality_lab.json

Run
---
    scientist-env/bin/python3 scripts/liquid/chaos_universality_lab.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import torch
    try:
        from models.liquid_core import LiquidPredictor as _LiquidPredictor
        _HAS_LIQUID = True
    except ImportError:
        _HAS_LIQUID = False
except ImportError:
    torch = None  # type: ignore
    _HAS_LIQUID = False

ROOT = Path(__file__).resolve().parent.parent  # dist/chaos-universality-paper
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "chaos_universality_lab.json"


# ── Reproducibility ──────────────────────────────────────────────────────────

GLOBAL_SEED = 20260528


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(GLOBAL_SEED + offset)


# ─────────────────────────────────────────────────────────────────────────────
# (1) The chaos system zoo
# ─────────────────────────────────────────────────────────────────────────────
#
# Each generator returns a (T, d) float32 numpy array of state trajectory
# starting from a perturbation of the canonical initial condition. We expose
# the underlying continuous vector field f(state)→dstate, dt step, and known
# conserved quantities (or None) for the quasi-invariant ground truth.

@dataclass
class SystemSpec:
    name: str
    family: str            # "map" | "flow_hamiltonian" | "flow_dissipative" | "control"
    dim: int
    integrate: Callable[[np.ndarray, int, float], np.ndarray]
    base_ic: np.ndarray
    dt: float
    known_invariants: list = field(default_factory=list)  # human-readable names
    description: str = ""


# ── Continuous-time vector fields (used for both integration and Lyapunov) ──

def _rk4_step(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_flow(f: Callable[[np.ndarray], np.ndarray],
                    x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    out = np.empty((n_steps, x0.size), dtype=np.float64)
    x = x0.astype(np.float64).copy()
    for i in range(n_steps):
        out[i] = x
        x = _rk4_step(f, x, dt)
    return out.astype(np.float32)


# Lorenz 1963
def _lorenz_f(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    return np.array([sigma * (x[1] - x[0]),
                     x[0] * (rho - x[2]) - x[1],
                     x[0] * x[1] - beta * x[2]])


def gen_lorenz(x0, n, dt):
    return _integrate_flow(_lorenz_f, x0, n, dt)


# Rössler
def _rossler_f(x, a=0.2, b=0.2, c=5.7):
    return np.array([-x[1] - x[2],
                     x[0] + a * x[1],
                     b + x[2] * (x[0] - c)])


def gen_rossler(x0, n, dt):
    return _integrate_flow(_rossler_f, x0, n, dt)


# Duffing (forced)
def _duffing_f(state, alpha=-1.0, beta=1.0, delta=0.3, gamma=0.5, omega=1.2):
    # state = [x, v, phase]
    x, v, phase = state
    dx = v
    dv = -delta * v - alpha * x - beta * x ** 3 + gamma * math.cos(phase)
    dphase = omega
    return np.array([dx, dv, dphase])


def gen_duffing(x0, n, dt):
    # x0 of shape (3,): [x, v, phase]
    return _integrate_flow(_duffing_f, x0, n, dt)


# Double pendulum in CANONICAL coordinates [θ1, θ2, p1, p2] (point masses,
# m1=m2=1, l1=l2=1). Integrating Hamilton's equations directly guarantees the
# phase-space divergence is identically zero (∂θ̇/∂θ + ∂ṗ/∂p = ∂²H/∂θ∂p −
# ∂²H/∂p∂θ = 0), so the volume-contraction cross-check reads ≈0 exactly — not
# the coordinate artifact the old velocity-coordinate form produced.
#
# H = ½ (p1² − 2 p1 p2 cosΔ + 2 p2²)/(2 − cos²Δ) − 2g cosθ1 − g cosθ2,  Δ=θ1−θ2
def _double_pendulum_f(state, g=9.81):
    th1, th2, p1, p2 = state
    d = th1 - th2
    c = math.cos(d)
    s = math.sin(d)
    D = 2.0 - c * c                      # determinant of mass matrix, ∈ [1,2]
    th1_dot = (p1 - p2 * c) / D
    th2_dot = (2.0 * p2 - p1 * c) / D
    # ∂T/∂Δ where T = ½ N/D, N = p1² − 2 p1 p2 cosΔ + 2 p2²
    N = p1 * p1 - 2.0 * p1 * p2 * c + 2.0 * p2 * p2
    dN = 2.0 * p1 * p2 * s               # dN/dΔ = 2 p1 p2 sinΔ
    dD = 2.0 * c * s                     # dD/dΔ = sin(2Δ)
    dT_dDelta = 0.5 * (dN * D - N * dD) / (D * D)
    p1_dot = -(dT_dDelta + 2.0 * g * math.sin(th1))
    p2_dot = -(-dT_dDelta + g * math.sin(th2))
    return np.array([th1_dot, th2_dot, p1_dot, p2_dot])


def gen_double_pendulum(x0, n, dt):
    return _integrate_flow(_double_pendulum_f, x0, n, dt)


# Hénon–Heiles (planar quartic; classic chaotic Hamiltonian)
def _hh_f(state):
    x, y, px, py = state
    dx = px
    dy = py
    dpx = -x - 2.0 * x * y
    dpy = -y - (x * x - y * y)
    return np.array([dx, dy, dpx, dpy])


def gen_henon_heiles(x0, n, dt):
    return _integrate_flow(_hh_f, x0, n, dt)


# ── Additional conservative (Hamiltonian) chaotic flows ─────────────────────
# All autonomous, energy-conserving, 4-D phase space [q1,q2,p1,p2] (or
# [r,theta,vr,vtheta] for the spring pendulum). Each has an analytic energy
# exposed in the quasi-invariant hunter so the integrator-hygiene check can
# certify the K≈0 reading is genuine conservation, not numerical dissipation.

# Yang-Mills mechanics (x²y² model): H = ½(px²+py²) + ½ x²y²
def _yang_mills_f(s):
    x, y, px, py = s
    return np.array([px, py, -x * y * y, -x * x * y])


def gen_yang_mills(x0, n, dt):
    return _integrate_flow(_yang_mills_f, x0, n, dt)


# Pullen–Edmonds: H = ½(px²+py²) + ½(x²+y²) + α x²y²
def _pullen_edmonds_f(s, alpha=0.5):
    x, y, px, py = s
    return np.array([px, py,
                     -x - 2.0 * alpha * x * y * y,
                     -y - 2.0 * alpha * x * x * y])


def gen_pullen_edmonds(x0, n, dt):
    return _integrate_flow(_pullen_edmonds_f, x0, n, dt)


# Coupled quartic oscillator (textbook bounded ergodic Hamiltonian):
# H = ½(px²+py²) + ¼(x⁴+y⁴) + ½ b x²y². Fully confining (quartic), so unlike
# the open Barbanis/Hénon-Heiles potentials it never escapes — robust for
# fixed-step RK4 at the chaotic energy.
def _quartic_coupled_f(s, b=0.6):
    x, y, px, py = s
    return np.array([px, py,
                     -x ** 3 - b * x * y * y,
                     -y ** 3 - b * x * x * y])


def gen_quartic_coupled(x0, n, dt):
    return _integrate_flow(_quartic_coupled_f, x0, n, dt)


# Elastic (spring) pendulum in CANONICAL coordinates [r, θ, p_r, p_θ] with
# p_r = m ṙ, p_θ = m r² θ̇. Hamilton's equations make every diagonal Jacobian
# term vanish individually (q̇ depends only on p, ṗ only on q), so divergence
# is structurally zero. Resonant chaos at spring freq ≈ 2× pendulum freq.
# H = p_r²/(2m) + p_θ²/(2 m r²) + ½ k (r−L0)² − m g r cosθ
def _spring_pendulum_f(s, m=1.0, k=39.5, L0=1.0, g=9.81):
    r, th, p_r, p_th = s
    r_safe = r if abs(r) > 1e-6 else (1e-6 if r >= 0 else -1e-6)
    r_dot = p_r / m
    th_dot = p_th / (m * r_safe * r_safe)
    p_r_dot = p_th * p_th / (m * r_safe ** 3) - k * (r - L0) + m * g * math.cos(th)
    p_th_dot = -m * g * r * math.sin(th)
    return np.array([r_dot, th_dot, p_r_dot, p_th_dot])


def gen_spring_pendulum(x0, n, dt):
    return _integrate_flow(_spring_pendulum_f, x0, n, dt)


# Coupled double-well (undamped Hamiltonian Duffing pair):
# H = ½(px²+py²) − ½(x²+y²) + ¼(x⁴+y⁴) + ½ k (x−y)²
def _coupled_duffing_ham_f(s, k=0.30):
    x, y, px, py = s
    return np.array([px, py,
                     x - x ** 3 - k * (x - y),
                     y - y ** 3 + k * (x - y)])


def gen_coupled_duffing_ham(x0, n, dt):
    return _integrate_flow(_coupled_duffing_ham_f, x0, n, dt)


# Circular Restricted Three-Body Problem (planar, Earth-Moon rotating frame)
def _cr3bp_f(state, mu=0.01215):  # Earth-Moon mass parameter
    x, y, vx, vy = state
    r1 = math.sqrt((x + mu) ** 2 + y * y) + 1e-9
    r2 = math.sqrt((x - 1.0 + mu) ** 2 + y * y) + 1e-9
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1.0 + mu) / r2 ** 3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    return np.array([vx, vy, ax, ay])


def gen_cr3bp(x0, n, dt):
    return _integrate_flow(_cr3bp_f, x0, n, dt)


# Lorenz-96 (N=5, F=8 — chaotic)
def _lorenz96_f(x, F=8.0):
    N = x.size
    out = np.empty_like(x)
    for i in range(N):
        out[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return out


def gen_lorenz96(x0, n, dt):
    return _integrate_flow(_lorenz96_f, x0, n, dt)


# Aizawa attractor (3D dissipative, two-lobed bell shape)
def _aizawa_f(s, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f_=0.1):
    x, y, z = s
    return np.array([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z ** 3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f_ * z * x ** 3,
    ])


def gen_aizawa(x0, n, dt):
    return _integrate_flow(_aizawa_f, x0, n, dt)


# Sprott B (one of the algebraically simplest 3D chaotic flows)
def _sprott_b_f(s):
    x, y, z = s
    return np.array([y * z, x - y, 1.0 - x * y])


def gen_sprott_b(x0, n, dt):
    return _integrate_flow(_sprott_b_f, x0, n, dt)


# Halvorsen cyclically symmetric attractor (3D)
def _halvorsen_f(s, a=1.4):
    x, y, z = s
    return np.array([
        -a * x - 4.0 * y - 4.0 * z - y * y,
        -a * y - 4.0 * z - 4.0 * x - z * z,
        -a * z - 4.0 * x - 4.0 * y - x * x,
    ])


def gen_halvorsen(x0, n, dt):
    return _integrate_flow(_halvorsen_f, x0, n, dt)


# Thomas' cyclically symmetric attractor (very low-dim chaos, 3D)
def _thomas_f(s, b=0.208186):
    x, y, z = s
    return np.array([math.sin(y) - b * x,
                     math.sin(z) - b * y,
                     math.sin(x) - b * z])


def gen_thomas(x0, n, dt):
    return _integrate_flow(_thomas_f, x0, n, dt)


# Chen attractor (3D, classical chaotic flow distinct from Lorenz)
def _chen_f(s, a=35.0, b=3.0, c=28.0):
    x, y, z = s
    return np.array([a * (y - x),
                     (c - a) * x - x * z + c * y,
                     x * y - b * z])


def gen_chen(x0, n, dt):
    return _integrate_flow(_chen_f, x0, n, dt)


# Ueda oscillator — duffing-like but with simpler chaotic regime
def _ueda_f(state, k=0.05, B=7.5, omega=1.0):
    x, v, phase = state
    dx = v
    dv = -k * v - x ** 3 + B * math.cos(phase)
    dphase = omega
    return np.array([dx, dv, dphase])


def gen_ueda(x0, n, dt):
    return _integrate_flow(_ueda_f, x0, n, dt)


# Lorenz-96 N=10 (higher-dimensional version, still chaotic at F=8)
def gen_lorenz96_n10(x0, n, dt):
    return _integrate_flow(_lorenz96_f, x0, n, dt)


# Mackey-Glass delay differential equation (1D state, infinite-dim phase
# space). We integrate with explicit Euler over a circular buffer holding the
# last τ/dt samples; the chaotic regime is at τ=17 (β=0.2, γ=0.1, n_exp=10).
def gen_mackey_glass(x0, n, dt, beta=0.2, gamma=0.1, n_exp=10, tau=17.0):
    delay_steps = max(2, int(round(tau / dt)))
    # warm history seeded near the fixed point
    hist = np.full(delay_steps + 1, float(x0[0]), dtype=np.float64)
    head = 0
    out = np.empty((n, 1), dtype=np.float32)
    x = float(x0[0])
    for i in range(n):
        out[i, 0] = x
        x_tau = hist[(head - delay_steps) % hist.size]
        dx = beta * x_tau / (1.0 + x_tau ** n_exp) - gamma * x
        x = x + dt * dx
        head = (head + 1) % hist.size
        hist[head] = x
    return out


# ── Discrete-time maps ──────────────────────────────────────────────────────

def gen_logistic(x0, n, dt, r=3.9):
    out = np.empty((n, 1), dtype=np.float32)
    x = float(x0[0])
    for i in range(n):
        out[i, 0] = x
        x = r * x * (1.0 - x)
    return out


def gen_henon_map(x0, n, dt, a=1.4, b=0.3):
    out = np.empty((n, 2), dtype=np.float32)
    x, y = float(x0[0]), float(x0[1])
    for i in range(n):
        out[i, 0] = x
        out[i, 1] = y
        x_new = 1.0 - a * x * x + y
        y_new = b * x
        x, y = x_new, y_new
    return out


def gen_standard_map(x0, n, dt, K=1.2):
    # Chirikov standard map on the torus
    out = np.empty((n, 2), dtype=np.float32)
    p, theta = float(x0[0]), float(x0[1])
    two_pi = 2.0 * math.pi
    for i in range(n):
        out[i, 0] = p
        out[i, 1] = theta
        p = (p + K * math.sin(theta)) % two_pi
        theta = (theta + p) % two_pi
    return out


# ── Controls ────────────────────────────────────────────────────────────────

def gen_harmonic(x0, n, dt, omega=1.0):
    # Pure SHO — not chaotic
    out = np.empty((n, 2), dtype=np.float32)
    t = np.arange(n) * dt
    A = math.sqrt(x0[0] ** 2 + (x0[1] / omega) ** 2)
    phi = math.atan2(-x0[1] / omega, x0[0])
    out[:, 0] = A * np.cos(omega * t + phi)
    out[:, 1] = -A * omega * np.sin(omega * t + phi)
    return out


def gen_quasiperiodic(x0, n, dt):
    # Two incommensurate frequencies — non-chaotic but space-filling on a torus
    out = np.empty((n, 2), dtype=np.float32)
    t = np.arange(n) * dt
    out[:, 0] = np.cos(t) + np.cos(np.sqrt(2.0) * t)
    out[:, 1] = np.sin(t) + np.sin(np.sqrt(2.0) * t)
    return out


def gen_random_walk(x0, n, dt):
    rng = np.random.default_rng(int(x0[0] * 1e6) ^ GLOBAL_SEED)
    steps = rng.standard_normal(size=(n, 2)).astype(np.float32) * 0.1
    return np.cumsum(steps, axis=0)


# ── System registry ────────────────────────────────────────────────────────

def build_zoo() -> list[SystemSpec]:
    specs: list[SystemSpec] = []

    # Flows — dissipative
    specs.append(SystemSpec("lorenz63", "flow_dissipative", 3, gen_lorenz,
                            np.array([1.0, 1.0, 1.0]), 0.01,
                            description="Lorenz 1963 — canonical strange attractor"))
    specs.append(SystemSpec("rossler", "flow_dissipative", 3, gen_rossler,
                            np.array([1.0, 1.0, 0.0]), 0.05,
                            description="Rössler — single spiraling lobe"))
    specs.append(SystemSpec("duffing_forced", "flow_dissipative", 3, gen_duffing,
                            np.array([0.5, 0.0, 0.0]), 0.05,
                            description="Forced Duffing oscillator (chaotic regime)"))
    specs.append(SystemSpec("lorenz96_n5", "flow_dissipative", 5, gen_lorenz96,
                            np.array([8.0, 8.0, 8.0, 8.01, 8.0]), 0.02,
                            description="Lorenz-96 N=5, F=8 — atmospheric toy model"))
    specs.append(SystemSpec("aizawa", "flow_dissipative", 3, gen_aizawa,
                            np.array([0.1, 0.0, 0.0]), 0.01,
                            description="Aizawa attractor — two-lobed bell-shaped chaos"))
    specs.append(SystemSpec("sprott_b", "flow_dissipative", 3, gen_sprott_b,
                            np.array([0.05, 0.05, 0.0]), 0.02,
                            description="Sprott B — algebraically simplest 3D chaos"))
    specs.append(SystemSpec("halvorsen", "flow_dissipative", 3, gen_halvorsen,
                            np.array([-5.0, 0.0, 0.0]), 0.01,
                            description="Halvorsen cyclically symmetric attractor"))
    specs.append(SystemSpec("thomas", "flow_dissipative", 3, gen_thomas,
                            np.array([0.1, 0.0, 0.0]), 0.05,
                            description="Thomas cyclically symmetric (very low-dim chaos)"))
    specs.append(SystemSpec("chen", "flow_dissipative", 3, gen_chen,
                            np.array([-10.0, 0.0, 37.0]), 0.005,
                            description="Chen attractor — Lorenz-like alternate"))
    specs.append(SystemSpec("ueda_forced", "flow_dissipative", 3, gen_ueda,
                            np.array([2.5, 0.0, 0.0]), 0.05,
                            description="Ueda forced oscillator — pure-cubic Duffing"))
    specs.append(SystemSpec("lorenz96_n10", "flow_dissipative", 10, gen_lorenz96_n10,
                            np.array([8.0, 8.0, 8.0, 8.01, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
                            0.02,
                            description="Lorenz-96 N=10, F=8 — 10D high-dim chaos"))

    # Flows — Hamiltonian
    specs.append(SystemSpec("double_pendulum", "flow_hamiltonian", 4, gen_double_pendulum,
                            np.array([math.pi / 2.0, math.pi / 2.0 + 0.01, 0.0, 0.0]), 0.005,
                            known_invariants=["energy_H"],
                            description="Double pendulum — visible Hamiltonian chaos"))
    specs.append(SystemSpec("henon_heiles", "flow_hamiltonian", 4, gen_henon_heiles,
                            np.array([0.0, 0.1, 0.5, 0.0]), 0.02,
                            known_invariants=["energy_H"],
                            description="Hénon–Heiles — quartic stellar potential"))
    specs.append(SystemSpec("cr3bp_earth_moon", "flow_hamiltonian", 4, gen_cr3bp,
                            np.array([0.5, 0.0, 0.0, 0.8]), 0.01,
                            known_invariants=["jacobi_C"],
                            description="Circular restricted 3-body, Earth–Moon"))
    specs.append(SystemSpec("yang_mills_x2y2", "flow_hamiltonian", 4, gen_yang_mills,
                            np.array([0.4, 0.3, 0.2, 0.15]), 0.02,
                            known_invariants=["energy_H"],
                            description="Yang-Mills mechanics x²y² — ergodic Hamiltonian chaos"))
    specs.append(SystemSpec("pullen_edmonds", "flow_hamiltonian", 4, gen_pullen_edmonds,
                            np.array([1.5, 1.5, 1.0, 1.0]), 0.02,
                            known_invariants=["energy_H"],
                            description="Pullen–Edmonds coupled quartic Hamiltonian (E≈5.8, compact chaotic)"))
    specs.append(SystemSpec("quartic_coupled", "flow_hamiltonian", 4, gen_quartic_coupled,
                            np.array([0.8, 0.8, 0.6, 0.6]), 0.02,
                            known_invariants=["energy_H"],
                            description="Coupled quartic oscillator — bounded ergodic Hamiltonian chaos"))
    specs.append(SystemSpec("spring_pendulum", "flow_hamiltonian", 4, gen_spring_pendulum,
                            np.array([1.0, 0.6, 0.0, 0.0]), 0.005,
                            known_invariants=["energy_H"],
                            description="Elastic (spring) pendulum at 2:1 resonance"))
    specs.append(SystemSpec("coupled_duffing_ham", "flow_hamiltonian", 4, gen_coupled_duffing_ham,
                            np.array([0.6, -0.5, 0.0, 0.1]), 0.02,
                            known_invariants=["energy_H"],
                            description="Coupled double-well Hamiltonian Duffing pair"))

    # Maps
    specs.append(SystemSpec("logistic_r3p9", "map", 1, gen_logistic,
                            np.array([0.4]), 1.0,
                            description="Logistic map r=3.9 — paradigm 1D chaos"))
    specs.append(SystemSpec("henon_map", "map", 2, gen_henon_map,
                            np.array([0.1, 0.0]), 1.0,
                            description="Hénon map (a=1.4, b=0.3) — fractal attractor"))
    specs.append(SystemSpec("standard_map_K1p2", "map", 2, gen_standard_map,
                            np.array([0.5, 0.5]), 1.0,
                            description="Chirikov standard map K=1.2 — mixed phase space"))

    # Delay-differential — Mackey-Glass; chaotic regime at τ=17. Treated as a
    # flow with infinite-dimensional phase space but exposed as 1D state.
    specs.append(SystemSpec("mackey_glass_tau17", "flow_dissipative", 1, gen_mackey_glass,
                            np.array([0.5]), 0.1,
                            description="Mackey-Glass delay equation τ=17 — infinite-dim chaos"))

    # Controls — non-chaotic
    specs.append(SystemSpec("harmonic", "control", 2, gen_harmonic,
                            np.array([1.0, 0.0]), 0.05,
                            known_invariants=["energy_H"],
                            description="Simple harmonic oscillator (regular control)"))
    specs.append(SystemSpec("quasiperiodic_2tori", "control", 2, gen_quasiperiodic,
                            np.array([0.0, 0.0]), 0.05,
                            description="Two-frequency torus (non-chaotic control)"))
    specs.append(SystemSpec("random_walk", "control", 2, gen_random_walk,
                            np.array([0.12345]), 1.0,
                            description="Brownian motion (stochastic control)"))

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# (2) Dynamical fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _continuous_vector_field(name: str):
    """Map a system name to its continuous f(state) when one exists."""
    return {
        "lorenz63": _lorenz_f,
        "rossler": _rossler_f,
        "duffing_forced": _duffing_f,
        "lorenz96_n5": _lorenz96_f,
        "double_pendulum": _double_pendulum_f,
        "henon_heiles": _hh_f,
        "cr3bp_earth_moon": _cr3bp_f,
        "yang_mills_x2y2": _yang_mills_f,
        "pullen_edmonds": _pullen_edmonds_f,
        "quartic_coupled": _quartic_coupled_f,
        "spring_pendulum": _spring_pendulum_f,
        "coupled_duffing_ham": _coupled_duffing_ham_f,
        "aizawa": _aizawa_f,
        "sprott_b": _sprott_b_f,
        "halvorsen": _halvorsen_f,
        "thomas": _thomas_f,
        "chen": _chen_f,
        "ueda_forced": _ueda_f,
        "lorenz96_n10": _lorenz96_f,
    }.get(name)


def lyapunov_continuous(spec: SystemSpec, n_steps: int = 4000,
                        renorm_every: int = 25, eps: float = 1e-8) -> float:
    """Benettin's algorithm for the largest Lyapunov exponent of a flow."""
    f = _continuous_vector_field(spec.name)
    if f is None:
        return float("nan")
    rng = _rng(11)
    x = spec.base_ic.astype(np.float64).copy()
    # warm-up
    for _ in range(400):
        x = _rk4_step(f, x, spec.dt)
    pert = rng.standard_normal(x.shape)
    pert = pert / np.linalg.norm(pert) * eps
    y = x + pert
    log_sum = 0.0
    n_renorms = 0
    for step in range(n_steps):
        x = _rk4_step(f, x, spec.dt)
        y = _rk4_step(f, y, spec.dt)
        if (step + 1) % renorm_every == 0:
            d = y - x
            dnorm = float(np.linalg.norm(d))
            if dnorm <= 0:
                continue
            log_sum += math.log(dnorm / eps)
            y = x + d * (eps / dnorm)
            n_renorms += 1
    if n_renorms == 0:
        return float("nan")
    return log_sum / (n_renorms * renorm_every * spec.dt)


def local_lyapunov_distribution(spec: SystemSpec, n_windows: int = 40,
                                window_steps: int = 200, renorm_every: int = 10,
                                eps: float = 1e-8) -> dict:
    """Run Benettin in many short windows along one long trajectory; return the
    *distribution* of local Lyapunov estimates.

    Chaotic flows have a wide distribution because the stretching rate varies
    along the attractor (regions of strong divergence, regions of folding).
    Regular motion (harmonic, quasi-periodic) has a narrow distribution
    centered at zero. The std of this distribution is the discriminator that
    separates 'actually chaotic' from 'regular' even when both have small λ₁.
    """
    f = _continuous_vector_field(spec.name)
    if f is None:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "range": float("nan"), "n_windows": 0, "method": "continuous"}
    rng = _rng(23)
    x = spec.base_ic.astype(np.float64).copy()
    # warm-up to attractor
    for _ in range(400):
        x = _rk4_step(f, x, spec.dt)
    locals_: list[float] = []
    for _ in range(n_windows):
        # fresh perturbation per window so each estimate is locally honest
        pert = rng.standard_normal(x.shape)
        pn = float(np.linalg.norm(pert))
        if pn <= 0:
            continue
        pert = pert / pn * eps
        y = x + pert
        log_sum = 0.0
        n_renorms = 0
        for step in range(window_steps):
            x = _rk4_step(f, x, spec.dt)
            y = _rk4_step(f, y, spec.dt)
            if (step + 1) % renorm_every == 0:
                d = y - x
                dnorm = float(np.linalg.norm(d))
                if dnorm <= 0:
                    continue
                log_sum += math.log(dnorm / eps)
                y = x + d * (eps / dnorm)
                n_renorms += 1
        if n_renorms == 0:
            continue
        locals_.append(log_sum / (n_renorms * renorm_every * spec.dt))
    if not locals_:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "range": float("nan"), "n_windows": 0, "method": "continuous"}
    arr = np.array(locals_)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "range": float(arr.max() - arr.min()),
        "n_windows": int(arr.size),
        "method": "continuous",
    }


def local_lyapunov_distribution_map(spec: SystemSpec, n_windows: int = 40,
                                    window_steps: int = 200, eps: float = 1e-8) -> dict:
    """Same idea but for discrete maps — renormalize every step."""
    if spec.family != "map":
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "range": float("nan"), "n_windows": 0, "method": "map"}
    rng = _rng(25)
    base = spec.integrate(spec.base_ic, 400, spec.dt)
    x = base[-1].astype(np.float64).copy()
    locals_: list[float] = []
    for _ in range(n_windows):
        pert = rng.standard_normal(x.shape)
        pn = float(np.linalg.norm(pert))
        if pn <= 0:
            continue
        pert = pert / pn * eps
        y = x + pert
        log_sum = 0.0
        n_used = 0
        for _ in range(window_steps):
            nx = spec.integrate(x, 2, spec.dt)[-1].astype(np.float64)
            ny = spec.integrate(y, 2, spec.dt)[-1].astype(np.float64)
            x, y = nx, ny
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                x = base[-1].astype(np.float64).copy()
                pert = rng.standard_normal(x.shape)
                pn = float(np.linalg.norm(pert))
                if pn <= 0:
                    break
                pert = pert / pn * eps
                y = x + pert
                continue
            d = y - x
            dnorm = float(np.linalg.norm(d))
            if dnorm <= 1e-15:
                pert = rng.standard_normal(x.shape)
                pn = float(np.linalg.norm(pert))
                if pn <= 0:
                    break
                pert = pert / pn * eps
                y = x + pert
                continue
            log_sum += math.log(dnorm / eps)
            y = x + d * (eps / dnorm)
            n_used += 1
        if n_used == 0:
            continue
        locals_.append(log_sum / n_used)
    if not locals_:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "range": float("nan"), "n_windows": 0, "method": "map"}
    arr = np.array(locals_)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "range": float(arr.max() - arr.min()),
        "n_windows": int(arr.size),
        "method": "map",
    }


def k_trajectory_length(family: str, base_n: int) -> int:
    """Trajectory length for the 0-1 K test. Slow-mixing continuous flows need
    a longer series than the fingerprint default so the Gottwald–Melbourne
    diffusion has room to manifest — at the short length, weakly-mixing flows
    (Aizawa, Thomas) under-report K. Maps mix fast and need no extension."""
    if family in ("flow_dissipative", "flow_hamiltonian"):
        return min(3 * base_n, 15000)
    return base_n


def _first_decorrelation_lag(x: np.ndarray, target: float = 0.2,
                             max_lag: int = 200) -> int:
    """Smallest lag at which |autocorr(x, lag)| drops below `target`. Used as
    the chaos-natural sampling stride for the GM 0-1 test on continuous flows."""
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return 1
    upper = min(max_lag, x.size - 2)
    for lag in range(1, upper + 1):
        ac = float((x[:-lag] * x[lag:]).mean() / var)
        if abs(ac) < target:
            return lag
    return upper if upper > 0 else 1


def phase_volume_contraction(spec: SystemSpec, traj: np.ndarray,
                             n_sample: int = 600, h: float = 1e-5) -> dict:
    """Mean divergence of the vector field along the trajectory.

    For a flow ẋ = f(x), div f(x) = Σ_i ∂f_i/∂x_i is the local rate of
    phase-space volume change, and its trajectory average equals the sum of
    all Lyapunov exponents (the phase-space contraction rate). This is a
    *first-principles* label of conservative vs dissipative dynamics that is
    completely independent of the Gottwald–Melbourne 0-1 test:

      - Hamiltonian flows obey Liouville's theorem → div f ≡ 0 (to numerical
        precision). They neither contract nor expand phase-space volume.
      - Dissipative flows contract volume onto a lower-dimensional attractor
        → mean div f < 0.

    If this orthogonal measure agrees with the 0-1 K split (Hamiltonian K≈0
    AND div≈0; dissipative K≈1 AND div<0), the conservative/dissipative
    distinction is confirmed by two independent methods.

    Returns mean/std/p10/p90 of div f over sampled trajectory points.
    Maps and stochastic controls have no continuous vector field → nan.
    """
    f = _continuous_vector_field(spec.name)
    if f is None or traj.shape[0] < 20:
        return {"mean_divergence": float("nan"), "std_divergence": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "n_sampled": 0, "available": False}
    rng = _rng(41)
    idx = rng.choice(traj.shape[0], size=min(n_sample, traj.shape[0]), replace=False)
    dim = traj.shape[1]
    divs = []
    for k in idx:
        x = traj[k].astype(np.float64)
        if not np.isfinite(x).all():
            continue
        d = 0.0
        ok = True
        for i in range(dim):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fp = f(xp); fm = f(xm)
            if not (np.isfinite(fp[i]) and np.isfinite(fm[i])):
                ok = False
                break
            d += (fp[i] - fm[i]) / (2.0 * h)
        if ok and np.isfinite(d):
            divs.append(d)
    if not divs:
        return {"mean_divergence": float("nan"), "std_divergence": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "n_sampled": 0, "available": False}
    arr = np.array(divs)
    return {
        "mean_divergence": float(arr.mean()),
        "std_divergence": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "n_sampled": int(arr.size),
        "available": True,
    }


def gottwald_melbourne_01_test(traj: np.ndarray, n_c: int = 12,
                               n_max: int | None = None,
                               target_samples: int = 1500) -> dict:
    """0-1 test for chaos (Gottwald & Melbourne, 2009).

    For a univariate observable φ(j) (we use the first state component),
    define translation variables

        p_c(n) = Σ_{j=1}^n φ(j) cos(j c)
        q_c(n) = Σ_{j=1}^n φ(j) sin(j c)

    and the mean-square displacement

        M_c(n) = (1/N) Σ_{j=1}^{N-n} (p_c(j+n)-p_c(j))² + (q_c(j+n)-q_c(j))²

    Then K_c = corr(log n, log M_c(n)) over a band of n. K averaged over many
    random c ∈ (π/5, 4π/5) is ≈ 0 for regular dynamics and ≈ 1 for chaos.
    Robust against noise and against the choice of observable.

    Implementation note (important for continuous flows). The GM test assumes
    the observable is sampled at a chaos-natural rate, not at the integrator's
    RK4 timestep where consecutive samples are nearly identical. We pick the
    decimation stride as the first decorrelation lag of φ (where autocorr
    drops below 0.2), so each sampled value carries independent information
    about the dynamics. For maps the natural stride is 1.
    """
    if traj.shape[0] < 200:
        return {"K_median": float("nan"), "K_mean": float("nan"),
                "K_std": float("nan"), "n_c": 0,
                "K_per_c": [], "stride": 0}
    phi_full = traj[:, 0].astype(np.float64)
    # Pick a system-natural stride. Floor it so we always get enough samples.
    decorr_lag = _first_decorrelation_lag(phi_full)
    min_stride_for_target = max(1, phi_full.size // target_samples)
    stride = max(min_stride_for_target, min(decorr_lag, phi_full.size // 200))
    stride = max(stride, 1)
    phi = phi_full[::stride]
    # Mean-subtract to suppress drift artifacts (recommended by GM in their later
    # papers; helps for non-zero-mean observables).
    phi = phi - phi.mean()
    N = phi.size
    if n_max is None:
        n_max = max(20, N // 10)
    n_max = min(n_max, N // 5)
    if n_max < 10:
        return {"K_median": float("nan"), "K_mean": float("nan"),
                "K_std": float("nan"), "n_c": 0, "K_per_c": []}
    n_grid = np.unique(np.round(np.geomspace(2, n_max, 30)).astype(int))
    rng = _rng(31)
    c_values = rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0, size=n_c)
    Ks = []
    j = np.arange(1, N + 1, dtype=np.float64)
    for c in c_values:
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))
        Ms = []
        for n in n_grid:
            if n >= N:
                continue
            dp = p[n:] - p[:-n]
            dq = q[n:] - q[:-n]
            Ms.append(float(((dp * dp + dq * dq).mean())))
        Ms = np.asarray(Ms, dtype=np.float64)
        if Ms.size < 5 or (Ms <= 0).all():
            continue
        valid = Ms > 0
        if valid.sum() < 5:
            continue
        log_n = np.log(n_grid[:Ms.size][valid].astype(np.float64))
        log_M = np.log(Ms[valid])
        # correlation coefficient
        ln, lM = log_n - log_n.mean(), log_M - log_M.mean()
        denom = math.sqrt(float((ln * ln).sum() * (lM * lM).sum()))
        if denom <= 0:
            continue
        K = float((ln * lM).sum() / denom)
        Ks.append(K)
    if not Ks:
        return {"K_median": float("nan"), "K_mean": float("nan"),
                "K_std": float("nan"), "n_c": 0, "K_per_c": [], "stride": stride}
    arr = np.array(Ks)
    return {
        "K_median": float(np.median(arr)),
        "K_mean": float(arr.mean()),
        "K_std": float(arr.std()),
        "n_c": int(arr.size),
        "K_per_c": [float(k) for k in arr.tolist()],
        "stride": int(stride),
        "n_samples_used": int(phi.size),
    }


def lyapunov_map(spec: SystemSpec, n_steps: int = 5000, eps: float = 1e-8) -> float:
    """Benettin for discrete maps — renormalize every step to keep distances
    bounded and avoid NaN blow-ups on unbounded maps."""
    if spec.family != "map":
        return float("nan")
    rng = _rng(13)
    base = spec.integrate(spec.base_ic, 400, spec.dt)
    x = base[-1].astype(np.float64).copy()
    pert = rng.standard_normal(x.shape)
    pert = pert / np.linalg.norm(pert) * eps
    y = x + pert
    log_sum = 0.0
    n_used = 0
    for _ in range(n_steps):
        nx = spec.integrate(x, 2, spec.dt)[-1].astype(np.float64)
        ny = spec.integrate(y, 2, spec.dt)[-1].astype(np.float64)
        x, y = nx, ny
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            # escape — re-seed both at the warm-up endpoint
            x = base[-1].astype(np.float64).copy()
            pert = rng.standard_normal(x.shape)
            pert = pert / np.linalg.norm(pert) * eps
            y = x + pert
            continue
        d = y - x
        dnorm = float(np.linalg.norm(d))
        if dnorm <= 1e-15:
            pert = rng.standard_normal(x.shape)
            pert = pert / np.linalg.norm(pert) * eps
            y = x + pert
            continue
        log_sum += math.log(dnorm / eps)
        y = x + d * (eps / dnorm)
        n_used += 1
    if n_used == 0:
        return float("nan")
    return log_sum / n_used


def correlation_dimension(traj: np.ndarray, n_sample: int = 1000,
                          n_radii: int = 18) -> float:
    """Grassberger–Procaccia: slope of log C(r) vs log r in the scaling band."""
    if traj.shape[0] < 50:
        return float("nan")
    rng = _rng(17)
    idx = rng.choice(traj.shape[0], size=min(n_sample, traj.shape[0]), replace=False)
    pts = traj[idx].astype(np.float64)
    # pairwise distances (upper triangle only)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1))
    iu = np.triu_indices_from(d, k=1)
    dists = d[iu]
    dists = dists[dists > 0]
    if dists.size == 0:
        return float("nan")
    r_min = np.quantile(dists, 0.02)
    r_max = np.quantile(dists, 0.6)
    if not (r_max > r_min > 0):
        return float("nan")
    radii = np.geomspace(r_min, r_max, n_radii)
    counts = np.array([(dists <= r).sum() for r in radii], dtype=np.float64)
    counts = counts / max(dists.size, 1)
    mask = counts > 0
    if mask.sum() < 4:
        return float("nan")
    log_r = np.log(radii[mask])
    log_c = np.log(counts[mask])
    # use middle 60% for the slope
    lo = int(0.2 * log_r.size)
    hi = int(0.8 * log_r.size)
    if hi - lo < 3:
        lo, hi = 0, log_r.size
    slope, _ = np.polyfit(log_r[lo:hi], log_c[lo:hi], 1)
    return float(slope)


def spectral_entropy(signal: np.ndarray) -> float:
    """Normalized Shannon entropy of the power spectrum of the first component."""
    if signal.ndim == 2:
        signal = signal[:, 0]
    s = signal - signal.mean()
    if s.std() == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(s)) ** 2
    spec = spec[1:]  # drop DC
    p = spec / max(spec.sum(), 1e-30)
    p = p[p > 0]
    H = -float((p * np.log(p)).sum())
    return H / math.log(p.size) if p.size > 1 else 0.0


def hankel_dmd(traj: np.ndarray, delay: int = 16, rank: int = 12) -> dict:
    """Exact DMD on a delay-embedded trajectory; return Koopman descriptors."""
    if traj.shape[0] < delay + 10:
        return {"radius": float("nan"), "gap": float("nan"),
                "top_moduli": [], "imag_fraction": float("nan")}
    # Build Hankel block on the first state component (works for any system)
    x = traj[:, 0].astype(np.float64)
    T = x.size - delay
    H = np.empty((delay, T))
    for i in range(delay):
        H[i] = x[i:i + T]
    X1, X2 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, S.size, max(1, (S > 1e-10 * S[0]).sum()))
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r]
    A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1.0 / np.maximum(S_r, 1e-12))
    eigs = np.linalg.eigvals(A_tilde)
    mods = np.sort(np.abs(eigs))[::-1]
    radius = float(mods[0]) if mods.size else float("nan")
    gap = float(mods[0] - mods[1]) if mods.size >= 2 else float("nan")
    imag_fraction = float((np.abs(eigs.imag) > 1e-6).mean())
    return {
        "radius": radius,
        "gap": gap,
        "top_moduli": [float(m) for m in mods[:5]],
        "imag_fraction": imag_fraction,
    }


def transfer_operator_gap(traj: np.ndarray, n_bins: int = 18) -> dict:
    """Ulam discretization → row-stochastic transition matrix → spectral gap.

    The first eigenvalue is 1 (stochastic). 1 − |λ₂| is the decorrelation rate
    along the attractor; the "weather forgetting" speed."""
    if traj.shape[0] < 200:
        return {"gap": float("nan"), "n_used_bins": 0, "second_modulus": float("nan")}
    # project to 2D — for 1D systems, use the value and a delay-1 copy so the
    # Ulam grid still sees attractor structure.
    if traj.shape[1] >= 2:
        X = traj - traj.mean(axis=0, keepdims=True)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        proj = (U[:, :2] * S[:2])
    else:
        v = traj[:, 0].astype(np.float64)
        proj = np.stack([v[:-1], v[1:]], axis=1)
    # bin
    x_edges = np.linspace(proj[:, 0].min(), proj[:, 0].max(), n_bins + 1)
    y_edges = np.linspace(proj[:, 1].min(), proj[:, 1].max(), n_bins + 1)
    xi = np.clip(np.searchsorted(x_edges, proj[:, 0], side="right") - 1, 0, n_bins - 1)
    yi = np.clip(np.searchsorted(y_edges, proj[:, 1], side="right") - 1, 0, n_bins - 1)
    cell = xi * n_bins + yi
    # transitions cell[t] → cell[t+1]
    src, dst = cell[:-1], cell[1:]
    K = n_bins * n_bins
    M = np.zeros((K, K), dtype=np.float64)
    for s, d in zip(src, dst):
        M[s, d] += 1.0
    row_sums = M.sum(axis=1)
    occupied = row_sums > 0
    M = M[occupied][:, occupied]
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    M = M / rs
    if M.shape[0] < 2:
        return {"gap": float("nan"), "n_used_bins": int(occupied.sum()),
                "second_modulus": float("nan")}
    eigs = np.linalg.eigvals(M)
    mods = np.sort(np.abs(eigs))[::-1]
    gap = float(1.0 - mods[1]) if mods.size >= 2 else float("nan")
    return {"gap": gap, "n_used_bins": int(occupied.sum()),
            "second_modulus": float(mods[1]) if mods.size >= 2 else float("nan")}


def foundation_predictability(traj: np.ndarray, hidden: int = 32,
                              train_steps: int = 300, dt: float = 1.0,
                              seed: int = 0) -> dict:
    """Train a Foundation LiquidPredictor briefly; report normalized eval MSE.

    Normalized against the per-step persistence baseline MSE so values <1 mean
    the predictor beats persistence. Returns NaN fields when the Foundation
    model is not available (standalone package without models/)."""
    _nan = {"lnn_mse": float("nan"), "persistence_mse": float("nan"),
            "skill": float("nan")}
    if not _HAS_LIQUID or torch is None:
        return _nan
    if traj.shape[0] < 400:
        return _nan
    torch.manual_seed(seed)
    np.random.seed(seed)
    sig = traj.astype(np.float32)
    # standardize per channel
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    sig = (sig - mu) / sd
    half = sig.shape[0] // 2
    train_seq = torch.tensor(sig[:half]).unsqueeze(0)            # (1, T, d)
    eval_seq = torch.tensor(sig[half:]).unsqueeze(0)
    d = sig.shape[1]
    model = _LiquidPredictor(input_size=d, hidden_size=hidden, dt=dt, multi_scale=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    # windowed training over the first half
    win = 200
    n_wins = max(1, (train_seq.shape[1] - win) // 20)
    for step in range(train_steps):
        start = (step * 7) % max(1, train_seq.shape[1] - win - 1)
        batch = train_seq[:, start:start + win, :]
        preds, _ = model(batch)
        loss = ((preds - batch[:, 1:, :]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        eval_preds, _ = model(eval_seq)
        lnn_mse = float(((eval_preds - eval_seq[:, 1:, :]) ** 2).mean().item())
        persistence = eval_seq[:, :-1, :]
        pers_mse = float(((persistence - eval_seq[:, 1:, :]) ** 2).mean().item())
    skill = 1.0 - lnn_mse / max(pers_mse, 1e-12)
    return {"lnn_mse": lnn_mse, "persistence_mse": pers_mse, "skill": skill,
            "hidden": hidden, "train_steps": train_steps,
            "n_wins_per_epoch": n_wins}


# ─────────────────────────────────────────────────────────────────────────────
# (3) Quasi-invariant hunter (AI-Poincaré style)
# ─────────────────────────────────────────────────────────────────────────────

def _monomial_basis_indices(d: int, max_degree: int = 2) -> list[tuple]:
    """All monomial exponent tuples up to total degree max_degree, excluding
    the constant 1. Includes degree-1, degree-2 (squares and cross terms),
    and (if requested) degree-3 (cubes, x²y, xyz triples).

    For high-dimensional systems (d ≥ 6) we cap at degree 2 because the
    degree-3 basis grows as O(d³) and offers limited interpretability for
    coupled high-dim systems like Lorenz-96 N=10."""
    out: list[tuple] = []
    effective_max = max_degree if d < 6 else min(max_degree, 2)
    if effective_max >= 1:
        for i in range(d):
            e = [0] * d
            e[i] = 1
            out.append(tuple(e))
    if effective_max >= 2:
        for i in range(d):
            e = [0] * d
            e[i] = 2
            out.append(tuple(e))
            for j in range(i + 1, d):
                e2 = [0] * d
                e2[i] = 1
                e2[j] = 1
                out.append(tuple(e2))
    if effective_max >= 3:
        # cubes x_i^3
        for i in range(d):
            e = [0] * d
            e[i] = 3
            out.append(tuple(e))
        # x_i^2 · x_j  and  x_i · x_j^2  for i != j
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                e = [0] * d
                e[i] = 2
                e[j] = 1
                out.append(tuple(e))
        # triples x_i · x_j · x_k  for i < j < k
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    e = [0] * d
                    e[i] = 1
                    e[j] = 1
                    e[k] = 1
                    out.append(tuple(e))
    return out


def _eval_monomial(traj: np.ndarray, exponents: tuple) -> np.ndarray:
    out = np.ones(traj.shape[0], dtype=np.float64)
    for k, e in enumerate(exponents):
        if e == 0:
            continue
        out *= traj[:, k].astype(np.float64) ** e
    return out


def _autocorrelation(x: np.ndarray, lags: list[int]) -> list[float]:
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return [float("nan")] * len(lags)
    out = []
    for L in lags:
        if L >= x.size or L <= 0:
            out.append(float("nan"))
            continue
        ac = float((x[:-L] * x[L:]).mean()) / var
        out.append(ac)
    return out


def hunt_quasi_invariants(spec: SystemSpec, traj: np.ndarray, lambda1: float,
                          max_degree: int = 2, top_k: int = 5) -> dict:
    """For each monomial Q over the state, measure how flat Q(t) is along the
    trajectory and how long it survives in Lyapunov time before scrambling.

    Survival time: the time at which |Q(t)−Q(0)| first exceeds the long-run
    std(Q), expressed in Lyapunov units τ_λ = 1/λ₁."""
    d = traj.shape[1]
    basis = _monomial_basis_indices(d, max_degree=max_degree)
    T = traj.shape[0]
    dt = spec.dt
    candidates = []
    # Also include the system's first-principles invariant if known
    known_series = {}
    if spec.name == "double_pendulum":
        # Canonical coords [θ1, θ2, p1, p2], m1=m2=l1=l2=1:
        # H = ½ (p1² − 2 p1 p2 cosΔ + 2 p2²)/(2 − cos²Δ) − 2g cosθ1 − g cosθ2
        th1, th2, p1, p2 = [traj[:, i].astype(np.float64) for i in range(4)]
        g = 9.81
        c = np.cos(th1 - th2)
        D = 2.0 - c * c
        T_kin = 0.5 * (p1 ** 2 - 2.0 * p1 * p2 * c + 2.0 * p2 ** 2) / D
        V = -2.0 * g * np.cos(th1) - g * np.cos(th2)
        known_series["energy_H"] = T_kin + V
    elif spec.name == "henon_heiles":
        x, y, px, py = [traj[:, i].astype(np.float64) for i in range(4)]
        known_series["energy_H"] = 0.5 * (px ** 2 + py ** 2) + 0.5 * (x ** 2 + y ** 2) \
            + x ** 2 * y - y ** 3 / 3.0
    elif spec.name == "cr3bp_earth_moon":
        mu = 0.01215
        x, y, vx, vy = [traj[:, i].astype(np.float64) for i in range(4)]
        r1 = np.sqrt((x + mu) ** 2 + y ** 2) + 1e-9
        r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2) + 1e-9
        U = (1.0 - mu) / r1 + mu / r2 + 0.5 * (x ** 2 + y ** 2)
        # Jacobi constant C = 2U − (vx²+vy²)
        known_series["jacobi_C"] = 2.0 * U - (vx ** 2 + vy ** 2)
    elif spec.name == "harmonic":
        known_series["energy_H"] = 0.5 * (traj[:, 0] ** 2 + traj[:, 1] ** 2)
    elif spec.name == "yang_mills_x2y2":
        x, y, px, py = [traj[:, i].astype(np.float64) for i in range(4)]
        known_series["energy_H"] = 0.5 * (px ** 2 + py ** 2) + 0.5 * x ** 2 * y ** 2
    elif spec.name == "pullen_edmonds":
        alpha = 0.5
        x, y, px, py = [traj[:, i].astype(np.float64) for i in range(4)]
        known_series["energy_H"] = (0.5 * (px ** 2 + py ** 2)
                                    + 0.5 * (x ** 2 + y ** 2)
                                    + alpha * x ** 2 * y ** 2)
    elif spec.name == "quartic_coupled":
        b = 0.6
        x, y, px, py = [traj[:, i].astype(np.float64) for i in range(4)]
        known_series["energy_H"] = (0.5 * (px ** 2 + py ** 2)
                                    + 0.25 * (x ** 4 + y ** 4)
                                    + 0.5 * b * x ** 2 * y ** 2)
    elif spec.name == "spring_pendulum":
        m, k, L0, g = 1.0, 39.5, 1.0, 9.81
        # Canonical coords [r, θ, p_r, p_θ]:
        # H = p_r²/(2m) + p_θ²/(2 m r²) + ½ k (r−L0)² − m g r cosθ
        r, th, p_r, p_th = [traj[:, i].astype(np.float64) for i in range(4)]
        r_safe = np.where(np.abs(r) > 1e-9, r, 1e-9)
        known_series["energy_H"] = (p_r ** 2 / (2.0 * m)
                                    + p_th ** 2 / (2.0 * m * r_safe ** 2)
                                    + 0.5 * k * (r - L0) ** 2
                                    - m * g * r * np.cos(th))
    elif spec.name == "coupled_duffing_ham":
        k = 0.30
        x, y, px, py = [traj[:, i].astype(np.float64) for i in range(4)]
        known_series["energy_H"] = (0.5 * (px ** 2 + py ** 2)
                                    - 0.5 * (x ** 2 + y ** 2)
                                    + 0.25 * (x ** 4 + y ** 4)
                                    + 0.5 * k * (x - y) ** 2)

    # Score monomials by autocorrelation persistence
    if (lambda1 == lambda1) and (lambda1 > 1e-6):
        lyap_step = 1.0 / (lambda1 * dt)
        lyap_step_finite = True
    else:
        # non-chaotic or stochastic — fall back to a fixed lag schedule
        lyap_step = max(50.0, T / 40.0)
        lyap_step_finite = False
    eval_lags_raw = [int(round(lyap_step * f)) for f in (0.5, 1.0, 2.0, 4.0, 8.0)]
    eval_lags = [L for L in eval_lags_raw if 0 < L < T - 5]

    def _survival_time_in_lyap(series: np.ndarray) -> float:
        x = series - series[0]
        sigma = float(series.std())
        if sigma <= 0 or not np.isfinite(sigma):
            return float("inf")
        breach = np.where(np.abs(x) > sigma)[0]
        if breach.size == 0:
            return float("inf")
        t_breach = int(breach[0])
        return t_breach * dt * max(lambda1, 1e-9)

    for exps in basis:
        Q = _eval_monomial(traj, exps)
        if not np.isfinite(Q).all():
            continue
        if Q.std() < 1e-12:
            continue
        acs = _autocorrelation(Q, eval_lags) if eval_lags else []
        survival = _survival_time_in_lyap(Q)
        # Mean absolute autocorrelation across measured lags; high = flat
        if acs:
            mean_ac = float(np.nanmean([abs(a) for a in acs]))
        else:
            mean_ac = float("nan")
        candidates.append({
            "exponents": list(exps),
            "label": _exp_label(exps),
            "mean_abs_autocorrelation": mean_ac,
            "survival_lyap_times": survival,
            "long_run_std": float(Q.std()),
            "long_run_mean": float(Q.mean()),
        })

    candidates.sort(key=lambda c: (-c["mean_abs_autocorrelation"]
                                   if c["mean_abs_autocorrelation"] == c["mean_abs_autocorrelation"]
                                   else 1.0,
                                   -c["survival_lyap_times"]))
    top = candidates[:top_k]

    # Known invariant drift in Lyapunov time
    known_report = {}
    for name, series in known_series.items():
        rel_drift = float((series.max() - series.min()) / (abs(series.mean()) + 1e-9))
        survival = _survival_time_in_lyap(series)
        known_report[name] = {
            "relative_drift_minmax": rel_drift,
            "survival_lyap_times": survival,
            "mean": float(series.mean()),
            "std": float(series.std()),
        }

    return {
        "n_candidates": len(candidates),
        "top_quasi_invariants": top,
        "known_invariants": known_report,
        "eval_lags_steps": eval_lags,
        "lyapunov_step_count": lyap_step if lyap_step_finite else None,
    }


def _exp_label(exps: tuple) -> str:
    parts = []
    for i, e in enumerate(exps):
        if e == 0:
            continue
        v = f"x{i}"
        if e == 1:
            parts.append(v)
        else:
            parts.append(f"{v}^{e}")
    return "·".join(parts) if parts else "1"


# ─────────────────────────────────────────────────────────────────────────────
# (4) Symbolic-dynamics itinerary predictor
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans(X: np.ndarray, K: int, n_iter: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny k-means returning cluster assignments (labels of length N)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.choice(N, size=K, replace=False)
    centers = X[idx].astype(np.float64).copy()
    labels = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
    return labels


def itinerary_grammar(traj: np.ndarray, K: int = 6) -> dict:
    """Coarsen attractor → symbol sequence → first-order Markov.

    Returns block entropies, Markov transition matrix, next-symbol top-1
    accuracy, and the symbol-level Lempel-Ziv complexity."""
    if traj.shape[0] < 200:
        return {"K": K, "block_entropy_h1": float("nan"),
                "block_entropy_h2": float("nan"),
                "next_symbol_top1": float("nan"),
                "lz_complexity": float("nan"),
                "stationary": [], "transition_matrix": []}
    X = traj.astype(np.float64)
    if X.shape[1] >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
        proj_dim = int(min(4, Vh.shape[0]))
        proj = U[:, :proj_dim] * S[:proj_dim]
    else:
        # 1D — embed via delay coordinates so k-means sees structure
        v = X[:, 0]
        proj = np.stack([v[:-2], v[1:-1], v[2:]], axis=1)
        labels_pad = 2  # account for shorter length
        # We will pad symbol sequence at the end so the lengths match downstream

    labels = _kmeans(proj, K, seed=int(GLOBAL_SEED & 0xffff))
    # transitions
    src, dst = labels[:-1], labels[1:]
    P = np.zeros((K, K), dtype=np.float64)
    for s, d in zip(src, dst):
        P[s, d] += 1.0
    row = P.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = P / row
    # stationary distribution
    p = np.bincount(labels, minlength=K).astype(np.float64)
    p = p / p.sum()
    # block entropies
    h1 = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    pair_counts = np.zeros((K, K), dtype=np.float64)
    for s, d in zip(src, dst):
        pair_counts[s, d] += 1.0
    q = pair_counts / pair_counts.sum()
    h2_raw = float(-(q[q > 0] * np.log(q[q > 0])).sum())
    h2_per_symbol = h2_raw / 2.0
    # next-symbol Markov accuracy (greedy)
    pred = P.argmax(axis=1)
    top1 = float((pred[src] == dst).mean())
    # Lempel-Ziv complexity (LZ76)
    lz = _lz76(labels)
    return {
        "K": K,
        "block_entropy_h1_nats": h1,
        "block_entropy_h2_per_symbol_nats": h2_per_symbol,
        "conditional_entropy_h2_minus_h1": h2_per_symbol - h1 / 2.0,
        "next_symbol_top1": top1,
        "lz76_complexity": lz,
        "stationary": [float(v) for v in p.tolist()],
        "transition_matrix": [[float(v) for v in row.tolist()] for row in P],
    }


def _lz76(seq: np.ndarray) -> int:
    """Lempel-Ziv 1976 complexity over an integer symbol sequence."""
    s = "".join(str(int(x)) for x in seq.tolist())
    n = len(s)
    i, c, l, k = 0, 1, 1, 1
    k_max = 1
    while True:
        if i + k > n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


# ─────────────────────────────────────────────────────────────────────────────
# (5) Universality clustering
# ─────────────────────────────────────────────────────────────────────────────

FINGERPRINT_KEYS = [
    "lambda1",
    "local_lyap_std",
    "chaos_01_K",
    "corr_dim_d2",
    "spectral_entropy",
    "koopman_radius",
    "koopman_gap",
    "transfer_op_gap",
    "lnn_skill",
    "itinerary_top1",
    "block_entropy_h1",
    "lz76_complexity",
]

# Per-feature weights used when clustering. The three Lyapunov/0-1 features
# carry the cleanest discrimination signal between universality classes
# (Hamiltonian vs dissipative vs map vs stochastic), so they get a larger
# voice in the pairwise distance. Other features are kept at 1.0 — they
# still contribute to the cluster geometry, they just don't dominate.
FEATURE_WEIGHTS: dict[str, float] = {
    "lambda1": 2.0,
    "local_lyap_std": 2.5,
    "chaos_01_K": 3.0,
    "transfer_op_gap": 1.5,
    "lnn_skill": 0.5,   # noisy across regimes; downweight rather than drop
}


def _zscore(M: np.ndarray) -> np.ndarray:
    mu = np.nanmean(M, axis=0, keepdims=True)
    sd = np.nanstd(M, axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z = (M - mu) / sd
    Z = np.where(np.isnan(Z), 0.0, Z)
    return Z


def _pca_2d(Z: np.ndarray) -> np.ndarray:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Zc, full_matrices=False)
    return (U[:, :2] * S[:2])


def _hierarchical_clustering(D2: np.ndarray, n_clusters: int,
                             linkage: str = "ward",
                             initial_sizes: np.ndarray | None = None
                             ) -> tuple[np.ndarray, list[dict]]:
    """Bottom-up agglomerative clustering.

    Parameters
    ----------
    D2 : (n, n)  matrix of *squared* Euclidean distances in weighted feature space.
    linkage : "ward" | "complete" | "average".
    initial_sizes : optional cluster sizes (defaults to ones).

    Returns the cluster-id assignment of length n and the full merge sequence
    (list of {step, a, b, distance, size, members}) — enough to draw a
    dendrogram.
    """
    n = D2.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    sizes = (initial_sizes.copy().astype(np.float64)
             if initial_sizes is not None else np.ones(n, dtype=np.float64))
    # For Ward we use the Lance–Williams recurrence on *squared* distances.
    dist = D2.copy().astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    merges: list[dict] = []
    step = 0
    while dist.shape[0] > n_clusters:
        flat = int(np.argmin(dist))
        i, j = divmod(flat, dist.shape[0])
        if i > j:
            i, j = j, i
        d_ij = float(dist[i, j])
        ni, nj = sizes[i], sizes[j]
        # build new row using the linkage rule
        new_row = np.empty(dist.shape[0])
        for k in range(dist.shape[0]):
            if k == i or k == j:
                continue
            nk = sizes[k]
            d_ik, d_jk = dist[i, k], dist[j, k]
            if linkage == "ward":
                total = ni + nj + nk
                new_row[k] = ((ni + nk) * d_ik + (nj + nk) * d_jk
                              - nk * d_ij) / total
            elif linkage == "complete":
                new_row[k] = max(d_ik, d_jk)
            else:  # average
                new_row[k] = (ni * d_ik + nj * d_jk) / (ni + nj)
        new_row[i] = np.inf
        dist[i] = new_row
        dist[:, i] = new_row
        sizes[i] = ni + nj
        # capture the merge for the dendrogram
        merges.append({
            "step": step,
            "a_members": list(clusters[i]),
            "b_members": list(clusters[j]),
            "distance": math.sqrt(max(d_ij, 0.0)),
            "size_after": int(ni + nj),
        })
        step += 1
        clusters[i] = clusters[i] + clusters[j]
        # drop j
        dist = np.delete(dist, j, axis=0)
        dist = np.delete(dist, j, axis=1)
        sizes = np.delete(sizes, j)
        clusters.pop(j)
        np.fill_diagonal(dist, np.inf)
    labels = np.zeros(n, dtype=np.int32)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels, merges


def _weighted_zscore(M: np.ndarray, keys: list[str]) -> np.ndarray:
    """Per-feature z-score then multiply each column by its FEATURE_WEIGHTS
    coefficient. NaNs are filled with the column mean before z-scoring; columns
    with zero std become zero columns and contribute nothing."""
    weights = np.array([FEATURE_WEIGHTS.get(k, 1.0) for k in keys], dtype=np.float64)
    Z = M.copy().astype(np.float64)
    # column-wise fillna with column mean
    for j in range(Z.shape[1]):
        col = Z[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            Z[:, j] = 0.0
            continue
        mu = float(col[mask].mean())
        col = np.where(mask, col, mu)
        sd = float(col.std())
        Z[:, j] = (col - mu) / sd if sd > 1e-8 else 0.0
    return Z * weights[None, :]


def _cluster_characterizations(per_system: list[dict], labels: np.ndarray,
                               keys: list[str]) -> list[dict]:
    """For each cluster, compute mean fingerprint, most-distinctive feature,
    and a one-line semantic label like "Hamiltonian chaos (K≈0)"."""
    out: list[dict] = []
    M = np.array(
        [[s["fingerprints"].get(k, float("nan")) for k in keys]
         for s in per_system],
        dtype=np.float64,
    )
    # global means / stds for the distinctiveness measure
    global_mu = np.nanmean(M, axis=0)
    global_sd = np.nanstd(M, axis=0)
    global_sd = np.where(global_sd > 1e-8, global_sd, 1.0)
    for cid in sorted({int(c) for c in labels}):
        idx = [i for i, c in enumerate(labels) if int(c) == cid]
        members = [per_system[i]["name"] for i in idx]
        if not members:
            continue
        cluster_M = M[idx]
        cluster_mu = np.nanmean(cluster_M, axis=0)
        # how far does this cluster's mean deviate from the global mean,
        # in global-z units
        deviation = (cluster_mu - global_mu) / global_sd
        order = np.argsort(-np.abs(deviation))
        ranked = [
            {"feature": keys[k], "cluster_mean": float(cluster_mu[k]),
             "global_mean": float(global_mu[k]),
             "z_distance": float(deviation[k])}
            for k in order[:5]
            if np.isfinite(deviation[k])
        ]
        # semantic label
        K = cluster_mu[keys.index("chaos_01_K")] if "chaos_01_K" in keys else float("nan")
        lam = cluster_mu[keys.index("lambda1")] if "lambda1" in keys else float("nan")
        loc_std = (cluster_mu[keys.index("local_lyap_std")]
                   if "local_lyap_std" in keys else float("nan"))
        # family majority for context
        family_counts: dict[str, int] = {}
        for i in idx:
            fam = per_system[i].get("family", "?")
            family_counts[fam] = family_counts.get(fam, 0) + 1
        majority_family = max(family_counts.items(), key=lambda kv: kv[1])[0]
        # heuristic semantic label
        if np.isfinite(K) and np.isfinite(lam):
            if K < 0.2 and lam > 0.05 and majority_family == "flow_hamiltonian":
                label = "Hamiltonian chaos (energy-bounded, K≈0)"
            elif K > 0.7 and lam > 0.3 and majority_family == "flow_dissipative":
                label = "Dissipative chaos (K≈1, fast Lyapunov)"
            elif K > 0.7 and lam > 0.3 and majority_family == "map":
                label = "Discrete-map chaos (K≈1)"
            elif K < 0.2 and lam < 0.05:
                label = "Regular / quasi-periodic (K≈0, λ≈0)"
            elif majority_family == "control" and abs(lam) < 0.05:
                label = "Non-chaotic control"
            elif loc_std > 1.0:
                label = "Hyper-spread Lyapunov (large local-λ σ)"
            else:
                label = "Mixed / ambiguous cluster"
        else:
            label = "Singleton or undefined"
        out.append({
            "cluster_id": int(cid),
            "size": len(members),
            "members": members,
            "majority_family": majority_family,
            "family_counts": family_counts,
            "semantic_label": label,
            "cluster_mean": {k: float(v) for k, v in zip(keys, cluster_mu)},
            "most_distinctive_features": ranked,
        })
    return out


def universality_clustering(per_system: list[dict], n_clusters: int = 6,
                            linkage: str = "ward") -> dict:
    """Weighted-feature hierarchical clustering on the chaos fingerprint."""
    M = np.array(
        [[s["fingerprints"].get(k, float("nan")) for k in FINGERPRINT_KEYS]
         for s in per_system],
        dtype=np.float64,
    )
    Z = _weighted_zscore(M, FINGERPRINT_KEYS)
    coords = _pca_2d(Z)
    diff = Z[:, None, :] - Z[None, :, :]
    D2 = (diff * diff).sum(axis=-1)
    labels, merges = _hierarchical_clustering(D2, n_clusters=n_clusters,
                                              linkage=linkage)
    clusters: dict[int, list[str]] = {}
    for s, c in zip(per_system, labels):
        clusters.setdefault(int(c), []).append(s["name"])
    chars = _cluster_characterizations(per_system, labels, FINGERPRINT_KEYS)

    # also record a dendrogram for downstream rendering
    D = np.sqrt(np.maximum(D2, 0.0))
    return {
        "n_clusters": int(n_clusters),
        "linkage": linkage,
        "feature_keys": FINGERPRINT_KEYS,
        "feature_weights": {k: float(FEATURE_WEIGHTS.get(k, 1.0))
                            for k in FINGERPRINT_KEYS},
        "z_matrix_weighted": Z.tolist(),
        "raw_matrix": M.tolist(),
        "pca_coords": coords.tolist(),
        "cluster_labels": [int(l) for l in labels.tolist()],
        "cluster_members": {str(c): names for c, names in clusters.items()},
        "characterizations": chars,
        "dendrogram_merges": merges,
        "pairwise_distance_max": float(np.nanmax(D)),
        "pairwise_distance_mean": float(np.nanmean(D[D > 0])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# (6) Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabConfig:
    n_steps_flow: int = 4000
    n_steps_map: int = 6000
    n_clusters: int = 6
    linkage: str = "ward"
    foundation_train_steps: int = 250
    foundation_hidden: int = 32
    itinerary_K: int = 6
    quasi_max_degree: int = 3


def fingerprint_system(spec: SystemSpec, cfg: LabConfig, log) -> dict:
    log(f"  · integrating {spec.name} ({spec.family}, dim={spec.dim})")
    n = cfg.n_steps_map if spec.family == "map" else cfg.n_steps_flow
    traj = spec.integrate(spec.base_ic, n, spec.dt)

    log(f"    Lyapunov...")
    t0 = time.time()
    if spec.family == "map":
        lam1 = lyapunov_map(spec)
    elif spec.family == "control" and spec.name == "harmonic":
        lam1 = 0.0
    elif spec.family == "control" and spec.name == "quasiperiodic_2tori":
        lam1 = 0.0
    elif spec.family == "control" and spec.name == "random_walk":
        lam1 = float("nan")  # stochastic, not a chaos exponent
    else:
        lam1 = lyapunov_continuous(spec)
    lyap_time = time.time() - t0

    log(f"    local-Lyapunov distribution...")
    if spec.family == "map":
        lloc = local_lyapunov_distribution_map(spec)
    elif _continuous_vector_field(spec.name) is not None:
        lloc = local_lyapunov_distribution(spec)
    else:
        # controls without a continuous f — use trajectory-based fallback:
        # the std of log-step-stretching of a 1D-projected observable
        x = traj[:, 0].astype(np.float64) if traj.ndim == 2 else traj.astype(np.float64)
        if x.size > 50 and x.std() > 0:
            diffs = np.abs(np.diff(x)) + 1e-12
            chunks = np.array_split(np.log(diffs), 40)
            locs = np.array([float(c.mean()) for c in chunks if c.size > 0])
            lloc = {"mean": float(locs.mean()), "std": float(locs.std()),
                    "p10": float(np.quantile(locs, 0.10)),
                    "p50": float(np.quantile(locs, 0.50)),
                    "p90": float(np.quantile(locs, 0.90)),
                    "range": float(locs.max() - locs.min()),
                    "n_windows": int(locs.size), "method": "fallback_diff"}
        else:
            lloc = {"mean": float("nan"), "std": float("nan"),
                    "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                    "range": float("nan"), "n_windows": 0, "method": "skipped"}

    log(f"    0-1 chaos test...")
    n_k = k_trajectory_length(spec.family, n)
    if n_k != n:
        traj_k = spec.integrate(spec.base_ic, n_k, spec.dt)
    else:
        traj_k = traj
    gm01 = gottwald_melbourne_01_test(traj_k)
    gm01["k_trajectory_steps"] = int(n_k)

    log(f"    phase-volume contraction (divergence)...")
    pvc = phase_volume_contraction(spec, traj)

    log(f"    correlation dimension...")
    d2 = correlation_dimension(traj)

    log(f"    spectral entropy + DMD...")
    H_psd = spectral_entropy(traj)
    dmd = hankel_dmd(traj)

    log(f"    transfer-operator gap...")
    tog = transfer_operator_gap(traj)

    log(f"    Foundation predictor...")
    fp = foundation_predictability(traj,
                                   hidden=cfg.foundation_hidden,
                                   train_steps=cfg.foundation_train_steps,
                                   dt=1.0)

    log(f"    quasi-invariants...")
    qi = hunt_quasi_invariants(spec, traj, lam1,
                               max_degree=cfg.quasi_max_degree)

    log(f"    itinerary grammar...")
    itin = itinerary_grammar(traj, K=cfg.itinerary_K)

    fingerprints = {
        "lambda1": lam1,
        "local_lyap_std": lloc.get("std", float("nan")),
        "chaos_01_K": gm01.get("K_median", float("nan")),
        "corr_dim_d2": d2,
        "spectral_entropy": H_psd,
        "koopman_radius": dmd["radius"],
        "koopman_gap": dmd["gap"],
        "transfer_op_gap": tog["gap"],
        "lnn_skill": fp["skill"],
        "itinerary_top1": itin["next_symbol_top1"],
        "block_entropy_h1": itin["block_entropy_h1_nats"],
        "lz76_complexity": float(itin["lz76_complexity"]),
    }

    return {
        "name": spec.name,
        "family": spec.family,
        "dim": spec.dim,
        "dt": spec.dt,
        "n_steps": int(n),
        "description": spec.description,
        "trajectory_stats": {
            "mean_per_channel": [float(v) for v in traj.mean(axis=0).tolist()],
            "std_per_channel": [float(v) for v in traj.std(axis=0).tolist()],
            "abs_max_per_channel": [float(v) for v in np.abs(traj).max(axis=0).tolist()],
        },
        "fingerprints": fingerprints,
        "dmd": dmd,
        "transfer_operator": tog,
        "foundation_predictor": fp,
        "quasi_invariants": qi,
        "itinerary": itin,
        "local_lyapunov": lloc,
        "gottwald_melbourne_01": gm01,
        "phase_volume_contraction": pvc,
        "lyapunov_time_seconds": lyap_time,
    }


def run_lab(cfg: LabConfig | None = None, verbose: bool = True) -> dict:
    cfg = cfg or LabConfig()
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    log("Chaos Universality Lab — start")
    t_global = time.time()
    specs = build_zoo()
    per_system: list[dict] = []
    for spec in specs:
        t_sys = time.time()
        try:
            entry = fingerprint_system(spec, cfg, log)
        except Exception as exc:  # report and continue
            entry = {"name": spec.name, "family": spec.family,
                     "error": f"{type(exc).__name__}: {exc}",
                     "fingerprints": {k: float("nan") for k in FINGERPRINT_KEYS}}
            log(f"    !! {spec.name} failed: {exc}")
        entry["wall_seconds"] = round(time.time() - t_sys, 3)
        per_system.append(entry)
        log(f"  ✓ {spec.name} done in {entry['wall_seconds']}s")

    log("Universality clustering...")
    universality = universality_clustering(per_system,
                                            n_clusters=cfg.n_clusters,
                                            linkage=cfg.linkage)

    # Aggregate summary
    chaotic = [s for s in per_system if s["family"] in ("flow_dissipative",
                                                         "flow_hamiltonian",
                                                         "map")]
    controls = [s for s in per_system if s["family"] == "control"]
    chaotic_lams = [s["fingerprints"]["lambda1"] for s in chaotic
                    if np.isfinite(s["fingerprints"]["lambda1"])]
    control_lams = [s["fingerprints"]["lambda1"] for s in controls
                    if np.isfinite(s["fingerprints"]["lambda1"])]
    chaotic_top1 = [s["fingerprints"]["itinerary_top1"] for s in chaotic
                    if np.isfinite(s["fingerprints"]["itinerary_top1"])]
    control_top1 = [s["fingerprints"]["itinerary_top1"] for s in controls
                    if np.isfinite(s["fingerprints"]["itinerary_top1"])]

    longest_quasi = []
    for s in per_system:
        top = s.get("quasi_invariants", {}).get("top_quasi_invariants", [])
        if top:
            best = top[0]
            longest_quasi.append({
                "system": s["name"],
                "label": best.get("label"),
                "survival_lyap_times": best.get("survival_lyap_times"),
                "mean_abs_autocorrelation": best.get("mean_abs_autocorrelation"),
            })
    longest_quasi.sort(key=lambda r: (
        -(r["survival_lyap_times"] if r["survival_lyap_times"] != float("inf") else 1e12)))

    summary = {
        "n_systems": len(per_system),
        "n_chaotic": len(chaotic),
        "n_controls": len(controls),
        "mean_lambda1_chaotic": float(np.mean(chaotic_lams)) if chaotic_lams else None,
        "max_lambda1_chaotic": float(np.max(chaotic_lams)) if chaotic_lams else None,
        "mean_lambda1_controls": float(np.mean(control_lams)) if control_lams else None,
        "mean_itinerary_top1_chaotic": float(np.mean(chaotic_top1)) if chaotic_top1 else None,
        "mean_itinerary_top1_controls": float(np.mean(control_top1)) if control_top1 else None,
        "itinerary_advantage": (
            float(np.mean(chaotic_top1) - np.mean(control_top1))
            if chaotic_top1 and control_top1 else None
        ),
        "longest_quasi_invariants": longest_quasi[:8],
        "universality_clusters": universality["cluster_members"],
    }

    result = {
        "version": "chaos_universality_lab/0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": GLOBAL_SEED,
        "config": {
            "n_steps_flow": cfg.n_steps_flow,
            "n_steps_map": cfg.n_steps_map,
            "n_clusters": cfg.n_clusters,
            "linkage": cfg.linkage,
            "foundation_train_steps": cfg.foundation_train_steps,
            "foundation_hidden": cfg.foundation_hidden,
            "itinerary_K": cfg.itinerary_K,
            "quasi_max_degree": cfg.quasi_max_degree,
        },
        "summary": summary,
        "universality": universality,
        "systems": per_system,
        "wall_seconds_total": round(time.time() - t_global, 3),
    }
    return result


def main():
    cfg = LabConfig()
    result = run_lab(cfg, verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"Wall time: {result['wall_seconds_total']:.1f}s")
    print(f"Systems: {result['summary']['n_systems']} "
          f"({result['summary']['n_chaotic']} chaotic, "
          f"{result['summary']['n_controls']} controls)")
    print(f"Universality clusters: {result['universality']['cluster_members']}")


if __name__ == "__main__":
    main()
