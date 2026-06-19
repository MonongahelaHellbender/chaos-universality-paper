"""
chaos_observable_selection.py
=============================
Idea B: Which coordinate converges K fastest, and does the integrated
autocorrelation time τ_int (Green-Kubo proxy) predict the ordering?

For each multi-dimensional chaotic system: compute K using each state
coordinate separately and compute τ_int = decorrelation lag (proxy for
integrated autocorrelation time).  Verify: shorter τ_int → higher K
at finite trajectory length (faster convergence).

Output: results/chaos_observable_selection.json
"""
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260619
GLOBAL_SEED = 20260528
N_WARMUP   = 0      # match lab (fingerprint_system uses no warmup, runs from IC)
N_ANALYSIS = 4000   # match Table 1 primary runs
N_C        = 12
TARGET_SAMPLES = 1500
N_LENGTHS  = [1000, 2000, 4000, 8000, 16000]  # for convergence curve

# ── integrators ──────────────────────────────────────────────────────────────

def _rk4(f, x, dt):
    k1 = f(x);  k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _flow(f, x0, n, dt):
    out = np.empty((n, x0.size), dtype=np.float64)
    x = x0.astype(np.float64).copy()
    for i in range(n):
        out[i] = x
        x = _rk4(f, x, dt)
    return out

# ── vector fields ─────────────────────────────────────────────────────────────

def _lorenz(x, sigma=10., rho=28., beta=8./3.):
    return np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]])

def _rossler(x, a=0.2, b=0.2, c=5.7):
    return np.array([-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)])

def _chen(x, a=35., b=3., c=28.):
    return np.array([a*(x[1]-x[0]), (c-a)*x[0]-x[0]*x[2]+c*x[1], x[0]*x[1]-b*x[2]])

def _dp(s, g=9.81):
    th1,th2,p1,p2 = s
    d=th1-th2; c=math.cos(d); sn=math.sin(d); D=2.-c*c
    th1d=(p1-p2*c)/D; th2d=(2.*p2-p1*c)/D
    N=p1*p1-2.*p1*p2*c+2.*p2*p2; dN=2.*p1*p2*sn; dD=2.*c*sn
    dT=.5*(dN*D-N*dD)/(D*D)
    return np.array([th1d, th2d, -(dT+2.*g*math.sin(th1)), -(-dT+g*math.sin(th2))])

def _halvorsen(x, a=1.4):
    return np.array([-a*x[0]-4*x[1]-4*x[2]-x[1]**2,
                     -a*x[1]-4*x[2]-4*x[0]-x[2]**2,
                     -a*x[2]-4*x[0]-4*x[1]-x[0]**2])

SYSTEMS = [
    # (name, label, f, x0, dt, coord_names)
    ('lorenz63', 'Lorenz 63',
     _lorenz,  [1.,1.,1.],  0.01,  ['x', 'y', 'z']),
    ('rossler', 'Rössler',
     _rossler, [1.,1.,0.],  0.05,  ['x', 'y', 'z']),
    ('chen',    'Chen',
     _chen,    [-10.,0.,37.], 0.005, ['x', 'y', 'z']),
    ('halvorsen', 'Halvorsen',
     _halvorsen, [-5.,0.,0.], 0.01,  ['x', 'y', 'z']),
    ('double_pendulum', 'Double pendulum',
     _dp, [math.pi/2., math.pi/2.+0.01, 0., 0.], 0.005,
     ['θ₁', 'θ₂', 'p₁', 'p₂']),
]

# ── 0-1 implementation ────────────────────────────────────────────────────────

def _decorr_lag(x, target=0.2, max_lag=500):
    x = x - x.mean(); var = float(np.mean(x*x))
    if var <= 0: return 1
    upper = min(max_lag, len(x)-2)
    for lag in range(1, upper+1):
        ac = float(np.mean(x[:-lag]*x[lag:]) / var)
        if abs(ac) < target: return lag
    return max(upper, 1)

def _tau_int(x, max_lag=500):
    """Windowed integrated autocorrelation time (Sokal estimator)."""
    x = x - x.mean(); var = float(np.mean(x*x))
    if var <= 0: return 1.
    upper = min(max_lag, len(x)-2)
    tau = 0.5
    for lag in range(1, upper+1):
        ac = float(np.mean(x[:-lag]*x[lag:]) / var)
        tau += (1. - lag/upper) * ac
        if lag >= 4*tau:  # Sokal window criterion
            break
    return max(tau, 0.5)

def _gm_K(phi_full, c_vals, target_samples=1500):
    decorr = _decorr_lag(phi_full)
    min_stride = max(1, len(phi_full)//target_samples)
    stride = max(min_stride, min(decorr, len(phi_full)//200))
    phi = phi_full[::stride] - phi_full[::stride].mean()
    N = len(phi)
    if N < 50: return float('nan'), stride
    n_max = max(10, N//10)
    j = np.arange(1, N+1, dtype=np.float64)
    Ks = []
    for c in c_vals:
        p = np.cumsum(phi * np.cos(j*c))
        q = np.cumsum(phi * np.sin(j*c))
        lags = np.arange(1, n_max+1)
        Mc = np.array([np.mean((p[l:]-p[:-l])**2 + (q[l:]-q[:-l])**2)
                       for l in lags])
        valid = Mc > 0
        if valid.sum() >= 3:
            r = np.corrcoef(np.log(lags[valid]), np.log(Mc[valid]))[0,1]
            Ks.append(float(r))
    return (float(np.median(Ks)) if Ks else float('nan')), stride

def main():
    rng_c = np.random.default_rng(31)
    c_vals = rng_c.uniform(math.pi/5., 4.*math.pi/5., size=N_C)

    out_systems = []
    t0 = time.time()
    print(f"Observable selection analysis  (N={N_ANALYSIS} steps, seed {SEED})\n")

    for sname, slabel, f, x0_list, dt, coord_names in SYSTEMS:
        x0 = np.array(x0_list, dtype=np.float64)
        n_total = max(N_LENGTHS)
        traj_full = _flow(f, x0, n_total, dt)

        print(f"  {slabel}")
        coords_out = []
        tau_ints = []
        K_finals = []

        for ci, cname in enumerate(coord_names):
            phi_full = traj_full[:max(N_LENGTHS), ci]
            tau = _tau_int(phi_full)
            tau_ints.append(tau)

            # K convergence at multiple lengths
            K_curve = []
            for nlen in N_LENGTHS:
                phi_seg = phi_full[:nlen]
                K, stride = _gm_K(phi_seg, c_vals, TARGET_SAMPLES)
                K_curve.append({'n': nlen, 'K': round(K,4), 'stride': stride})

            K_final = K_curve[-1]['K']
            K_finals.append(K_final)
            print(f"    coord {ci} ({cname:4s}): τ_int={tau:6.1f}  K_final={K_final:+.3f}")
            coords_out.append({
                'index': ci, 'name': cname,
                'tau_int': round(tau, 2),
                'decorr_lag': _decorr_lag(phi_full),
                'K_final': K_final,
                'K_curve': K_curve,
            })

        # rank by τ_int (ascending = best coord) vs K_final (descending = best)
        tau_rank  = np.argsort(tau_ints)          # 0 = smallest τ_int
        K_rank    = np.argsort(K_finals)[::-1]    # 0 = highest K
        best_tau  = int(tau_rank[0])
        best_K    = int(K_rank[0])
        match     = best_tau == best_K
        print(f"    Best τ_int: coord {best_tau} ({coord_names[best_tau]})  "
              f"Best K: coord {best_K} ({coord_names[best_K]})  match={match}\n")

        # Spearman between τ_int and K_final (negative correlation expected)
        if len(tau_ints) >= 3:
            rs, pval = spearmanr(tau_ints, K_finals)
        else:
            rs, pval = float('nan'), float('nan')

        out_systems.append({
            'name': sname, 'label': slabel,
            'n_coords': len(coord_names),
            'best_coord_by_tau': best_tau,
            'best_coord_by_K': best_K,
            'tau_K_spearman_rs': round(float(rs), 3),
            'tau_K_spearman_p': round(float(pval), 4),
            'coords': coords_out,
        })

    # pooled Spearman across all system×coord pairs
    all_tau = [c['tau_int'] for s in out_systems for c in s['coords']]
    all_K   = [c['K_final'] for s in out_systems for c in s['coords']
               if not math.isnan(c['K_final'])]
    all_tau_f = [c['tau_int'] for s in out_systems for c in s['coords']
                 if not math.isnan(c['K_final'])]
    rs_pool, p_pool = spearmanr(all_tau_f, all_K)
    n_matches = sum(1 for s in out_systems
                    if s['best_coord_by_tau'] == s['best_coord_by_K'])
    n_systems = len(out_systems)

    print(f"Pooled Spearman (τ_int vs K_final): rs={rs_pool:.3f}  p={p_pool:.4f}")
    print(f"Best-coord match: {n_matches}/{n_systems} systems")

    result = {
        'version': 'chaos_observable_selection/0.1.0',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED,
        'config': {
            'n_warmup': N_WARMUP,
            'n_analysis': N_ANALYSIS,
            'n_c': N_C,
            'lengths': N_LENGTHS,
            'target_samples': TARGET_SAMPLES,
        },
        'pooled_spearman_rs': round(float(rs_pool), 3),
        'pooled_spearman_p':  round(float(p_pool), 4),
        'n_coord_matches': n_matches,
        'n_systems': n_systems,
        'systems': out_systems,
        'wall_seconds': round(time.time()-t0, 1),
    }

    out_path = Path(__file__).parent.parent / 'results' / 'chaos_observable_selection.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}  ({time.time()-t0:.1f}s total)")

if __name__ == '__main__':
    main()
