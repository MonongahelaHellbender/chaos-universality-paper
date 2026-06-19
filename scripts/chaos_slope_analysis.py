"""
chaos_slope_analysis.py
=======================
Compute the power-law exponent α of M_c(n) for all 26 benchmark systems.

For each system:
  M_c(n) ~ n^α
  α ≈ 0  → regular / bounded
  α ≈ 1  → chaotic diffusive (correct asymptotic for chaos)
  α ≈ 2  → quasiperiodic / ballistic (near-resonant orbit)

The log-log correlation K_c = corr(log n, log M_c) is ~1 for both
chaotic (α≈1) and quasiperiodic (α≈2) systems. The slope α distinguishes them.

Output: results/chaos_slope_analysis.json
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

SEED = 20260619
GLOBAL_SEED = 20260528   # matches main lab for random walk
N_WARMUP   = 3000        # discarded
N_ANALYSIS = 12000       # raw steps, decimated before M_c computation
N_C        = 12          # number of random frequencies (matches paper)
TARGET_SAMPLES = 1500    # target number of samples after decimation (matches lab)
N_LAG_GRID = 30          # number of lag points for slope fit
SHORT_LAG_FRAC = 0.25    # fit α_short over lags ≤ this fraction of n_max
LONG_LAG_FRAC  = 0.50    # fit α_long  over lags ≥ this fraction of n_max

# ── integrators ────────────────────────────────────────────────────────────

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

# ── system vector fields (same params as main lab) ────────────────────────

def _f_lorenz(x, sigma=10., rho=28., beta=8./3.):
    return np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]])

def _f_rossler(x, a=0.2, b=0.2, c=5.7):
    return np.array([-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)])

def _f_duffing(s, alpha=-1., beta=1., delta=0.3, gamma=0.5, omega=1.2):
    x,v,ph = s
    return np.array([v, -delta*v - alpha*x - beta*x**3 + gamma*math.cos(ph), omega])

def _f_dp(s, g=9.81):
    th1,th2,p1,p2 = s
    d=th1-th2; c=math.cos(d); sn=math.sin(d); D=2.-c*c
    th1d=(p1-p2*c)/D; th2d=(2.*p2-p1*c)/D
    N=p1*p1-2.*p1*p2*c+2.*p2*p2
    dN=2.*p1*p2*sn; dD=2.*c*sn
    dT=.5*(dN*D-N*dD)/(D*D)
    return np.array([th1d, th2d, -(dT+2.*g*math.sin(th1)), -(-dT+g*math.sin(th2))])

def _f_hh(s):
    x,y,px,py = s
    return np.array([px, py, -x-2.*x*y, -y-(x*x-y*y)])

def _f_ym(s):
    x,y,px,py = s
    return np.array([px, py, -x*y*y, -x*x*y])

def _f_pe(s, alpha=0.5):
    x,y,px,py = s
    return np.array([px, py, -x-2.*alpha*x*y*y, -y-2.*alpha*x*x*y])

def _f_qc(s, b=0.6):
    x,y,px,py = s
    return np.array([px, py, -x**3-b*x*y*y, -y**3-b*x*x*y])

def _f_sp(s, m=1., k=39.5, L0=1., g=9.81):
    r,th,pr,pth = s
    rs = r if abs(r)>1e-6 else 1e-6
    return np.array([pr/m, pth/(m*rs*rs),
                     pth*pth/(m*rs**3)-k*(r-L0)+m*g*math.cos(th),
                     -m*g*r*math.sin(th)])

def _f_cdh(s, k=0.30):
    x,y,px,py = s
    return np.array([px, py, x-x**3-k*(x-y), y-y**3+k*(x-y)])

def _f_cr3bp(s, mu=0.01215):
    x,y,vx,vy = s
    r1=math.sqrt((x+mu)**2+y*y)+1e-9; r2=math.sqrt((x-1.+mu)**2+y*y)+1e-9
    ax=2.*vy+x-(1.-mu)*(x+mu)/r1**3-mu*(x-1.+mu)/r2**3
    ay=-2.*vx+y-(1.-mu)*y/r1**3-mu*y/r2**3
    return np.array([vx,vy,ax,ay])

def _f_l96(x, F=8.):
    N=x.size; out=np.empty(N)
    for i in range(N): out[i]=(x[(i+1)%N]-x[i-2])*x[i-1]-x[i]+F
    return out

def _f_aizawa(s, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f_=0.1):
    x,y,z = s
    return np.array([(z-b)*x-d*y, d*x+(z-b)*y,
                     c+a*z-z**3/3.-(x*x+y*y)*(1.+e*z)+f_*z*x**3])

def _f_sprb(s):
    x,y,z = s
    return np.array([y*z, x-y, 1.-x*y])

def _f_halv(s, a=1.4):
    x,y,z = s
    return np.array([-a*x-4.*y-4.*z-y*y, -a*y-4.*z-4.*x-z*z, -a*z-4.*x-4.*y-x*x])

def _f_thomas(s, b=0.208186):
    x,y,z = s
    return np.array([math.sin(y)-b*x, math.sin(z)-b*y, math.sin(x)-b*z])

def _f_chen(s, a=35., b=3., c=28.):
    x,y,z = s
    return np.array([a*(y-x), (c-a)*x-x*z+c*y, x*y-b*z])

def _f_ueda(s, k=0.05, B=7.5, omega=1.):
    x,v,ph = s
    return np.array([v, -k*v-x**3+B*math.cos(ph), omega])

# ── trajectory generators ──────────────────────────────────────────────────

def gen(name, x0, n_warm, n_anal, dt):
    """Generate (n_warm + n_anal) steps, discard warmup, return analysis."""
    x0 = np.array(x0, dtype=np.float64)
    total = n_warm + n_anal
    if name == 'lorenz63':
        traj = _flow(_f_lorenz, x0, total, dt)
    elif name == 'rossler':
        traj = _flow(_f_rossler, x0, total, dt)
    elif name == 'duffing_forced':
        traj = _flow(_f_duffing, x0, total, dt)
    elif name == 'double_pendulum':
        traj = _flow(_f_dp, x0, total, dt)
    elif name == 'henon_heiles':
        traj = _flow(_f_hh, x0, total, dt)
    elif name == 'yang_mills_x2y2':
        traj = _flow(_f_ym, x0, total, dt)
    elif name == 'pullen_edmonds':
        traj = _flow(_f_pe, x0, total, dt)
    elif name == 'quartic_coupled':
        traj = _flow(_f_qc, x0, total, dt)
    elif name == 'spring_pendulum':
        traj = _flow(_f_sp, x0, total, dt)
    elif name == 'coupled_duffing_ham':
        traj = _flow(_f_cdh, x0, total, dt)
    elif name == 'cr3bp_earth_moon':
        traj = _flow(_f_cr3bp, x0, total, dt)
    elif name in ('lorenz96_n5', 'lorenz96_n10'):
        traj = _flow(_f_l96, x0, total, dt)
    elif name == 'aizawa':
        traj = _flow(_f_aizawa, x0, total, dt)
    elif name == 'sprott_b':
        traj = _flow(_f_sprb, x0, total, dt)
    elif name == 'halvorsen':
        traj = _flow(_f_halv, x0, total, dt)
    elif name == 'thomas':
        traj = _flow(_f_thomas, x0, total, dt)
    elif name == 'chen':
        traj = _flow(_f_chen, x0, total, dt)
    elif name == 'ueda_forced':
        traj = _flow(_f_ueda, x0, total, dt)
    elif name == 'mackey_glass':
        beta,gamma,n_exp,tau = 0.2, 0.1, 10, 17.0
        delay_steps = max(2, int(round(tau/dt)))
        hist = np.full(delay_steps+1, float(x0[0]))
        head = 0; out = np.empty((total, 1)); xv = float(x0[0])
        for i in range(total):
            out[i,0] = xv; xtau = hist[(head-delay_steps)%hist.size]
            dx = beta*xtau/(1.+xtau**n_exp)-gamma*xv; xv += dt*dx
            head=(head+1)%hist.size; hist[head]=xv
        traj = out
    elif name == 'logistic_r3p9':
        out=np.empty((total,1)); xv=float(x0[0])
        for i in range(total): out[i,0]=xv; xv=3.9*xv*(1.-xv)
        traj = out
    elif name == 'henon_map':
        out=np.empty((total,2)); xv,yv=float(x0[0]),float(x0[1])
        for i in range(total):
            out[i,0]=xv; out[i,1]=yv; xn=1.-1.4*xv*xv+yv; yn=0.3*xv; xv,yv=xn,yn
        traj = out
    elif name == 'standard_map_K1p2':
        out=np.empty((total,2)); pv,th=float(x0[0]),float(x0[1]); tp=2.*math.pi
        for i in range(total):
            out[i,0]=pv; out[i,1]=th; pv=(pv+1.2*math.sin(th))%tp; th=(th+pv)%tp
        traj = out
    elif name == 'harmonic':
        t=np.arange(total)*dt; A=math.sqrt(x0[0]**2+(x0[1]/1.0)**2)
        ph=math.atan2(-x0[1]/1.0,x0[0])
        traj=np.column_stack([A*np.cos(t+ph),-A*np.sin(t+ph)])
    elif name == 'quasiperiodic_2tori':
        t=np.arange(total)*dt
        traj=np.column_stack([np.cos(t)+np.cos(np.sqrt(2.)*t),
                              np.sin(t)+np.sin(np.sqrt(2.)*t)])
    elif name == 'random_walk':
        rng=np.random.default_rng(int(x0[0]*1e6)^GLOBAL_SEED)
        steps=rng.standard_normal(size=(total,2))*0.1
        traj=np.cumsum(steps,axis=0)
    else:
        raise ValueError(f"Unknown system: {name}")
    return traj[n_warm:]   # discard warmup

# ── decimation (exactly as in the lab's 0-1 implementation) ───────────────

def first_decorr_lag(x, target=0.2, max_lag=500):
    """Smallest lag where |autocorr(x)| < target."""
    x = x - x.mean()
    var = float(np.mean(x*x))
    if var <= 0:
        return 1
    upper = min(max_lag, len(x)-2)
    for lag in range(1, upper+1):
        ac = float(np.mean(x[:-lag]*x[lag:]) / var)
        if abs(ac) < target:
            return lag
    return max(upper, 1)

def decimate(phi_full, target_samples=1500):
    """Decimate phi to ~target_samples at the decorrelation stride."""
    decorr = first_decorr_lag(phi_full)
    min_stride = max(1, len(phi_full) // target_samples)
    stride = max(min_stride, min(decorr, len(phi_full) // 200))
    stride = max(stride, 1)
    return phi_full[::stride], int(stride)

# ── M_c(n) computation ─────────────────────────────────────────────────────

def compute_Mc_and_slope(phi_full, c_values, n_lag_grid=30, target_samples=1500):
    """
    Decimate to chaos-natural rate, compute M_c(n) over log-spaced lags,
    return the per-c K values (correlation) and the cross-c mean M_c curve.
    """
    phi, stride = decimate(phi_full, target_samples)
    phi = phi - phi.mean()
    N = len(phi)
    if N < 50:
        return None, None, stride

    n_max = max(10, N // 10)
    lag_grid = np.unique(np.round(np.geomspace(2, n_max, n_lag_grid)).astype(int))

    Mc_sum = np.zeros(len(lag_grid))
    K_per_c = []
    j = np.arange(1, N+1, dtype=np.float64)

    for c in c_values:
        p = np.cumsum(phi * np.cos(j*c))
        q = np.cumsum(phi * np.sin(j*c))
        Ms = []
        for n in lag_grid:
            if n >= N:
                Ms.append(np.nan)
                continue
            dp = p[n:] - p[:-n]
            dq = q[n:] - q[:-n]
            Ms.append(float(np.mean(dp*dp + dq*dq)))
        Ms = np.array(Ms)
        Mc_sum += np.where(np.isfinite(Ms), Ms, 0.)
        # log-log correlation = K_c
        valid = np.isfinite(Ms) & (Ms > 0)
        if valid.sum() >= 5:
            ln = np.log(lag_grid[valid].astype(float))
            lM = np.log(Ms[valid])
            ln -= ln.mean(); lM -= lM.mean()
            denom = math.sqrt(float((ln*ln).sum()*(lM*lM).sum()))
            if denom > 0:
                K_per_c.append(float((ln*lM).sum()/denom))

    Mc_mean = Mc_sum / N_C
    return Mc_mean, lag_grid, stride, K_per_c

def fit_slope(lag_grid, Mc_vals, frac_lo=None, frac_hi=None):
    """Power-law slope α: log M_c = α log n + const.
    frac_lo / frac_hi restrict the lag range as fraction of n_max."""
    lags = np.array(lag_grid, dtype=float)
    Mc   = np.array(Mc_vals,  dtype=float)
    mask = np.isfinite(Mc) & (Mc > 0)
    if frac_lo is not None: mask &= (lags >= frac_lo * lags[-1])
    if frac_hi is not None: mask &= (lags <= frac_hi * lags[-1])
    if mask.sum() < 3:
        return float('nan'), float('nan')
    ln = np.log(lags[mask]); lM = np.log(Mc[mask])
    coeffs = np.polyfit(ln, lM, 1)
    alpha = float(coeffs[0])
    pred = np.polyval(coeffs, ln)
    ss_res = np.sum((lM-pred)**2); ss_tot = np.sum((lM-lM.mean())**2)
    r2 = float(1. - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    return alpha, r2

def classify(alpha_full):
    if math.isnan(alpha_full): return 'unknown'
    if alpha_full < 0.5:       return 'regular'
    if alpha_full > 1.5:       return 'quasiperiodic'
    return 'chaotic'

# ── system registry (matches chaos_universality_lab.py) ───────────────────

SYSTEMS = [
    # (name, family, x0, dt)  — ICs and dt match chaos_universality_lab.py exactly
    ('lorenz63',           'flow_dissipative',  [1.0,1.0,1.0],                     0.01),
    ('rossler',            'flow_dissipative',  [1.0,1.0,0.0],                     0.05),
    ('duffing_forced',     'flow_dissipative',  [0.5,0.0,0.0],                     0.05),
    ('lorenz96_n5',        'flow_dissipative',  [8.0,8.0,8.0,8.01,8.0],            0.02),
    ('aizawa',             'flow_dissipative',  [0.1,0.0,0.0],                     0.01),
    ('sprott_b',           'flow_dissipative',  [0.05,0.05,0.0],                   0.02),
    ('halvorsen',          'flow_dissipative',  [-5.0,0.0,0.0],                    0.01),
    ('thomas',             'flow_dissipative',  [0.1,0.0,0.0],                     0.05),
    ('chen',               'flow_dissipative',  [-10.0,0.0,37.0],                  0.005),
    ('ueda_forced',        'flow_dissipative',  [2.5,0.0,0.0],                     0.05),
    ('lorenz96_n10',       'flow_dissipative',  [8.0,8.0,8.0,8.01,8.0,8.0,8.0,8.0,8.0,8.0], 0.02),
    # Hamiltonian — ICs and dt exactly as in the primary IC of chaos_universality_lab.py
    ('double_pendulum',    'flow_hamiltonian',  [1.5707963,1.5807963,0.0,0.0],     0.005),
    ('henon_heiles',       'flow_hamiltonian',  [0.0,0.1,0.5,0.0],                 0.02),
    ('cr3bp_earth_moon',   'flow_hamiltonian',  [0.5,0.0,0.0,0.8],                 0.01),
    ('yang_mills_x2y2',    'flow_hamiltonian',  [0.4,0.3,0.2,0.15],                0.02),
    ('pullen_edmonds',     'flow_hamiltonian',  [1.5,1.5,1.0,1.0],                 0.02),
    ('quartic_coupled',    'flow_hamiltonian',  [0.8,0.8,0.6,0.6],                 0.02),
    ('spring_pendulum',    'flow_hamiltonian',  [1.0,0.6,0.0,0.0],                 0.005),
    ('coupled_duffing_ham','flow_hamiltonian',  [0.6,-0.5,0.0,0.1],                0.02),
    ('logistic_r3p9',      'map',               [0.4],                             1.),
    ('henon_map',          'map',               [0.1,0.0],                         1.),
    ('standard_map_K1p2',  'map',               [0.5,0.5],                         1.),
    ('mackey_glass',       'flow_delay',        [0.5],                             0.1),
    ('harmonic',           'control',           [1.0,0.0],                         0.05),
    ('quasiperiodic_2tori','control',           [0.0,0.0],                         0.05),
    ('random_walk',        'control',           [0.12345],                         1.),
]

# ── main ───────────────────────────────────────────────────────────────────

def main():
    # c values in (π/5, 4π/5): same range as the original lab, seeded identically
    rng_c = np.random.default_rng(31)   # matches lab's _rng(31)
    c_values = rng_c.uniform(math.pi/5., 4.*math.pi/5., size=N_C)

    out_systems = []
    t0 = time.time()

    for name, family, x0_list, dt in SYSTEMS:
        t1 = time.time()
        print(f"  {name:<28s}", end='', flush=True)

        x0 = np.array(x0_list, dtype=np.float64)
        traj = gen(name, x0, N_WARMUP, N_ANALYSIS, dt)
        phi = traj[:, 0].astype(np.float64)   # first coordinate, full resolution

        result = compute_Mc_and_slope(phi, c_values, N_LAG_GRID, TARGET_SAMPLES)
        Mc_mean, lag_grid, stride, K_per_c = result

        if Mc_mean is None:
            alpha_full = alpha_short = alpha_long = float('nan')
            r2_full = float('nan')
            K_median = float('nan')
        else:
            alpha_full,  r2_full  = fit_slope(lag_grid, Mc_mean)
            alpha_short, _        = fit_slope(lag_grid, Mc_mean, frac_hi=SHORT_LAG_FRAC)
            alpha_long,  _        = fit_slope(lag_grid, Mc_mean, frac_lo=LONG_LAG_FRAC)
            K_median = float(np.median(K_per_c)) if K_per_c else float('nan')

        label = classify(alpha_full)

        print(f" stride={stride:3d}  α={alpha_full:.3f}  "
              f"α_sh={alpha_short:.3f}  α_lo={alpha_long:.3f}  "
              f"K={K_median:.3f}  → {label}  [{time.time()-t1:.1f}s]")

        def _r(v): return round(float(v),4) if v is not None and not math.isnan(v) else None
        out_systems.append({
            'name':        name,
            'family':      family,
            'stride':      int(stride),
            'alpha_full':  _r(alpha_full),
            'alpha_short': _r(alpha_short),
            'alpha_long':  _r(alpha_long),
            'r2_full':     _r(r2_full),
            'K_median':    _r(K_median),
            'label':       label,
            'lag_grid':    [int(l) for l in lag_grid] if lag_grid is not None else [],
            'Mc_mean':     [_r(v) for v in Mc_mean] if Mc_mean is not None else [],
        })

    result = {
        'version':      'chaos_slope_analysis/0.1.0',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'seed':         SEED,
        'config': {
            'n_warmup':      N_WARMUP,
            'n_analysis':    N_ANALYSIS,
            'n_c':           N_C,
            'target_samples': TARGET_SAMPLES,
            'n_lag_grid':    N_LAG_GRID,
            'short_lag_frac': SHORT_LAG_FRAC,
            'long_lag_frac':  LONG_LAG_FRAC,
            'c_range':       [math.pi/5., 4.*math.pi/5.],
        },
        'classification_rule': 'alpha_full < 0.5 → regular; 0.5-1.5 → chaotic; > 1.5 → quasiperiodic',
        'systems': out_systems,
        'wall_seconds': round(time.time()-t0, 1),
    }

    out_path = Path(__file__).parent.parent / 'results' / 'chaos_slope_analysis.json'
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}  ({time.time()-t0:.1f}s total)")

if __name__ == '__main__':
    main()
