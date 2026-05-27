'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Boom Convergence Sweep.

Subdivides the leading-edge wrap (seg1: B2 -> LE -> B1) and trailing-edge
wrap (seg3 + seg4: B3 -> P4 -> B4) into N sub-panels, applies Megson
pair-attribution per sub-panel, and tracks I_XX convergence toward the
continuous-skin "ground truth" computed about the pre-idealization centroid.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC  = _ROOT / 'src'

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Default Database Paths ================
_AFL_DIR = _ROOT / 'data' / 'airfoils'
_FIG_OUT = _HERE / 'boom_convergence.png'

# ================ Module imports ================
from geometry.geom_properties import GeomPropCalculator

# ================ Global variables ================
_CHORD     = 300.0
_TWIST     = 0.0
_Y_STA     = 0.0
_XW1       = 60.0  / _CHORD
_XW2       = 210.0 / _CHORD
_E1        = 200_000.0
_E2        =  10_000.0
_G         =   6_000.0
_T_SEG     = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 4.0, 2.0])
_T_FLANGE  = np.array([3.0, 2.5, 3.0, 2.0])
_BF        = np.array([12.0, 10.0, 8.0, 8.0])

_N_RANGE   = list(range(0, 11))   # number of intermediate booms per arc
_AFL_NAME  = 'wortmannfx63137_AirfoilData.json'


# ================================================================================
# Helpers
# ================================================================================

def build_pre_id_calc(afl_filename: str) -> GeomPropCalculator:
    '''Run pipeline up to (but not including) idealization.'''
    with open(_AFL_DIR / afl_filename) as fh:
        raw = json.load(fh)
    afl = {k: np.array(v) for k, v in raw.items()}

    calc = GeomPropCalculator(
        x_upper    = afl['x_upper'],
        y_upper    = afl['y_upper'],
        x_lower    = afl['x_lower'],
        y_lower    = afl['y_lower'],
        x_camber   = afl['x_camber'],
        y_camber   = afl['y_camber'],
        chord      = _CHORD,
        twist  = _TWIST,
        Y_sta      = _Y_STA,
        xw1        = _XW1,
        xw2        = _XW2,
        bf1        = _BF[0], bf2 = _BF[1],
        bf3        = _BF[2], bf4 = _BF[3],
        t_seg      = _T_SEG,
        G_seg      = np.full(7, _G),
        E1_seg     = np.full(7, _E1),
        E2_seg     = np.full(7, _E2),
        t_flange   = _T_FLANGE,
        E1_flange  = np.full(4, _E1),
        G_flange   = np.full(4, _G),
        enable_logging = False,
    )
    calc._scale_airfoil()
    calc._find_spar_intersections()
    calc._segment_T1()
    calc._segment_T2()
    calc._segment_T3()
    calc._compute_segment_properties()
    calc._compute_delta_matrix()
    calc._compute_cell_areas()
    calc._compute_Delta_and_J()
    calc._compute_centroid()
    calc._compute_inertia()
    calc._compute_principal_inertia()
    return calc


def resample_polyline(
    x: np.ndarray, z: np.ndarray, n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    '''Resample a polyline at n_nodes uniform arc-length stations.'''
    dx = np.diff(x); dz = np.diff(z)
    ds = np.sqrt(dx**2 + dz**2)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    L  = float(s[-1])
    s_new = np.linspace(0.0, L, n_nodes)
    return np.interp(s_new, s, x), np.interp(s_new, s, z), L


def megson_pair(
    tL: float,
    sigma_n: float, sigma_m: float,
) -> tuple[float, float]:
    '''Megson pair contribution for a straight panel.'''
    Bn = (tL / 6.0) * (2.0 + sigma_m / sigma_n)
    Bm = (tL / 6.0) * (2.0 + sigma_n / sigma_m)
    return Bn, Bm


# ================================================================================
# Core sweep
# ================================================================================

def _subdivide_arc(
    x: np.ndarray, z: np.ndarray, N: int,
    booms_u: list, booms_w: list, booms_A: list,
    end_n: int, end_m: int,
    Xc: float, Zc: float,
    t: float, sigma,
) -> None:
    '''
    Place N intermediate booms uniformly along the polyline (x, z) between
    the existing booms `end_n` and `end_m`, then apply Megson pair to each
    sub-panel. Mutates the boom lists in-place.
    '''
    x_n, z_n, L = resample_polyline(x, z, N + 2)
    chain = [end_n]
    for k in range(1, N + 1):
        booms_u.append(float(x_n[k] - Xc))
        booms_w.append(float(z_n[k] - Zc))
        booms_A.append(0.0)
        chain.append(len(booms_u) - 1)
    chain.append(end_m)

    tL_sub = t * (L / (N + 1))
    for k in range(N + 1):
        i_n, i_m = chain[k], chain[k + 1]
        s_n = sigma(booms_u[i_n], booms_w[i_n])
        s_m = sigma(booms_u[i_m], booms_w[i_m])
        Bn, Bm = megson_pair(tL_sub, s_n, s_m)
        booms_A[i_n] += Bn
        booms_A[i_m] += Bm


def evaluate(
    N_LE: int, N_TE: int, calc: GeomPropCalculator,
    N_MID: int = 0,
) -> float:
    '''
    Build the idealized boom set with N_LE intermediate booms on the LE
    wrap, N_TE on the TE wrap, and N_MID on each upper/lower mid-skin
    (seg2, seg5). Returns I_XX of the boom-only section computed about
    the pre-idealization centroid.
    '''
    Xc, Zc     = calc.Xc, calc.Zc
    I_ZZ, I_XZ = calc.I_ZZ, calc.I_XZ
    T1         = {s['label']: s for s in calc.T1}

    def sigma(u: float, w: float) -> float:
        return I_ZZ * w - I_XZ * u

    # -------- Original 4 booms + flange areas --------
    B_pos   = np.array([calc.B1, calc.B2, calc.B3, calc.B4])
    booms_u = list(B_pos[:, 0] - Xc)
    booms_w = list(B_pos[:, 1] - Zc)
    booms_A = list(np.asarray(calc.A_flange, dtype=float))

    # -------- LE arc: pin P1 (LE) as a forced node when N_LE >= 1 --------
    seg1 = T1['seg1']
    if N_LE >= 1:
        # split seg1 at its argmin-x (P1 = LE)
        ba = int(np.argmin(seg1['x']))
        x_lower = seg1['x'][:ba + 1];  z_lower = seg1['z'][:ba + 1]   # B2 -> LE
        x_upper = seg1['x'][ba:];      z_upper = seg1['z'][ba:]       # LE -> B1
        # add LE as a forced intermediate boom
        booms_u.append(float(seg1['x'][ba] - Xc))
        booms_w.append(float(seg1['z'][ba] - Zc))
        booms_A.append(0.0)
        i_LE = len(booms_u) - 1
        # split remaining (N_LE - 1) booms across the two halves by length
        L_lo = float(np.sum(np.sqrt(np.diff(x_lower)**2 + np.diff(z_lower)**2)))
        L_up = float(np.sum(np.sqrt(np.diff(x_upper)**2 + np.diff(z_upper)**2)))
        n_extra  = N_LE - 1
        n_lo     = int(round(n_extra * L_lo / (L_lo + L_up)))
        n_up     = n_extra - n_lo
        _subdivide_arc(x_lower, z_lower, n_lo, booms_u, booms_w, booms_A,
                       1, i_LE, Xc, Zc, _T_SEG[0], sigma)
        _subdivide_arc(x_upper, z_upper, n_up, booms_u, booms_w, booms_A,
                       i_LE, 0, Xc, Zc, _T_SEG[0], sigma)
    else:
        _subdivide_arc(seg1['x'], seg1['z'], 0, booms_u, booms_w, booms_A,
                       1, 0, Xc, Zc, _T_SEG[0], sigma)

    # -------- TE arc: pin P4 (TE) as a forced node when N_TE >= 1 --------
    seg3, seg4 = T1['seg3'], T1['seg4']
    L_seg3 = float(np.sum(np.sqrt(np.diff(seg3['x'])**2 + np.diff(seg3['z'])**2)))
    L_seg4 = float(np.sum(np.sqrt(np.diff(seg4['x'])**2 + np.diff(seg4['z'])**2)))
    t_TE_avg = (_T_SEG[2] * L_seg3 + _T_SEG[3] * L_seg4) / (L_seg3 + L_seg4)
    if N_TE >= 1:
        # P4 is the shared endpoint of seg3 and seg4
        x_te_up = seg3['x'];  z_te_up = seg3['z']      # B3 -> P4
        x_te_lo = seg4['x'];  z_te_lo = seg4['z']      # P4 -> B4
        booms_u.append(float(seg3['x'][-1] - Xc))
        booms_w.append(float(seg3['z'][-1] - Zc))
        booms_A.append(0.0)
        i_TE = len(booms_u) - 1
        n_extra = N_TE - 1
        n_up    = int(round(n_extra * L_seg3 / (L_seg3 + L_seg4)))
        n_lo    = n_extra - n_up
        _subdivide_arc(x_te_up, z_te_up, n_up, booms_u, booms_w, booms_A,
                       2, i_TE, Xc, Zc, _T_SEG[2], sigma)
        _subdivide_arc(x_te_lo, z_te_lo, n_lo, booms_u, booms_w, booms_A,
                       i_TE, 3, Xc, Zc, _T_SEG[3], sigma)
    else:
        x_TE = np.concatenate([seg3['x'], seg4['x'][1:]])
        z_TE = np.concatenate([seg3['z'], seg4['z'][1:]])
        _subdivide_arc(x_TE, z_TE, 0, booms_u, booms_w, booms_A,
                       2, 3, Xc, Zc, t_TE_avg, sigma)

    # -------- Upper / lower mid skins (seg2 and seg5) with N_MID --------
    seg2 = T1['seg2']    # B1 -> B3
    _subdivide_arc(seg2['x'], seg2['z'], N_MID, booms_u, booms_w, booms_A,
                   0, 2, Xc, Zc, _T_SEG[1], sigma)
    seg5 = T1['seg5']    # B4 -> B2
    _subdivide_arc(seg5['x'], seg5['z'], N_MID, booms_u, booms_w, booms_A,
                   3, 1, Xc, Zc, _T_SEG[4], sigma)

    # -------- Spar webs (straight panels, no subdivision needed) --------
    for seg_idx, n, m in [(5, 0, 1), (6, 2, 3)]:
        tL  = _T_SEG[seg_idx] * float(calc.s_k[seg_idx])
        s_n = sigma(booms_u[n], booms_w[n])
        s_m = sigma(booms_u[m], booms_w[m])
        Bn, Bm = megson_pair(tL, s_n, s_m)
        booms_A[n] += Bn
        booms_A[m] += Bm

    A = np.asarray(booms_A)
    w = np.asarray(booms_w)
    return float(np.sum(A * w**2))


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    calc     = build_pre_id_calc(_AFL_NAME)
    I_XX_pre = float(calc.I_XX)

    # 2-D sweep over (N_LE, N_TE)
    grid = np.zeros((len(_N_RANGE), len(_N_RANGE)))
    for i, n_le in enumerate(_N_RANGE):
        for j, n_te in enumerate(_N_RANGE):
            grid[i, j] = evaluate(n_le, n_te, calc)

    print(f"Airfoil           : {_AFL_NAME}")
    print(f"Pre-id I_XX (truth): {I_XX_pre:>12.0f} mm^4")
    print()
    print("Post-idealization I_XX as a function of (N_LE, N_TE):")
    print()

    header = "  N_LE\\N_TE " + "".join(f"{n:>9d}" for n in _N_RANGE)
    print(header)
    print("-" * len(header))
    for i, n_le in enumerate(_N_RANGE):
        row = f"  {n_le:>8d}  "
        for j in range(len(_N_RANGE)):
            err = (grid[i, j] / I_XX_pre - 1.0) * 100.0
            row += f"{err:>+8.2f} "
        print(row)
    print()
    print("(values are percent error vs pre-id I_XX)")
    print()

    # Diagonal slice for compact reading
    print("Diagonal sweep (N_LE = N_TE = N):")
    print(f"{'N':>4}  {'I_XX':>12}  {'err [%]':>8}")
    for n in _N_RANGE:
        v = evaluate(n, n, calc)
        err = (v / I_XX_pre - 1.0) * 100.0
        print(f"{n:>4}  {v:>12.0f}  {err:>+8.3f}")
    print()

    # Effect of also subdividing the upper/lower mid skins (seg2, seg5)
    print("Adding N_MID intermediate booms on each mid-skin (N_LE=N_TE=N_MID):")
    print(f"{'N':>4}  {'I_XX':>12}  {'err [%]':>8}")
    for n in _N_RANGE:
        v   = evaluate(n, n, calc, N_MID=n)
        err = (v / I_XX_pre - 1.0) * 100.0
        print(f"{n:>4}  {v:>12.0f}  {err:>+8.3f}")
    print()

    # Minimum N along each axis (others fixed at 0) to reach a target error
    def first_below(target_pct: float, knob: str) -> int | None:
        for n in _N_RANGE:
            if knob == 'LE':
                v = evaluate(n, 0, calc, N_MID=0)
            elif knob == 'TE':
                v = evaluate(0, n, calc, N_MID=0)
            elif knob == 'MID':
                v = evaluate(0, 0, calc, N_MID=n)
            else:
                v = evaluate(n, n, calc, N_MID=n)
            if abs(v / I_XX_pre - 1.0) * 100.0 < target_pct:
                return n
        return None

    for tgt in (5.0, 1.0, 0.1):
        print(f"Min N for <{tgt:>4.1f}% err: "
              f"LE-only={first_below(tgt,'LE')}  "
              f"TE-only={first_below(tgt,'TE')}  "
              f"MID-only={first_below(tgt,'MID')}  "
              f"all={first_below(tgt,'ALL')}")

    # -------- Plot --------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    err_grid = (grid / I_XX_pre - 1.0) * 100.0
    im = ax[0].imshow(
        err_grid, origin='lower', cmap='RdBu_r',
        vmin=-max(abs(err_grid.min()), abs(err_grid.max())),
        vmax= max(abs(err_grid.min()), abs(err_grid.max())),
        extent=[_N_RANGE[0]-0.5, _N_RANGE[-1]+0.5,
                _N_RANGE[0]-0.5, _N_RANGE[-1]+0.5],
        aspect='auto',
    )
    ax[0].set_xlabel('N_TE (intermediate booms on TE arc)')
    ax[0].set_ylabel('N_LE (intermediate booms on LE arc)')
    ax[0].set_title('I_XX error [%] vs reference skin integral')
    fig.colorbar(im, ax=ax[0], label='error [%]')

    diag = [(evaluate(n, n, calc) / I_XX_pre - 1.0) * 100.0 for n in _N_RANGE]
    le_only = [(evaluate(n, 0, calc) / I_XX_pre - 1.0) * 100.0 for n in _N_RANGE]
    te_only = [(evaluate(0, n, calc) / I_XX_pre - 1.0) * 100.0 for n in _N_RANGE]
    ax[1].plot(_N_RANGE, diag,    'o-', label='diagonal (N_LE=N_TE)')
    ax[1].plot(_N_RANGE, le_only, 's--', label='LE only (N_TE=0)')
    ax[1].plot(_N_RANGE, te_only, '^--', label='TE only (N_LE=0)')
    ax[1].axhline(0.0, color='k', linewidth=0.5)
    ax[1].set_xlabel('N intermediate booms')
    ax[1].set_ylabel('I_XX error [%]')
    ax[1].set_title('Convergence to skin-truth I_XX')
    ax[1].grid(True, alpha=0.4)
    ax[1].legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(_FIG_OUT, dpi=140)
    print(f"\nFigure saved to {_FIG_OUT}")
