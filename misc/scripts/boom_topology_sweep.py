'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Boom Topology Sweep.

Compares post-idealization geometric properties for three boom topologies:
  T_4   : 4 corner booms only (B1..B4)
  T_T2  : 10-boom T2 staging  (B1..B4 + P1..P6)
  T_NQ  : 10-boom proposal    (B1..B4 + N1, N2, Q1, Q2 + P5, P6)
            * P1 replaced by two nose booms between LE and xw1
            * P4 replaced by two tail booms between xw2 and TE
            * P2 and P3 dropped (spar webs become single panels)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
import json
from pathlib import Path

import numpy as np

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC  = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Default Database Paths ================
_AFL_DIR = _ROOT / 'data' / 'airfoils'

# ================ Module imports ================
from geometry.geom_properties import GeomPropCalculator

# ================ Global variables ================
_CHORD     = 300.0
_XW1       = 60.0  / _CHORD
_XW2       = 210.0 / _CHORD
_T_SEG     = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 4.0, 2.0])
_T_FLANGE  = np.array([3.0, 2.5, 3.0, 2.0])
_BF        = np.array([12.0, 10.0, 8.0, 8.0])
_AFLS      = [
    'wortmannfx63137_AirfoilData.json',
    'e169_AirfoilData.json',
]


# ================================================================================
# Helpers
# ================================================================================

def build_pre_id_calc(afl_filename: str) -> GeomPropCalculator:
    with open(_AFL_DIR / afl_filename) as fh:
        raw = json.load(fh)
    afl = {k: np.array(v) for k, v in raw.items()}
    calc = GeomPropCalculator(
        x_upper=afl['x_upper'], y_upper=afl['y_upper'],
        x_lower=afl['x_lower'], y_lower=afl['y_lower'],
        x_camber=afl['x_camber'], y_camber=afl['y_camber'],
        chord=_CHORD, twist=0.0, Y_sta=0.0,
        xw1=_XW1, xw2=_XW2,
        bf1=_BF[0], bf2=_BF[1], bf3=_BF[2], bf4=_BF[3],
        t_seg=_T_SEG, G_seg=np.full(7, 6000.0),
        E1_seg=np.full(7, 200000.0), E2_seg=np.full(7, 10000.0),
        t_flange=_T_FLANGE,
        E1_flange=np.full(4, 200000.0), G_flange=np.full(4, 6000.0),
        enable_logging=False,
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


def arc_len(x: np.ndarray, z: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(z)**2)))


def split_at_x(x: np.ndarray, z: np.ndarray, x_split: float):
    '''Split a polyline at a target x value (assumes x monotonic).'''
    if x[0] > x[-1]:
        x_rev, z_rev = x[::-1], z[::-1]
        a, b = split_at_x(x_rev, z_rev, x_split)
        return (a[0][::-1], a[1][::-1]), (b[0][::-1], b[1][::-1])
    idx = np.searchsorted(x, x_split)
    z_split = float(np.interp(x_split, x, z))
    x_left  = np.concatenate([x[:idx], [x_split]])
    z_left  = np.concatenate([z[:idx], [z_split]])
    x_right = np.concatenate([[x_split], x[idx:]])
    z_right = np.concatenate([[z_split], z[idx:]])
    return (x_left, z_left), (x_right, z_right)


def megson_pair(tL: float, sn: float, sm: float) -> tuple[float, float]:
    Bn = (tL / 6.0) * (2.0 + sm / sn)
    Bm = (tL / 6.0) * (2.0 + sn / sm)
    return Bn, Bm


# ================================================================================
# Topology evaluators
# ================================================================================

def _common(calc):
    Xc, Zc     = calc.Xc, calc.Zc
    I_ZZ, I_XZ = calc.I_ZZ, calc.I_XZ
    def sigma(u, w): return I_ZZ * w - I_XZ * u
    return Xc, Zc, sigma


def topo_T4(calc) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    '''4-boom corner-only topology: seg3+seg4 merged around TE.'''
    Xc, Zc, sigma = _common(calc)
    T1 = {s['label']: s for s in calc.T1}
    pos = np.array([calc.B1, calc.B2, calc.B3, calc.B4])
    u = pos[:, 0] - Xc; w = pos[:, 1] - Zc
    A = np.asarray(calc.A_flange, dtype=float).copy()

    panels = [
        ([0],    1, 0),  # seg1: B2 <-> B1
        ([1],    0, 2),  # seg2: B1 <-> B3
        ([2, 3], 2, 3),  # seg3+4: B3 <-> B4
        ([4],    3, 1),  # seg5: B4 <-> B2
        ([5],    0, 1),  # seg6
        ([6],    2, 3),  # seg7
    ]
    s_seg = np.array([arc_len(s['x'], s['z']) for s in calc.T1])
    for segs, n, m in panels:
        tL = float(sum(_T_SEG[k] * s_seg[k] for k in segs))
        sn, sm = sigma(u[n], w[n]), sigma(u[m], w[m])
        Bn, Bm = megson_pair(tL, sn, sm)
        A[n] += Bn; A[m] += Bm
    return pos, A, np.array(['B1','B2','B3','B4']), ['B1','B2','B3','B4']


def topo_T2(calc) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    '''10-boom T2 staging (B1-B4 + P1-P6).'''
    Xc, Zc, sigma = _common(calc)
    P_pts = np.array([calc.P1, calc.P2, calc.P3, calc.P4, calc.P5, calc.P6])
    pos = np.vstack([calc.B1, calc.B2, calc.B3, calc.B4, P_pts])
    labels = ['B1','B2','B3','B4','P1','P2','P3','P4','P5','P6']
    u = pos[:, 0] - Xc; w = pos[:, 1] - Zc

    A = np.zeros(10)
    A[:4] = np.asarray(calc.A_flange, dtype=float)

    pos_lookup = {(round(x, 6), round(z, 6)): i for i, (x, z) in enumerate(pos)}

    for r in calc.T2:
        n = int(r['boom'])
        m = pos_lookup[(round(float(r['far_x']), 6),
                        round(float(r['far_z']), 6))]
        tL = float(_T_SEG[r['T1_seg']]) * arc_len(r['x'], r['z'])
        sn, sm = sigma(u[n], w[n]), sigma(u[m], w[m])
        Bn, Bm = megson_pair(tL, sn, sm)
        A[n] += Bn; A[m] += Bm
    return pos, A, np.array(labels), labels


def topo_T2_no_P23(calc) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    '''8-boom variant: B1-B4 + P1, P4, P5, P6 (drop P2 and P3 only).'''
    Xc, Zc, sigma = _common(calc)
    pos = np.array([calc.B1, calc.B2, calc.B3, calc.B4,
                    calc.P1, calc.P4, calc.P5, calc.P6])
    labels = ['B1','B2','B3','B4','P1','P4','P5','P6']
    iB1, iB2, iB3, iB4, iP1, iP4, iP5, iP6 = range(8)
    u = pos[:, 0] - Xc; w = pos[:, 1] - Zc
    A = np.zeros(8); A[:4] = np.asarray(calc.A_flange, dtype=float)

    def add(tL, n, m):
        sn, sm = sigma(u[n], w[n]), sigma(u[m], w[m])
        Bn, Bm = megson_pair(tL, sn, sm)
        A[n] += Bn; A[m] += Bm

    T1 = {s['label']: s for s in calc.T1}
    seg1 = T1['seg1']; seg2 = T1['seg2']; seg3 = T1['seg3']
    seg4 = T1['seg4']; seg5 = T1['seg5']; seg6 = T1['seg6']
    seg7 = T1['seg7']
    t = _T_SEG

    # seg1 split at LE (P1): B2 -> P1 -> B1
    ba = int(np.argmin(seg1['x']))
    add(t[0] * arc_len(seg1['x'][:ba+1], seg1['z'][:ba+1]), iB2, iP1)
    add(t[0] * arc_len(seg1['x'][ba:],   seg1['z'][ba:]),   iP1, iB1)
    # seg2 split at P5
    P5_x = calc.P5[0]
    a, b = split_at_x(seg2['x'], seg2['z'], P5_x)
    add(t[1] * arc_len(*a), iB1, iP5)
    add(t[1] * arc_len(*b), iP5, iB3)
    # seg3 + seg4 split at P4 (TE)
    add(t[2] * arc_len(seg3['x'], seg3['z']), iB3, iP4)
    add(t[3] * arc_len(seg4['x'], seg4['z']), iP4, iB4)
    # seg5 split at P6 (need to reverse for monotone x)
    P6_x = calc.P6[0]
    s5_xr, s5_zr = seg5['x'][::-1], seg5['z'][::-1]
    a, b = split_at_x(s5_xr, s5_zr, P6_x)
    add(t[4] * arc_len(*a), iB2, iP6)
    add(t[4] * arc_len(*b), iP6, iB4)
    # seg6 single panel (no P2): B1 <-> B2
    add(t[5] * arc_len(seg6['x'], seg6['z']), iB1, iB2)
    # seg7 single panel (no P3): B3 <-> B4
    add(t[6] * arc_len(seg7['x'], seg7['z']), iB3, iB4)

    return pos, A, np.array(labels), labels


def topo_NQ(calc, x_N_frac: float = 0.5,
            x_Q_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    '''
    10-boom proposal: drop P1, P2, P3, P4. Add N1/N2 in the nose region,
    Q1/Q2 in the tail region. Keep P5, P6.
        x_N_frac : fractional position of N1, N2 inside [0, xw1] (0 = LE, 1 = spar)
        x_Q_frac : fractional position of Q1, Q2 inside [xw2, chord]
    '''
    Xc, Zc, sigma = _common(calc)
    T1 = {s['label']: s for s in calc.T1}
    seg1 = T1['seg1']; seg2 = T1['seg2']; seg3 = T1['seg3']
    seg4 = T1['seg4']; seg5 = T1['seg5']; seg6 = T1['seg6']
    seg7 = T1['seg7']

    xN  = x_N_frac * _CHORD * _XW1
    xQ  = _CHORD * _XW2 + x_Q_frac * _CHORD * (1.0 - _XW2)

    # seg1 = B2 -> LE -> B1: split at LE, then at xN on each half
    ba = int(np.argmin(seg1['x']))
    x_lo, z_lo = seg1['x'][:ba+1], seg1['z'][:ba+1]   # B2 -> LE  (x decreasing)
    x_up, z_up = seg1['x'][ba:],   seg1['z'][ba:]     # LE -> B1  (x increasing)
    # N2 sits on lower at x = xN
    (a_lo, b_lo) = split_at_x(x_lo, z_lo, xN)         # a_lo: B2->N2, b_lo: N2->LE
    # N1 sits on upper at x = xN
    (a_up, b_up) = split_at_x(x_up, z_up, xN)         # a_up: LE->N1, b_up: N1->B1

    N2_xz = (a_lo[0][-1], a_lo[1][-1])
    N1_xz = (a_up[0][-1], a_up[1][-1])

    # seg3 = B3 -> P4 (TE), seg4 = P4 -> B4: split each at xQ
    (a_seg3, b_seg3) = split_at_x(seg3['x'], seg3['z'], xQ)  # a: B3->Q1, b: Q1->TE
    (a_seg4, b_seg4) = split_at_x(seg4['x'], seg4['z'], xQ)  # a: TE->Q2, b: Q2->B4
    Q1_xz = (a_seg3[0][-1], a_seg3[1][-1])
    Q2_xz = (a_seg4[0][-1], a_seg4[1][-1])

    pos = np.array([
        calc.B1, calc.B2, calc.B3, calc.B4,
        N1_xz, N2_xz, Q1_xz, Q2_xz,
        calc.P5, calc.P6,
    ])
    labels = ['B1','B2','B3','B4','N1','N2','Q1','Q2','P5','P6']
    iB1, iB2, iB3, iB4, iN1, iN2, iQ1, iQ2, iP5, iP6 = range(10)

    u = pos[:, 0] - Xc; w = pos[:, 1] - Zc
    A = np.zeros(10)
    A[:4] = np.asarray(calc.A_flange, dtype=float)

    # Pair attribution per sub-panel
    def add(tL, n, m):
        sn, sm = sigma(u[n], w[n]), sigma(u[m], w[m])
        Bn, Bm = megson_pair(tL, sn, sm)
        A[n] += Bn; A[m] += Bm

    t1 = _T_SEG[0]; t2 = _T_SEG[1]; t3 = _T_SEG[2]
    t4 = _T_SEG[3]; t5 = _T_SEG[4]; t6 = _T_SEG[5]; t7 = _T_SEG[6]

    # seg1 split into 3: B2->N2, N2->LE->N1 (around LE), N1->B1
    L_B2N2 = arc_len(*a_lo)
    # b_lo: N2 -> LE; b_up: LE -> N1; concatenate for the curved middle
    L_N2N1 = arc_len(*b_lo) + arc_len(*b_up)
    L_N1B1 = arc_len(*b_up) if False else arc_len(*b_up)  # placeholder
    L_N1B1 = arc_len(b_up[0], b_up[1])
    # WAIT: b_up = (LE -> N1), so the segment N1->B1 is the OTHER half of x_up
    # Re-split: x_up split at xN gave a_up=(LE->N1), b_up=(N1->B1)
    # so:
    L_N2_to_LE = arc_len(*b_lo)
    L_LE_to_N1 = arc_len(*a_up)
    L_N2N1 = L_N2_to_LE + L_LE_to_N1
    L_N1B1 = arc_len(*b_up)

    add(t1 * L_B2N2, iB2, iN2)
    add(t1 * L_N2N1, iN2, iN1)
    add(t1 * L_N1B1, iN1, iB1)

    # seg2 split at P5: B1 -> P5 -> B3 (existing T1 has the polyline)
    # Use P5 = midpoint x, get arc lengths from seg2 polyline
    P5_x = calc.P5[0]
    (a_s2, b_s2) = split_at_x(seg2['x'], seg2['z'], P5_x)
    add(t2 * arc_len(*a_s2), iB1, iP5)
    add(t2 * arc_len(*b_s2), iP5, iB3)

    # seg3 + seg4 split into 3: B3->Q1, Q1->TE->Q2, Q2->B4
    L_B3Q1 = arc_len(*a_seg3)
    L_Q1TE = arc_len(*b_seg3)
    L_TEQ2 = arc_len(*a_seg4)
    L_Q2B4 = arc_len(*b_seg4)
    L_Q1Q2 = L_Q1TE + L_TEQ2
    # Use a length-weighted thickness for the wrap-around panel
    t_Q1Q2 = (t3 * L_Q1TE + t4 * L_TEQ2) / L_Q1Q2
    add(t3 * L_B3Q1, iB3, iQ1)
    add(t_Q1Q2 * L_Q1Q2, iQ1, iQ2)
    add(t4 * L_Q2B4, iQ2, iB4)

    # seg5 split at P6: B4 -> P6 -> B2  (note: seg5 polyline is B4..P6..B2)
    # split_at_x assumes monotone; seg5 x decreases. Reverse first.
    P6_x = calc.P6[0]
    s5_xr, s5_zr = seg5['x'][::-1], seg5['z'][::-1]   # now B2..P6..B4
    (a_s5, b_s5) = split_at_x(s5_xr, s5_zr, P6_x)
    add(t5 * arc_len(*a_s5), iB2, iP6)
    add(t5 * arc_len(*b_s5), iP6, iB4)

    # seg6 single panel: B1 <-> B2
    add(t6 * arc_len(seg6['x'], seg6['z']), iB1, iB2)
    # seg7 single panel: B3 <-> B4
    add(t7 * arc_len(seg7['x'], seg7['z']), iB3, iB4)

    return pos, A, np.array(labels), labels


# ================================================================================
# Reporting
# ================================================================================

def report_topology(name: str, pos, A, labels, calc):
    sumA = float(np.sum(A))
    Xc = float(np.dot(pos[:, 0], A)) / sumA
    Zc = float(np.dot(pos[:, 1], A)) / sumA
    du = pos[:, 0] - Xc; dw = pos[:, 1] - Zc
    IXX = float(np.dot(dw**2, A))
    IZZ = float(np.dot(du**2, A))
    IXZ = float(np.dot(du * dw, A))

    # also I_XX about pre-id centroid for like-for-like comparison
    du_pre = pos[:, 0] - calc.Xc; dw_pre = pos[:, 1] - calc.Zc
    IXX_pre_axis = float(np.dot(dw_pre**2, A))
    IZZ_pre_axis = float(np.dot(du_pre**2, A))

    print(f"\n--- {name} (n_booms={len(A)}) ---")
    print(f"  about boom-only centroid:")
    print(f"    Xc={Xc:>9.3f}  Zc={Zc:>+8.3f}")
    print(f"    I_XX={IXX:>12.0f}  ({(IXX/calc.I_XX-1)*100:+6.2f}%)")
    print(f"    I_ZZ={IZZ:>12.0f}  ({(IZZ/calc.I_ZZ-1)*100:+6.2f}%)")
    print(f"    I_XZ={IXZ:>+12.0f}")
    print(f"  about pre-id centroid (same-axis comparison):")
    print(f"    I_XX={IXX_pre_axis:>12.0f}  ({(IXX_pre_axis/calc.I_XX-1)*100:+6.2f}%)")
    print(f"    I_ZZ={IZZ_pre_axis:>12.0f}  ({(IZZ_pre_axis/calc.I_ZZ-1)*100:+6.2f}%)")
    return {
        'name': name, 'Xc': Xc, 'Zc': Zc,
        'I_XX': IXX, 'I_ZZ': IZZ, 'I_XZ': IXZ,
        'I_XX_pre': IXX_pre_axis, 'I_ZZ_pre': IZZ_pre_axis,
    }


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    for afl in _AFLS:
        print("=" * 70)
        print(f"Airfoil: {afl}")
        print("=" * 70)
        calc = build_pre_id_calc(afl)
        print(f"Pre-id (skin truth):")
        print(f"  Xc={calc.Xc:>9.3f}  Zc={calc.Zc:>+8.3f}")
        print(f"  I_XX={calc.I_XX:>12.0f}  I_ZZ={calc.I_ZZ:>12.0f}  "
              f"I_XZ={calc.I_XZ:>+12.0f}")

        pos, A, labels, _ = topo_T4(calc)
        report_topology("T_4  (4 corner booms)", pos, A, labels, calc)

        pos, A, labels, _ = topo_T2(calc)
        report_topology("T_T2 (B1-4 + P1-P6)", pos, A, labels, calc)

        pos, A, labels, _ = topo_T2_no_P23(calc)
        report_topology("T_T2_noP23 (drop P2,P3 only)", pos, A, labels, calc)

        # Best-case T_NQ placement for compact summary (xN=0.5, xQ=0.5)
        pos, A, labels, _ = topo_NQ(calc, 0.5, 0.5)
        report_topology("T_NQ (xN=0.50, xQ=0.50)", pos, A, labels, calc)
        pos, A, labels, _ = topo_NQ(calc, 0.5, 0.25)
        report_topology("T_NQ (xN=0.50, xQ=0.25)", pos, A, labels, calc)
