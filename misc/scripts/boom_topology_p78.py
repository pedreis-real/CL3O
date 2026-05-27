'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
P7 / P8 Topology Sweep.

Adds two booms to the T2 topology at the airfoil's global z-extrema:
    P7 : upper surface point with maximum z
    P8 : lower surface point with minimum z

P7 and P8 dispatch to their host T1 segment based on x-position:

    x < xw1            -> seg1 (LE wrap)
    xw1 <= x < xw2     -> seg2 (upper mid) or seg5 (lower mid)
    x >= xw2           -> seg3 / seg4 (TE wrap)

Three scenarios w.r.t. xw1 are exercised by varying the front-spar
position so that P7 / P8 fall in different host segments:

    Scenario 1 : xw1 < x_P7 < x_P8                (both aft of xw1)
    Scenario 2 : x_P7 < xw1 < x_P8                (P7 forward, P8 aft)
    Scenario 3 : x_P7 < x_P8 < xw1                (both forward of xw1)

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
_XW2_FRAC  = 210.0 / _CHORD
_T_SEG     = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 4.0, 2.0])
_T_FLANGE  = np.array([3.0, 2.5, 3.0, 2.0])
_BF        = np.array([12.0, 10.0, 8.0, 8.0])
_AFL_NAME  = 'wortmannfx63137_AirfoilData.json'


# ================================================================================
# Helpers
# ================================================================================

def build_pre_id_calc(afl_filename: str, xw1_frac: float) -> GeomPropCalculator:
    with open(_AFL_DIR / afl_filename) as fh:
        raw = json.load(fh)
    afl = {k: np.array(v) for k, v in raw.items()}
    calc = GeomPropCalculator(
        x_upper=afl['x_upper'], y_upper=afl['y_upper'],
        x_lower=afl['x_lower'], y_lower=afl['y_lower'],
        x_camber=afl['x_camber'], y_camber=afl['y_camber'],
        chord=_CHORD, twist=0.0, Y_sta=0.0,
        xw1=xw1_frac, xw2=_XW2_FRAC,
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


def find_z_extrema(calc) -> tuple[tuple[float, float], tuple[float, float]]:
    '''Return (P7=(x,z) at upper z-max, P8=(x,z) at lower z-min).'''
    iu = int(np.argmax(calc.yu))
    il = int(np.argmin(calc.yl))
    return (float(calc.xu[iu]), float(calc.yu[iu])), \
           (float(calc.xl[il]), float(calc.yl[il]))


def arc_param(seg_xs: np.ndarray, seg_zs: np.ndarray,
              x_b: float, z_b: float) -> float:
    '''Arc length from segment start to the polyline node nearest to (x_b, z_b).'''
    d2 = (seg_xs - x_b)**2 + (seg_zs - z_b)**2
    i  = int(np.argmin(d2))
    if i == 0:
        return 0.0
    dx = np.diff(seg_xs[:i+1]); dz = np.diff(seg_zs[:i+1])
    return float(np.sum(np.sqrt(dx**2 + dz**2)))


def megson_pair(tL: float, sn: float, sm: float) -> tuple[float, float]:
    Bn = (tL / 6.0) * (2.0 + sm / sn)
    Bm = (tL / 6.0) * (2.0 + sn / sm)
    return Bn, Bm


# ================================================================================
# Topology builder (variable boom set, generic chain on each T1 seg)
# ================================================================================

def evaluate_topology(
    calc, boom_pos: np.ndarray, labels: list[str],
    seg_to_booms: dict[str, list[int]], flange_indices: list[int],
) -> dict:
    '''
    Apply Megson pair-attribution along each T1 segment using the supplied
    boom-to-segment map. Return a dict with centroid, I_XX, I_ZZ, I_XZ.
    '''
    Xc, Zc     = calc.Xc, calc.Zc
    I_ZZ, I_XZ = calc.I_ZZ, calc.I_XZ
    def sigma(u, w): return I_ZZ * w - I_XZ * u
    T1 = {s['label']: s for s in calc.T1}

    n = len(boom_pos)
    A = np.zeros(n)
    for k, fi in enumerate(flange_indices):
        A[fi] = float(calc.A_flange[k])

    u_arr = boom_pos[:, 0] - Xc
    w_arr = boom_pos[:, 1] - Zc

    for seg_label, boom_idx_list in seg_to_booms.items():
        seg_idx = int(seg_label.replace('seg', '')) - 1   # 0..6
        seg     = T1[seg_label]
        sx, sz  = seg['x'], seg['z']
        t       = float(_T_SEG[seg_idx])

        # Sort booms on this segment by arc-length parameter
        s_b = [arc_param(sx, sz, boom_pos[bi, 0], boom_pos[bi, 1])
               for bi in boom_idx_list]
        order = np.argsort(s_b)
        chain = [boom_idx_list[k] for k in order]
        s_chain = [s_b[k] for k in order]

        # Megson pair on consecutive pairs
        for k in range(len(chain) - 1):
            i_n = chain[k];  i_m = chain[k + 1]
            L_sub = s_chain[k + 1] - s_chain[k]
            tL = t * L_sub
            sn = sigma(u_arr[i_n], w_arr[i_n])
            sm = sigma(u_arr[i_m], w_arr[i_m])
            Bn, Bm = megson_pair(tL, sn, sm)
            A[i_n] += Bn
            A[i_m] += Bm

    sumA = float(np.sum(A))
    Xc_n = float(np.dot(boom_pos[:, 0], A)) / sumA
    Zc_n = float(np.dot(boom_pos[:, 1], A)) / sumA
    du   = boom_pos[:, 0] - Xc_n
    dw   = boom_pos[:, 1] - Zc_n
    return {
        'Xc'  : Xc_n, 'Zc' : Zc_n,
        'I_XX': float(np.dot(dw**2, A)),
        'I_ZZ': float(np.dot(du**2, A)),
        'I_XZ': float(np.dot(du * dw, A)),
        'A'   : A,
    }


# ================================================================================
# Boom-set assemblers
# ================================================================================

def assemble_T2(calc) -> tuple[np.ndarray, list[str], dict, list[int]]:
    '''10-boom T2 baseline.'''
    pos = np.array([calc.B1, calc.B2, calc.B3, calc.B4,
                    calc.P1, calc.P2, calc.P3, calc.P4, calc.P5, calc.P6])
    labels = ['B1','B2','B3','B4','P1','P2','P3','P4','P5','P6']
    iB1,iB2,iB3,iB4,iP1,iP2,iP3,iP4,iP5,iP6 = range(10)
    seg_to_booms = {
        'seg1': [iB2, iP1, iB1],
        'seg2': [iB1, iP5, iB3],
        'seg3': [iB3, iP4],
        'seg4': [iP4, iB4],
        'seg5': [iB4, iP6, iB2],
        'seg6': [iB1, iP2, iB2],
        'seg7': [iB3, iP3, iB4],
    }
    return pos, labels, seg_to_booms, [iB1, iB2, iB3, iB4]


def assemble_T2_with_P78(calc) -> tuple[np.ndarray, list[str], dict, list[int]]:
    '''12-boom topology: T2 plus P7 (upper z-max) and P8 (lower z-min).'''
    P7, P8 = find_z_extrema(calc)
    xw1 = calc.xw1_frac * _CHORD
    xw2 = calc.xw2_frac * _CHORD

    pos = np.array([calc.B1, calc.B2, calc.B3, calc.B4,
                    calc.P1, calc.P2, calc.P3, calc.P4,
                    calc.P5, calc.P6, P7, P8])
    labels = ['B1','B2','B3','B4','P1','P2','P3','P4','P5','P6','P7','P8']
    (iB1,iB2,iB3,iB4,iP1,iP2,iP3,iP4,iP5,iP6,iP7,iP8) = range(12)

    # Default mapping (scenario 1) then patch with P7, P8 hosts
    seg_to_booms = {
        'seg1': [iB2, iP1, iB1],
        'seg2': [iB1, iP5, iB3],
        'seg3': [iB3, iP4],
        'seg4': [iP4, iB4],
        'seg5': [iB4, iP6, iB2],
        'seg6': [iB1, iP2, iB2],
        'seg7': [iB3, iP3, iB4],
    }

    # Dispatch P7 (upper)
    if P7[0] < xw1:
        seg_to_booms['seg1'].append(iP7)
    elif P7[0] < xw2:
        seg_to_booms['seg2'].append(iP7)
    else:
        seg_to_booms['seg3'].append(iP7)

    # Dispatch P8 (lower)
    if P8[0] < xw1:
        seg_to_booms['seg1'].append(iP8)
    elif P8[0] < xw2:
        seg_to_booms['seg5'].append(iP8)
    else:
        seg_to_booms['seg4'].append(iP8)

    return pos, labels, seg_to_booms, [iB1, iB2, iB3, iB4]


# ================================================================================
# Reporting
# ================================================================================

def report(name: str, calc, res: dict) -> None:
    e_xx = (res['I_XX'] / calc.I_XX - 1) * 100
    e_zz = (res['I_ZZ'] / calc.I_ZZ - 1) * 100
    print(f"  {name:<26s}  Xc={res['Xc']:>8.3f}  "
          f"I_XX={res['I_XX']:>8.0f} ({e_xx:>+6.2f}%)  "
          f"I_ZZ={res['I_ZZ']:>10.0f} ({e_zz:>+6.2f}%)")


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # Determine z-extrema on a default-xw1 calc (geometry of the airfoil
    # is independent of xw1, so this fixes P7 / P8 once)
    calc_probe = build_pre_id_calc(_AFL_NAME, 60.0 / _CHORD)
    P7, P8 = find_z_extrema(calc_probe)

    print(f"Airfoil: {_AFL_NAME}")
    print(f"  P7 (upper z-max)  = ({P7[0]:6.2f}, {P7[1]:+6.2f})  -> {P7[0]/_CHORD*100:.1f}% chord")
    print(f"  P8 (lower z-min)  = ({P8[0]:6.2f}, {P8[1]:+6.2f})  -> {P8[0]/_CHORD*100:.1f}% chord")
    print()

    # Pick xw1 values to trigger each scenario.
    x_lo, x_hi = sorted([P7[0], P8[0]])
    scenarios = [
        ('Scenario 1 (xw1 < x_P7 < x_P8)',     0.5  * x_lo / _CHORD),
        ('Scenario 2 (x_P7 < xw1 < x_P8)',     0.5 * (x_lo + x_hi) / _CHORD),
        ('Scenario 3 (x_P7 < x_P8 < xw1)',     1.5  * x_hi / _CHORD),
    ]

    for name, xw1_frac in scenarios:
        xw1_mm = xw1_frac * _CHORD
        print(f"=== {name}    xw1 = {xw1_mm:.1f} mm "
              f"({xw1_frac*100:.1f}% chord) ===")
        calc = build_pre_id_calc(_AFL_NAME, xw1_frac)
        print(f"  Pre-id truth:                            "
              f"I_XX={calc.I_XX:>8.0f}             "
              f"I_ZZ={calc.I_ZZ:>10.0f}")

        # 10-boom T2 baseline
        pos, lbl, m, fl = assemble_T2(calc)
        res = evaluate_topology(calc, pos, lbl, m, fl)
        report("T2 baseline (10 booms)", calc, res)

        # 12-boom T2 + P7 + P8
        pos, lbl, m, fl = assemble_T2_with_P78(calc)
        res = evaluate_topology(calc, pos, lbl, m, fl)
        # also print which seg P7, P8 ended up on
        seg_of = {bi: sl for sl, lst in m.items() for bi in lst}
        print(f"  P7 host seg = {seg_of[lbl.index('P7')]}, "
              f"P8 host seg = {seg_of[lbl.index('P8')]}")
        report("T2 + P7 + P8 (12 booms)", calc, res)
        print()
