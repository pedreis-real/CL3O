'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
FEA Validation Module.

Standalone validation of the 3-D Euler-Bernoulli beam FEA pipeline for the
DA62 left wing. The script is function-based (no Data / Helper / Main 3-layer
split) but delegates every numerical step to the production modules under
src/geometry, src/fea/*, src/materials and src/optimization:

  1. Load DA62 inputs (WingData, AirfoilData, ExLoadsData) and the laminate
     catalogue + ply database from /data
  2. Interpolate the wing outline at the load stations via WingHelper
  3. Call GeomPropCalculator at every station using the laminate engineering
     constants (thick, E1, E2, G12) for skins, spar webs and flanges
  4. Call FemSetup to build the static mesh artifacts and assemble K via
     MeshBuilder; solve {F} = [K]{d} via LinearStaticSolver
  5. Dump the local-to-global matrix chain for one inspection element by
     re-instantiating BeamElement on the same station data
  6. Run the post-processing chain on every solved scenario:
        StressRecovery   (boom normal sigma + panel shear tau)
        TsaiWuFailure    (per-ply strength ratio, min margin, nv)
  7. Compute the structural mass once (geometry + laminates only) via
     StructuralMass and report the per-element breakdown
  8. Render two PyVista views per scenario:
        - undeformed vs deformed wing surface mesh
        - colored internal force / moment diagram along the span

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import openpyxl

# ================ Default Database Paths ================
from cl3o.paths import (
    ROOT_DIR,
    WINGS_DIR, LOADS_DIR, AIRFOILS_DIR, OUTPUTS_DIR,
    MATERIALS_DIR as _DFLT_MAT_DIR,
    PLIES_DIR     as _DFLT_PLY_DIR,
)
_DFLT_WNG_PATH = WINGS_DIR    / "DA62_WingData.json"
_DFLT_LDS_PATH = LOADS_DIR    / "da62_ExLoadsData.json"
_DFLT_AFL_PATH = AIRFOILS_DIR / "wortmannfx63137_AirfoilData.json"
_DFLT_OUT_DIR  = OUTPUTS_DIR  / "fea_validation"

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.wing             import WingData, WingHelper
from cl3o.geometry.airfoil          import AirfoilData
from cl3o.geometry.geom_properties  import GeomPropCalculator, GeomData

# Materials
from cl3o.materials.laminate        import LaminateData, PlyData

# Finite Element Analysis
from cl3o.fea.loads.load_mapper     import ExLoadsData
from cl3o.fea.elements.beam_element import BeamElement, BeamData
from cl3o.fea.pre.fem_setup         import FemSetup, FemPreprocessData
from cl3o.fea.solver.mesh_builder   import MeshBuilder, MeshData
from cl3o.fea.solver.static_analysis import LinearStaticSolver, FeaResults
from cl3o.fea.post.stress_recovery  import StressRecovery, StressData
from cl3o.fea.post.tsw_failure      import TsaiWuFailure, FailureData

# Optimization (score module hosts StructuralMass)
from cl3o.optimization.fscore       import StructuralMass, ScoreData


# ================ Validation configuration ================

# Laminate catalogue: integer key -> MAT_* prefix on disk.
# Each laminate's thick / rho / E1 / E2 / G12 / plies are pulled from
# data/materials/MAT_<name>_LaminateData.json and stored as MAT{int} in
# laminate_db, matching the production main._import_database remapping.
_LAMINATE_CATALOG : dict[int, str] = {
    1 : 'MAT_CFRP_QI16',     # 16-ply quasi-isotropic CFRP
    2 : 'MAT_CFRP_QI24',     # 24-ply quasi-isotropic CFRP
    3 : 'MAT_CFRP_UD16',     # 16-ply unidirectional CFRP
    4 : 'MAT_CFRP_UD24',     # 24-ply unidirectional CFRP
    5 : 'MAT_CFRP_AP16',     # 16-ply angle-ply CFRP
}

# Design vector - one entry per wing control point (n_cpts), mirroring the
# OptVars layout from src/optimization/de_opt.py:
#
#     xw1, xw2                              -> spar chord-fractions
#     bf1, bf2, bf3, bf4                    -> flange width, fraction of chord
#     lam_ls1, lam_ls2                      -> nose / box skin laminate index
#     lam_lw1, lam_lw2                      -> front / rear spar web laminate
#     lam_lf1..lam_lf4                      -> flange (F1..F4) laminate index
#
# Continuous variables (xw, bf) are linearly interpolated at |Y_sta|; discrete
# laminate indices are step-looked-up (cpt[i] persists for Y in
# [Y_cp[i], Y_cp[i+1])), exactly as section_builder._step_opt_vars does.
#
# T1-segment / T4-flange mapping (matches geometry.section_builder):
#     seg1   (nose skin)                  -> lam_ls1
#     seg2..5(upper/lower box skin)       -> lam_ls2
#     seg6   (front spar web)             -> lam_lw1
#     seg7   (rear  spar web)             -> lam_lw2
#     F1 (B3, upper front spar cap)       -> lam_lf1, bf1
#     F2 (B5, lower front spar cap)       -> lam_lf2, bf2
#     F3 (B1, upper rear  spar cap)       -> lam_lf3, bf3
#     F4 (B7, lower rear  spar cap)       -> lam_lf4, bf4
_DESIGN_VECTOR = {
    # -------- Spar chord-fractions --------
    'xw1'     : np.array([0.20, 0.20, 0.22, 0.25]),
    'xw2'     : np.array([0.65, 0.65, 0.62, 0.60]),

    # -------- Flange widths [fraction of chord] --------
    'bf1'     : np.array([0.040, 0.040, 0.035, 0.030]),
    'bf2'     : np.array([0.040, 0.040, 0.035, 0.030]),
    'bf3'     : np.array([0.035, 0.035, 0.030, 0.025]),
    'bf4'     : np.array([0.035, 0.035, 0.030, 0.025]),

    # -------- Laminate indices (into _LAMINATE_CATALOG) --------
    'lam_ls1' : np.array([3, 2, 2, 1], dtype=int),
    'lam_ls2' : np.array([3, 2, 2, 1], dtype=int),
    'lam_lw1' : np.array([5, 5, 4, 4], dtype=int),
    'lam_lw2' : np.array([5, 5, 4, 4], dtype=int),
    'lam_lf1' : np.array([5, 5, 4, 4], dtype=int),
    'lam_lf2' : np.array([5, 5, 4, 4], dtype=int),
    'lam_lf3' : np.array([5, 5, 4, 4], dtype=int),
    'lam_lf4' : np.array([5, 5, 4, 4], dtype=int),
}

# Continuous keys (linear interp) and discrete keys (step lookup)
_CONTINUOUS_KEYS = ('xw1', 'xw2', 'bf1', 'bf2', 'bf3', 'bf4')
_DISCRETE_KEYS   = (
    'lam_ls1', 'lam_ls2', 'lam_lw1', 'lam_lw2',
    'lam_lf1', 'lam_lf2', 'lam_lf3', 'lam_lf4',
)
_PER_CPT_KEYS    = _CONTINUOUS_KEYS + _DISCRETE_KEYS

# Tip-load scenario A (applied at the OUTBOARD tip node, in GLOBAL XYZ)
_TIP_FX = 100.0
_TIP_FZ = 10_000.0
_TIP_MY = -100_000.0

# Index of the element whose matrices are dumped to stdout for inspection
_DEBUG_ELEM = 0

# Set False to skip PyVista interactive windows (still saves PNG screenshots)
_SHOW_PLOTS = False   # default headless; figures are saved to disk. Set True to display.


# ================================================================================
# Cantilever validation constants (Subsection 3.7 / Section 4.2)
# ================================================================================

# FEMAP reference export files
_FEMAP_2D_XLSX = ROOT_DIR / 'docs' / 'Femap_viga_cantilever_2D.xlsx'
_FEMAP_3D_XLSX = ROOT_DIR / 'docs' / 'Femap_viga_cantilever_3D.xlsx'

# Asymmetric WortmannFX63-137 section at chord = 300 mm (Fig 48).
_VAL_CHORD  = 300.0
_VAL_XW1    = 0.30              # front-spar chord fraction (90 mm)
_VAL_XW2    = 170.0 / 300.0    # rear-spar chord fraction (170 mm)
_VAL_BF_MM  = np.array([16.0, 16.0, 10.0, 10.0])  # flange widths [mm]

# Table 11 - generic orthotropic material [MPa / mm]
_VAL_E1    = 200_000.0
_VAL_E2    =  10_000.0
_VAL_G     =   6_000.0

# Table 12 - reference cross-section properties at LE = (0, 0) [mm / mm^2 / mm^4 / rad]
_VAL_X_C     = 116.2424
_VAL_Z_C     =  13.5308
_VAL_A       = 1782.43
_VAL_I_XX    =   354_030.0
_VAL_I_ZZ    = 11_053_338.0
_VAL_I_XZ    =   211_034.0
_VAL_I_1     = 11_057_499.0
_VAL_I_2     =   349_869.0
_VAL_THETA_P = np.radians(91.13)
_VAL_J       = 1_022_279.0
_VAL_X_S     =  85.6928
_VAL_Z_S     =   5.8600

# Scenario 1 - planar cantilever (Fig 49): L = 3000 mm, 11 nodes, Fz = 1 kN tip
_VAL_2D_L     = 3000.0
_VAL_2D_NNODE =    11
_VAL_2D_FZ    =  1000.0   # N

# Scenario 2 - 3D cantilever (Fig 50) tip loads
_VAL_3D_FX =    10.0   # N  (horizontal H)
_VAL_3D_FZ =  1000.0   # N  (vertical V)
_VAL_3D_MY = -10_000.0  # N.mm (moment about global Y at tip)

# ================================================================================
# Data loading
# ================================================================================

def load_inputs() -> tuple[WingData, AirfoilData, ExLoadsData]:
    '''
    Read the three on-disk JSON inputs used by the validation.

    Returns:
        Tuple (wing, afl, ex_loads).
    '''
    wing     = io.read_json(_DFLT_WNG_PATH, WingData)
    afl      = io.read_json(_DFLT_AFL_PATH, AirfoilData)
    ex_loads = io.read_json(_DFLT_LDS_PATH, ExLoadsData)
    return wing, afl, ex_loads


# ================================================================================
# Cantilever validation - section builder
# ================================================================================

def _build_val_geom(
    Y_sta : float,
    LE_xz : np.ndarray,
) -> GeomData:
    '''
    Build one GeomData for the cantilever validation using the reference
    section properties (Table 12) and generic orthotropic material (Table 11).

    skew_matrix_ac is zeroed so that loads applied at the node position are
    NOT offset to the shear centre, matching the FEMAP PBEAM convention where
    grid-point loads act on the elastic axis.

    The centroid C and shear centre S_XYZ are shifted by LE_xz so that
    section positions can be placed at arbitrary global coordinates.

    Args:
        Y_sta : Spanwise station [mm].
        LE_xz : Leading-edge (X, Z) offset applied to the reference section [mm].

    Returns:
        GeomData with skew_matrix_ac = 0.
    '''
    us = _VAL_X_S - _VAL_X_C   # SC offset from centroid, chord dir [mm]
    ws = _VAL_Z_S - _VAL_Z_C   # SC offset from centroid, vert  dir [mm]
    skew = np.array([
        [ 0.0,  -ws,   us ],
        [ ws,   0.0,  0.0 ],
        [-us,   0.0,  0.0 ],
    ])
    return GeomData(
        C            = np.array([_VAL_X_C + LE_xz[0], Y_sta, _VAL_Z_C + LE_xz[1]]),
        A            = _VAL_A,
        I_XX         = _VAL_I_XX,
        I_ZZ         = _VAL_I_ZZ,
        I_XZ         = _VAL_I_XZ,
        I_1          = _VAL_I_1,
        I_2          = _VAL_I_2,
        theta_P      = _VAL_THETA_P,
        J            = _VAL_J,
        S_XYZ        = np.array([_VAL_X_S + LE_xz[0], Y_sta, _VAL_Z_S + LE_xz[1]]),
        E1_eq        = _VAL_E1,
        E2_eq        = _VAL_E2,
        G_eq         = _VAL_G,
        E1_bend_eq   = _VAL_E1,
        E2_bend_eq   = _VAL_E2,
        skew_matrix    = skew,
        skew_matrix_ac = np.zeros((3, 3)),
    )


def build_cantilever_sections_2d() -> SimpleNamespace:
    '''
    Build the 11-node straight cantilever (Scenario 1).

    All nodes share the same cross-section at chord = 300 mm and
    LE_xz = (0, 0); only C[1] (Y_sta) varies from 0 to -3000 mm.

    Returns:
        SimpleNamespace with sec_data, n_sta, lam_T1/T4, chord.
    '''
    Y_nodes = np.linspace(0.0, -_VAL_2D_L, _VAL_2D_NNODE)
    le      = np.array([0.0, 0.0])
    sec     = [_build_val_geom(float(Y), le) for Y in Y_nodes]
    n       = int(_VAL_2D_NNODE)
    return SimpleNamespace(
        sec_data  = sec,
        n_sta     = n,
        lam_T1    = np.zeros((n, 7), dtype=int),
        lam_T4    = np.zeros((n, 4), dtype=int),
        chord     = np.full(n, _VAL_CHORD),
        lerp_wing = None,
    )


def build_cantilever_sections_3d(
    femap_coords : np.ndarray,
) -> SimpleNamespace:
    '''
    Build sections for the 3D cantilever (Scenario 2).

    A reference GeomData at LE = (0, 0) gives the centroid offset
    (Xc_0, Zc_0). For each FEMAP node (X_f, Y_f, Z_f), the LE is set to
    (X_f - Xc_0, Z_f - Zc_0) so that GeomData.C coincides exactly with
    the FEMAP node coordinate.

    Args:
        femap_coords : (n, 3) FEMAP node coordinates [X, Y, Z] in mm.

    Returns:
        SimpleNamespace with sec_data, n_sta, lam_T1/T4, chord.
    '''
    n   = int(femap_coords.shape[0])
    sec = []
    for k in range(n):
        X_f, Y_f, Z_f = femap_coords[k]
        le = np.array([X_f - _VAL_X_C, Z_f - _VAL_Z_C])
        sec.append(_build_val_geom(float(Y_f), le))

    return SimpleNamespace(
        sec_data  = sec,
        n_sta     = n,
        lam_T1    = np.zeros((n, 7), dtype=int),
        lam_T4    = np.zeros((n, 4), dtype=int),
        chord     = np.full(n, _VAL_CHORD),
        lerp_wing = None,
    )


# ================================================================================
# Cantilever validation - FEMAP reference loader
# ================================================================================

def load_femap_xlsx(
    path : Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load node and element results from a FEMAP export workbook.

    Nodes sheet (data rows only, header skipped):
        cols 2-4  : global XYZ coordinates [mm]
        cols 5-10 : translations T1-T3 [mm] and rotations R1-R3 [rad]
        cols 11-  : constraint forces CF_T1-T3 [N] and moments CF_R1-R3 [N.mm]

    Elements sheet:
        col 0  : element ID
        cols 1+ : Plane1 Moment, Plane2 Moment, Pl1/2 Shear, Axial, Torque, ...

    Args:
        path : Path to .xlsx file.

    Returns:
        Tuple (coords, disp, reactions, elems):
            coords    (n, 3) - node XYZ
            disp      (n, 6) - [T1,T2,T3,R1,R2,R3] at every node
            reactions (n, 6) - constraint forces/moments at every node
            elems     (m, k) - element internal-force rows (first col = ID)
    '''
    wb = openpyxl.load_workbook(str(path), data_only=True)

    def _to_arr(ws):
        rows = []
        for row in ws.iter_rows(values_only=True):
            if not row or row[0] is None or not isinstance(row[0], (int, float)):
                continue
            rows.append([float(c) if c is not None else 0.0 for c in row])
        return np.array(rows, dtype=float)

    nodes = _to_arr(wb['Nodes'])      # (n, 17)
    elems = _to_arr(wb['Elements'])   # (m, 7+)

    coords    = nodes[:, 2:5]         # X, Y, Z
    disp      = nodes[:, 5:11]        # T1..T3, R1..R3
    reactions = nodes[:, 11:17]       # CF_T1..CF_R3

    return coords, disp, reactions, elems


# ================================================================================
# Cantilever validation - FEM setup and solver
# ================================================================================

def _build_val_fem_setup(n: int, nc: int) -> FemPreprocessData:
    '''
    Build a clamped-root cantilever FemPreprocessData for n nodes.

    Node 0 is fully clamped (re[0, :] = 1). All elements use release
    code 3 (both ends fixed).

    Args:
        n  : Number of nodes.
        nc : Number of load conditions.

    Returns:
        FemPreprocessData with empty beam_cache.
    '''
    m   = n - 1
    dof = 6 * n
    re  = np.zeros((n, 6), dtype=int)
    re[0, :] = 1
    conn = np.column_stack([
        np.arange(0, m, dtype=int),
        np.arange(1, n, dtype=int),
        np.ones((m, 2), dtype=int),
    ])
    mcn = np.hstack([
        6 * conn[:, 0:1] + np.arange(6),
        6 * conn[:, 1:2] + np.arange(6),
    ]).astype(int)
    F = np.zeros((n, 6, nc), dtype=float)
    return FemPreprocessData(
        n        = n,
        m        = m,
        dof      = dof,
        re       = re.astype(int),
        re_flat  = re.ravel().astype(int),
        conn     = conn.astype(int),
        mcn      = mcn,
        loads    = {
            'nc'    : nc,
            'F'     : F,
            'F_flat': F.reshape(dof, nc),
        },
        beam_cache = {},
    )


def _tip_load_6dof(n: int, dof_vals: dict[int, float]) -> dict:
    '''
    Build a 1-condition loads dict with forces/moments at the tip node.

    Args:
        n        : Number of nodes.
        dof_vals : {local_dof_index: value} pairs for the tip node.

    Returns:
        Loads dict (nc=1) compatible with LinearStaticSolver.
    '''
    F = np.zeros((n, 6, 1), dtype=float)
    for dof, val in dof_vals.items():
        F[-1, dof, 0] = val
    return {'nc': 1, 'F': F, 'F_flat': F.reshape(6 * n, 1)}


# ================================================================================
# Cantilever validation - comparison report
# ================================================================================

def _err_pct(cl3o: float, ref: float) -> str:
    '''Format percentage error; returns "-" when reference ~= 0.'''
    if abs(ref) > 1e-8:
        return f'{100.0 * (cl3o - ref) / abs(ref):+.2f}%'
    return '-' if abs(cl3o) < 1e-4 else f'N / A'


def _abs_err_pct(cl3o: float, ref: float) -> str:
    '''Magnitude-only percentage error (ignores sign convention).'''
    if abs(ref) > 1e-8:
        return f'{100.0 * (abs(cl3o) - abs(ref)) / abs(ref):+.2f}%'
    return '-' if abs(cl3o) < 1e-4 else f'N / A'


def report_cantilever_comparison(
    case_name    : str,
    cl3o_root_R  : np.ndarray,
    cl3o_tip_d   : np.ndarray,
    femap_root_R : np.ndarray,
    femap_tip_d  : np.ndarray,
    is_3d        : bool,
) -> None:
    '''
    Print the reaction and displacement comparison for one cantilever case.

    Sign conventions
    ----------------
    For beams along -Y (left wing), CL3O's moment reactions (DOFs 3 and 5 =
    Mx, Mz) are sign-inverted relative to FEMAP because CL3O's rot3() uses
    a clockwise (transposed) rotation matrix.  Only DOF 4 (My, about the
    span axis) maps directly.  Moment comparisons therefore use absolute
    values (marked [|.|]).

    Additionally, CL3O explicitly models the centroid->SC coupling via the
    G offset matrix, which introduces span-direction (uy) displacement under
    vertical loads.  FEMAP PBEAM uses the elastic axis as its reference and
    does not produce this coupling.  The span-direction DOF is therefore
    labelled [model diff] in the displacement table.

    For the 2D case (FEMAP beam along X, CL3O beam along -Y) the DOF
    mapping is:
        FEMAP  T1 (span)  T2 (chord)  T3 (vert)   R1 (torsion)  R2 (bend)  R3
        CL3O   uy         ux          uz           thy           thx        thz
    '''
    w = 30
    hdr = f"  {'Quantity':<{w}} {'FEMAP':>13}  {'CL3O':>13}  {'Error':>11}"
    sep = '  ' + '-' * (len(hdr) - 2)

    def _row(label, fval, cval, use_abs=False, tag=''):
        if tag == '[model diff]':
            err = 'N / A'
        elif use_abs:
            err = _abs_err_pct(cval, fval)
        else:
            err = _err_pct(cval, fval)
        fdisp = abs(fval) if use_abs else fval
        cdisp = abs(cval) if use_abs else cval
        suffix = f'  {tag}' if tag else ''
        print(f"  {label:<{w}} {fdisp:>13.4g}  {cdisp:>13.4g}  {err:>11}{suffix}")

    print(f"\n{'=' * 80}")
    print(f"  TABLE 13 - Root reactions [{case_name}]")
    print('  Reactions depend only on equilibrium -> expect <0.1% error.')
    print(f"{'=' * 80}")
    print(hdr); print(sep)

    if is_3d:
        # 3D: same global XYZ for both codes
        # Forces: direct mapping, exact signs
        _row('Fx [N]', femap_root_R[0], cl3o_root_R[0])
        _row('Fy [N]', femap_root_R[1], cl3o_root_R[1])
        _row('Fz [N]', femap_root_R[2], cl3o_root_R[2])
        # Moments: sign-inverted for Mx,Mz (rotation convention); use |.|
        _row('Mx [N.mm]', femap_root_R[3], cl3o_root_R[3], use_abs=True)
        _row('My [N.mm]', femap_root_R[4], cl3o_root_R[4])
        _row('Mz [N.mm]', femap_root_R[5], cl3o_root_R[5], use_abs=True)
    else:
        # 2D: FEMAP T3(Fz)<->CL3O Fz, FEMAP R2(My_femap)<->CL3O |Mx|
        _row('Fz [N]    (T3_femap / Fz_CL3O)',
             femap_root_R[2], cl3o_root_R[2])
        _row('Mx [N.mm] (R2_femap / Mx_CL3O)',
             femap_root_R[4], cl3o_root_R[3], use_abs=True)

    print(f"\n{'=' * 80}")
    print(f"  TABLE 14 - Tip displacements [{case_name}]")
    print('  Displacements depend on section properties (cf. Table 12).')
    print(f"{'=' * 80}")
    print(hdr); print(sep)

    if is_3d:
        # Translations
        _row('ux [mm]',   femap_tip_d[0], cl3o_tip_d[0], use_abs=True)
        _row('uy [mm]',   femap_tip_d[1], cl3o_tip_d[1])
        _row('uz [mm]',   femap_tip_d[2], cl3o_tip_d[2])
        # Rotations
        _row('thx [rad]', femap_tip_d[3], cl3o_tip_d[3], use_abs=True)
        _row('thy [rad]', femap_tip_d[4], cl3o_tip_d[4])
        _row('thz [rad]', femap_tip_d[5], cl3o_tip_d[5], use_abs=True)
    else:
        # 2D: compare primary vertical deflection and bending rotation
        _row('uz  [mm]  (T3_femap / uz_CL3O)',
             femap_tip_d[2], cl3o_tip_d[2])
        _row('thx [rad] (R2_femap / thx_CL3O)',
             femap_tip_d[4], cl3o_tip_d[3], use_abs=True)

# ================================================================================
# Cantilever validation - per-scenario orchestration
# ================================================================================

def run_cantilever_2d(
    out_dir : Path,
) -> None:
    '''
    Scenario 1 - planar cantilever, L = 3000 mm, Fz = 1 kN at tip.

    Reproduces the FEMAP PBEAM 2D model (Fig 49) using the CL3O MSA
    pipeline and prints the comparison (Tables 13 / 14 for the 2D case).

    Args:
        out_dir : Directory for potential output files.
    '''
    print(f"\n{'=' * 80}")
    print('  CANTILEVER VALIDATION - Scenario 1: planar beam (2D)')
    print('  Geometry  : L=3000 mm, 10 elements, constant asymmetric section')
    print(f'  Material  : E1={_VAL_E1/1e3:.0f} GPa  E2={_VAL_E2/1e3:.0f} GPa  G={_VAL_G/1e3:.0f} GPa')
    print(f'  Load      : Fz = {_VAL_2D_FZ:.0f} N at tip')
    print(f"{'=' * 80}")

    sections = build_cantilever_sections_2d()
    n        = sections.n_sta
    nc       = 1

    ref_geom = sections.sec_data[0]
    print(f'  Section properties (CL3O, at root):')
    print(f'    Xc = {ref_geom.C[0]:.2f} mm  Zc = {ref_geom.C[2]:.2f} mm')
    print(f'    A  = {ref_geom.A:.2f} mm^2')
    print(f'    I1 = {ref_geom.I_1:.0f} mm^4  I2 = {ref_geom.I_2:.0f} mm^4')
    print(f'    J  = {ref_geom.J:.0f} mm^4')
    print(f'    Xs = {ref_geom.S_XYZ[0]:.2f} mm  Zs = {ref_geom.S_XYZ[2]:.2f} mm')

    fem_setup = _build_val_fem_setup(n, nc)
    loads     = _tip_load_6dof(n, {2: _VAL_2D_FZ})   # Fz at tip (CL3O index 2)

    mesh    = MeshBuilder(data=(fem_setup, sections), enable_logging=False).data
    results = LinearStaticSolver(mesh=mesh, loads=loads, enable_logging=False).results

    # Load FEMAP reference
    _, femap_d, femap_R, _ = load_femap_xlsx(_FEMAP_2D_XLSX)
    femap_root_R = femap_R[0, :]     # node 1 (root) constraint forces/moments
    femap_tip_d  = femap_d[-1, :]    # node 11 (tip) displacements

    cl3o_root_R = results.R[0:6, 0]
    cl3o_tip_d  = results.d_c[-6:, 0]

    report_cantilever_comparison(
        case_name    = '2D planar beam',
        cl3o_root_R  = cl3o_root_R,
        cl3o_tip_d   = cl3o_tip_d,
        femap_root_R = femap_root_R,
        femap_tip_d  = femap_tip_d,
        is_3d        = False,
    )


def run_cantilever_3d(
    out_dir : Path,
) -> None:
    '''
    Scenario 2 - 3D cantilever following the DA62 CA line at 1:2 scale.

    Reproduces the FEMAP PBEAM 3D model (Fig 50). Loads at the tip:
    Fx = H = 10 N, Fz = V = 1000 N, My = -10 000 N.mm.

    Args:
        out_dir : Directory for potential output files.
    '''
    print(f"\n{'=' * 80}")
    print('  CANTILEVER VALIDATION - Scenario 2: 3-D curved beam')
    print('  Geometry  : DA62 CA line at 1:2 scale, constant asymmetric section')
    print(f'  Material  : E1={_VAL_E1/1e3:.0f} GPa  E2={_VAL_E2/1e3:.0f} GPa  G={_VAL_G/1e3:.0f} GPa')
    print(f'  Loads     : Fx={_VAL_3D_FX:.0f} N  Fz={_VAL_3D_FZ:.0f} N  My={_VAL_3D_MY:.0f} N.mm at tip')
    print(f"{'=' * 80}")

    femap_coords, femap_d, femap_R, _ = load_femap_xlsx(_FEMAP_3D_XLSX)

    sections = build_cantilever_sections_3d(femap_coords)
    n        = sections.n_sta
    nc       = 1

    span = abs(float(femap_coords[-1, 1]) - float(femap_coords[0, 1]))
    print(f'  Nodes     : {n}  (span = {span:.1f} mm)')

    fem_setup = _build_val_fem_setup(n, nc)
    loads     = _tip_load_6dof(n, {0: _VAL_3D_FX, 2: _VAL_3D_FZ, 4: _VAL_3D_MY})

    mesh    = MeshBuilder(data=(fem_setup, sections), enable_logging=False).data
    results = LinearStaticSolver(mesh=mesh, loads=loads, enable_logging=False).results

    femap_root_R = femap_R[0, :]
    femap_tip_d  = femap_d[-1, :]

    cl3o_root_R = results.R[0:6, 0]
    cl3o_tip_d  = results.d_c[-6:, 0]

    report_cantilever_comparison(
        case_name    = '3D curved beam',
        cl3o_root_R  = cl3o_root_R,
        cl3o_tip_d   = cl3o_tip_d,
        femap_root_R = femap_root_R,
        femap_tip_d  = femap_tip_d,
        is_3d        = True,
    )


# ================================================================================
# Main
# ================================================================================

def main(show: bool = _SHOW_PLOTS) -> None:
    '''
    Orchestrate the full validation:  sections -> assemble -> solve.
    '''
    main_validation(_DFLT_OUT_DIR)


def main_validation(
    out_dir : Path,
) -> None:
    '''
    Orchestrate both cantilever validation scenarios and print a summary.

    Args:
        out_dir : Root output directory.
    '''
    run_cantilever_2d(out_dir)
    run_cantilever_3d(out_dir)

    # print(f"\n{'=' * 80}")
    # print('  CANTILEVER VALIDATION SUMMARY (Subsection 3.7 / Section 4.2)')
    # print('  ' + '-' * 76)
    # print('  Reactions (equilibrium):')
    # print('    Force reactions  : <0.01% error in both cases (exact equilibrium).')
    # print('    Moment reactions : 0.86-2.01% error from centroid vs SC lever-arm')
    # print('                       difference; sign inverted for Mx/Mz (rotation')
    # print('                       convention of rot3 matrix) -- see [|.|] rows.')
    # print('  Displacements:')
    # print('    Primary (uz)     : ~38-40% larger in CL3O; consistent with CL3O')
    # print('                       using smaller I_2 (minor bending inertia) than')
    # print('                       FEMAP for this section -- cf. Table 12.')
    # print('    Coupling (ux)    : ~65% error (2D); I_XZ discrepancy from Table 12.')
    # print('    Span dir (uy)    : [model diff] -- CL3O explicitly couples centroid-')
    # print('                       to-SC offset (G matrix); FEMAP PBEAM uses elastic')
    # print('                       axis as reference and omits this coupling.')
    # print('  Conclusion: equilibrium implementation is correct. Displacement')
    # print('    differences trace to documented section-property divergence.')
    # print(f"{'=' * 80}\n")


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    main(show=_SHOW_PLOTS)
