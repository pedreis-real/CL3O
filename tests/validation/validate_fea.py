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

# ================ Default Database Paths ================
from cl3o.paths import (
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
from cl3o.fea.pre.fem_setup         import FemSetup
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


def load_laminate_catalog(
    catalog : dict[int, str] = _LAMINATE_CATALOG,
) -> tuple[dict[str, LaminateData], dict[str, PlyData]]:
    '''
    Load the catalogue laminates and every ply they reference.

    The returned laminate_db is keyed by 'MAT{k}' (integer string), matching
    the production main._import_database convention used by Tsai-Wu and the
    mass score; the ply_db is keyed by the bare ply name (as stored in
    LaminateData.plies).

    Args:
        catalog : Integer-to-laminate-name dict (see _LAMINATE_CATALOG).

    Returns:
        Tuple (laminate_db, ply_db).
    '''
    laminate_db : dict[str, LaminateData] = {}
    ply_names   : set[str] = set()
    for k, name in catalog.items():
        lam = io.read_json(
            filepath = _DFLT_MAT_DIR / f"{name}_LaminateData.json",
            dcls     = LaminateData,
        )
        laminate_db[f"MAT{int(k)}"] = lam
        ply_names.update(lam.plies)

    ply_db : dict[str, PlyData] = {}
    for p in sorted(ply_names):
        ply_db[p] = io.read_json(
            filepath = _DFLT_PLY_DIR / f"PlyData_{p}.json",
            dcls     = PlyData,
        )
    return laminate_db, ply_db


# ================================================================================
# Cross-section build - one GeomData per spanwise station
# ================================================================================

def validate_design_vector(
    dvec   : dict,
    n_cpts : int,
) -> None:
    '''
    Sanity-check the per-cpt design vector before it reaches the section
    builder.

    Args:
        dvec   : Design vector dict (see _DESIGN_VECTOR).
        n_cpts : Number of wing control points (matches wing.n_cpts).

    Raises:
        ValueError if any per-cpt array length differs from n_cpts.
    '''
    for k in _PER_CPT_KEYS:
        arr = np.asarray(dvec[k], dtype=float).ravel()
        if arr.size != n_cpts:
            raise ValueError(
                f"[CL3O] Design vector key '{k}' has length {arr.size}, "
                f"expected {n_cpts} (one entry per wing control point).\n"
                f"Edit _DESIGN_VECTOR in {__file__}."
            )


def _interp_cpt(
    y_abs : float,
    y_cp  : np.ndarray,
    arr   : np.ndarray,
) -> float:
    '''Linear interpolation of a per-cpt continuous design variable at |Y|.'''
    return float(np.interp(y_abs, y_cp, np.asarray(arr, dtype=float)))


def _step_cpt(
    y_abs : float,
    y_cp  : np.ndarray,
    arr   : np.ndarray,
) -> int:
    '''
    Piecewise-constant lookup of a per-cpt discrete variable at |Y|.

    Matches section_builder._cp_index: the value at Y_cp[i] persists for all
    Y_sta in [Y_cp[i], Y_cp[i+1]). The index is clipped to [0, n_cpts - 2],
    so the last cpt entry is never selected (consistent with the production
    SectionBuilder).
    '''
    idx = int(np.searchsorted(y_cp, y_abs, side='right')) - 1
    idx = int(np.clip(idx, 0, len(y_cp) - 2))
    return int(arr[idx])


def build_sections(
    wing        : WingData,
    afl         : AirfoilData,
    Y_sta       : np.ndarray,
    dvec        : dict,
    laminate_db : dict[str, LaminateData],
) -> SimpleNamespace:
    '''
    Build one GeomData per left-wing station via GeomPropCalculator and pack
    them, together with the per-station laminate-index arrays consumed by
    StressRecovery / TsaiWuFailure / StructuralMass, into a sections-like
    namespace.

    Continuous keys (xw1, xw2, bf1..bf4) are linearly interpolated at
    |Y_sta|; discrete laminate-index keys (lam_*) are step-looked-up using
    wing.pos as the cpt grid. The laminate engineering constants (thick,
    E1, E2, G12) are then resolved per segment / flange from the supplied
    laminate_db.

    Args:
        wing        : DA62 WingData (right-half geometry, root -> tip).
        afl         : Wortmann FX63137 airfoil profile.
        Y_sta       : Spanwise stations (left wing, root -> tip, Y < 0).
        dvec        : Design-vector dict (see _DESIGN_VECTOR).
        laminate_db : Dict mapping 'MAT{k}' to LaminateData.

    Returns:
        SimpleNamespace with:
            sec_data    : list[GeomData]                    (one per station)
            lerp_wing   : LerpWingData built from Y_sta
            chord       : (n_sta,) chord array
            lam_T1      : (n_sta, 7) integer per-segment laminate indices
            lam_T4      : (n_sta, 4) integer per-flange  laminate indices
    '''
    validate_design_vector(dvec, int(wing.n_cpts))
    lerp = WingHelper.lerp_from_data(wing, Y_sta)
    y_cp = np.asarray(wing.pos, dtype=float)

    afl_pts = (
        np.asarray(afl.x_upper, dtype=float),
        np.asarray(afl.y_upper, dtype=float),
        np.asarray(afl.x_lower, dtype=float),
        np.asarray(afl.y_lower, dtype=float),
    )

    def _props(
        lam_idx: int,
    ) -> tuple[float, float, float, float, float, float]:
        '''(thick, E1, E2, G12, E1_bend, E2_bend) for laminate MAT{lam_idx}.'''
        lam = laminate_db[f"MAT{int(lam_idx)}"]
        return (
            float(lam.thick),
            float(lam.E1),
            float(lam.E2),
            float(lam.G12),
            float(lam.E1_bend),
            float(lam.E2_bend),
        )

    n_sta : int = int(lerp.n_sta)
    sec       : list[GeomData] = []
    lam_T1_all = np.zeros((n_sta, 7), dtype=int)
    lam_T4_all = np.zeros((n_sta, 4), dtype=int)

    for k, Y in enumerate(lerp.Y_sta):
        y_abs   = abs(float(Y))
        chord_k = float(lerp.chord[k])

        # -------- Continuous interpolation --------
        xw1_k = _interp_cpt(y_abs, y_cp, dvec['xw1'])
        xw2_k = _interp_cpt(y_abs, y_cp, dvec['xw2'])
        bf = np.array([
            _interp_cpt(y_abs, y_cp, dvec['bf1']) * chord_k,
            _interp_cpt(y_abs, y_cp, dvec['bf2']) * chord_k,
            _interp_cpt(y_abs, y_cp, dvec['bf3']) * chord_k,
            _interp_cpt(y_abs, y_cp, dvec['bf4']) * chord_k,
        ])

        # -------- Discrete step lookup (laminate indices) --------
        lam_ls1 = _step_cpt(y_abs, y_cp, dvec['lam_ls1'])
        lam_ls2 = _step_cpt(y_abs, y_cp, dvec['lam_ls2'])
        lam_lw1 = _step_cpt(y_abs, y_cp, dvec['lam_lw1'])
        lam_lw2 = _step_cpt(y_abs, y_cp, dvec['lam_lw2'])
        lam_lf  = np.array([
            _step_cpt(y_abs, y_cp, dvec['lam_lf1']),
            _step_cpt(y_abs, y_cp, dvec['lam_lf2']),
            _step_cpt(y_abs, y_cp, dvec['lam_lf3']),
            _step_cpt(y_abs, y_cp, dvec['lam_lf4']),
        ], dtype=int)

        lam_T1 = np.array(
            [lam_ls1, lam_ls2, lam_ls2, lam_ls2, lam_ls2, lam_lw1, lam_lw2],
            dtype=int,
        )
        lam_T1_all[k] = lam_T1
        lam_T4_all[k] = lam_lf

        # -------- Resolve laminate engineering constants --------
        T1 = np.array([_props(i) for i in lam_T1])         # (7, 6)
        T4 = np.array([_props(i) for i in lam_lf])         # (4, 6)
        t_seg,  E1_seg,  E2_seg,  G_seg,  E1b_seg,  E2b_seg  = (
            T1[:, j] for j in range(6)
        )
        t_fln,  E1_fln,  E2_fln,  G_fln,  E1b_fln,  E2b_fln  = (
            T4[:, j] for j in range(6)
        )

        calc = GeomPropCalculator(
            afl_pts        = afl_pts,
            chord          = chord_k,
            twist          = float(np.degrees(lerp.twist[k])),
            Y_sta          = float(Y),
            xw1            = xw1_k,
            xw2            = xw2_k,
            T1_props       = (t_seg,  E1_seg,  E2_seg,  G_seg,
                              E1b_seg, E2b_seg),
            T4_props       = (t_fln,  E1_fln,  E2_fln,  G_fln,  bf,
                              E1b_fln, E2b_fln),
            LE_xz          = lerp.LE[k, [0, 2]],
            enable_logging = False,
        )
        sec.append(calc.run())

    return SimpleNamespace(
        sec_data  = sec,
        lerp_wing = lerp,
        chord     = lerp.chord,
        n_sta     = n_sta,
        lam_T1    = lam_T1_all,
        lam_T4    = lam_T4_all,
    )


def print_design_vector_summary(
    wing        : WingData,
    dvec        : dict,
    laminate_db : dict[str, LaminateData],
) -> None:
    '''Tabular dump of every per-cpt entry + laminate catalogue legend.'''
    table = {
        'Y_cp [mm]' : np.asarray(wing.pos, dtype=float),
        'chord [mm]': np.asarray(wing.chords, dtype=float),
    }
    for k in _PER_CPT_KEYS:
        table[k] = np.asarray(dvec[k], dtype=float).ravel()

    df = pd.DataFrame(table).T
    df.columns = [f"cpt{i}" for i in range(int(wing.n_cpts))]
    print("\n  Design vector (per wing control point) :")
    with pd.option_context(
        'display.float_format', '{:,.3f}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print(df.to_string())

    print("\n  Laminate catalogue (resolved):")
    print(f"  {'key':<6} {'name':<22} {'t[mm]':>7} "
          f"{'rho[t/mm^3]':>14} {'E1[MPa]':>10} {'E2[MPa]':>10} {'G12[MPa]':>10}")
    for k, name in _LAMINATE_CATALOG.items():
        lam = laminate_db[f"MAT{k}"]
        print(f"  MAT{k:<3} {name:<22} {lam.thick:>7.3f} "
              f"{lam.rho:>14.3e} {lam.E1:>10.0f} {lam.E2:>10.0f} {lam.G12:>10.0f}")


# ================================================================================
# Load vectors for the two scenarios
# ================================================================================

def loads_dict_tip(
    n  : int,
    fx : float,
    fz : float,
    my : float,
) -> dict:
    '''
    Pack a Scenario-A loads dict shaped exactly like FemSetup.loads.

    Args:
        n  : Number of nodes.
        fx : X-direction tip force [N].
        fz : Z-direction tip force [N].
        my : Y-axis tip moment    [N*mm].

    Returns:
        Dict with keys nc, F, F_flat compatible with LinearStaticSolver.
    '''
    F = np.zeros((n, 6, 1), dtype=float)
    F[-1, 0, 0] = fx
    F[-1, 2, 0] = fz
    F[-1, 4, 0] = my
    return {
        'nc'    : 1,
        'F'     : F,
        'F_flat': F.reshape(6 * n, 1),
    }


def loads_dict_from_fem(fem_setup) -> dict:
    '''
    Return the FemSetup-built distributed loads dictionary verbatim.

    Args:
        fem_setup : FemPreprocessData from FemSetup.

    Returns:
        Same dict instance carried inside fem_setup.loads (Scenario B).
    '''
    return fem_setup.loads


# ================================================================================
# Debug element - re-instantiate BeamElement to expose every BeamData field
# ================================================================================

def debug_beam_element(
    sections : SimpleNamespace,
    coord    : np.ndarray,
    conn     : np.ndarray,
    elem_idx : int,
) -> BeamData:
    '''
    Re-run BeamElement on one element so that all 12x12 matrices in the
    local -> sectional -> global chain are captured in a BeamData instance.

    Args:
        sections : SimpleNamespace with sec_data (list of GeomData).
        coord    : (n, 3) MeshData.coord.
        conn     : (m, 4) MeshData.conn.
        elem_idx : Element index to inspect.

    Returns:
        BeamData populated for that single element.
    '''
    conn_i   = conn[elem_idx]
    geomA    = sections.sec_data[conn_i[0]]
    geomB    = sections.sec_data[conn_i[1]]
    rls_code = 2 * int(conn_i[2]) + int(conn_i[3])
    return BeamElement(
        geomA          = geomA,
        geomB          = geomB,
        coord_vector   = coord[conn_i[1]] - coord[conn_i[0]],
        release_type   = rls_code,
        enable_logging = False,
    ).data


# ================================================================================
# Print helpers - pandas, fixed-point 2 decimal places
# ================================================================================

def _print_matrix(name: str, M: np.ndarray) -> None:
    '''
    Pretty-print a 1-D or 2-D matrix with pandas, fixed-point 2-decimal
    formatting (no scientific notation).
    '''
    print(f"\n  {name}  (shape {np.asarray(M).shape})")
    arr = np.asarray(M, dtype=float)
    if arr.ndim == 1:
        df = pd.DataFrame(arr.reshape(1, -1))
    else:
        df = pd.DataFrame(arr)
    with pd.option_context(
        'display.float_format', '{:,.2f}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print(df.to_string(index=False, header=False))


# ---- DOF labels and units ----
_DOF12_DISP_LABELS = (
    'u_A', 'v_A', 'w_A', 'rx_A', 'ry_A', 'rz_A',
    'u_B', 'v_B', 'w_B', 'rx_B', 'ry_B', 'rz_B',
)
_DOF12_DISP_UNITS = (
    'mm', 'mm', 'mm', 'deg', 'deg', 'deg',
    'mm', 'mm', 'mm', 'deg', 'deg', 'deg',
)
_DOF12_DISP_ROT_IDX = np.array([3, 4, 5, 9, 10, 11], dtype=int)

_DOF12_FORCE_LABELS = (
    'Fx_A', 'Fy_A', 'Fz_A', 'Mx_A', 'My_A', 'Mz_A',
    'Fx_B', 'Fy_B', 'Fz_B', 'Mx_B', 'My_B', 'Mz_B',
)
_DOF12_FORCE_UNITS = (
    'N', 'N', 'N', 'N*mm', 'N*mm', 'N*mm',
    'N', 'N', 'N', 'N*mm', 'N*mm', 'N*mm',
)

_DOF6_DISP_LABELS  = ('u', 'v', 'w', 'rx', 'ry', 'rz')
_DOF6_DISP_UNITS   = ('mm', 'mm', 'mm', 'deg', 'deg', 'deg')
_DOF6_DISP_ROT_IDX = np.array([3, 4, 5], dtype=int)

_DOF6_FORCE_LABELS = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')
_DOF6_FORCE_UNITS  = ('N', 'N', 'N', 'N*mm', 'N*mm', 'N*mm')


def _print_labeled_vector(
    name       : str,
    vec        : np.ndarray,
    labels     : tuple[str, ...],
    units      : tuple[str, ...],
    dec_places : int = 4,
    rad_to_deg : np.ndarray | None = None,
) -> None:
    '''
    Print a 1-D vector as a labelled column: one component per row, with
    the component name on the left and the unit on the right.

    Args:
        name       : Header text printed above the table.
        vec        : (N,) values to display.
        labels     : Per-row component names (length N).
        units      : Per-row unit strings (length N).
        rad_to_deg : Indices whose values are stored in radians and should
            be converted to degrees before printing. Pass None to skip.
    '''
    arr = np.asarray(vec, dtype=float).copy()
    if rad_to_deg is not None and rad_to_deg.size > 0:
        arr[rad_to_deg] = np.degrees(arr[rad_to_deg])
    df = pd.DataFrame({'value': arr, 'unit': units}, index=list(labels))
    df.index.name = 'dof'
    print(f"\n  {name}")
    with pd.option_context(
        'display.float_format', f'{{:,.{dec_places}f}}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print(df.to_string())


def report_matrices(
    scenario  : str,
    bd        : BeamData,
    mesh      : MeshData,
    results   : FeaResults,
    elem_idx  : int = _DEBUG_ELEM,
    cond_idx  : int = 0,
) -> None:
    '''
    Dump the FEA matrix chain produced by BeamElement / MeshBuilder /
    LinearStaticSolver for one inspection element + condition.
    '''
    n_a, n_b = mesh.conn[elem_idx, 0], mesh.conn[elem_idx, 1]
    print("=" * 80)
    print(f"  FEA MATRIX REPORT  -  scenario {scenario}")
    print(f"  element {elem_idx} : nodes ({n_a}, {n_b}), L = {bd.L:.2f} mm")
    print(f"  EA  = {bd.EA:.3e} N")
    print(f"  EIy = {bd.EIy:.3e} N*mm^2   (bending about local y, minor I_2)")
    print(f"  EIz = {bd.EIz:.3e} N*mm^2   (bending about local z, major I_1)")
    print(f"  GJ  = {bd.GJ:.3e} N*mm^2   (torsion)")
    print("=" * 80)

    _print_matrix("k (LOCAL stiffness at centroid)",                    bd.k)
    _print_matrix("G (rigid offset C->SC)",                             bd.Gmatrix)
    _print_matrix("k_sc (LOCAL stiffness at SC = G^T k G)",             bd.k_sc)
    # _print_matrix("M (member-release matrix)",                          bd.Mmatrix)
    # _print_matrix("k_sc_r (LOCAL released stiffness at SC = M k_sc)",   bd.k_sc_r)
    _print_matrix("R_a (3x3 azimuth rotation about w)",                 bd.R_a)
    _print_matrix("R_b (3x3 elevation rotation about y')",              bd.R_b)
    _print_matrix("R_c (3x3 web-angle rotation about x'')",             bd.R_c)
    _print_matrix("T (12x12 DOF rotation = kron(I_4, R_c R_b R_a))",    bd.Rmatrix)
    _print_matrix("k_gl (GLOBAL stiffness = T k_sc_r T^T)",             bd.k_gl)

    print("\n  Global stiffness K summary :")
    print(f"    shape       : {mesh.K.shape}")
    print(f"    symmetry    : max|K - K^T| = {float(np.max(np.abs(mesh.K - mesh.K.T))):.3e}")
    print(f"    non-zeros   : {int(np.count_nonzero(mesh.K))} / {mesh.K.size} "
          f"({100.0 * np.count_nonzero(mesh.K) / mesh.K.size:.2f} %)")

    # Per-element LOCAL displacement: d_loc = T @ d_glob[adr]
    adr   = mesh.adr[:, elem_idx]
    T_mat = bd.Rmatrix
    d_e   = results.d[adr, cond_idx]
    d_loc = T_mat @ d_e
    Q_sc  = mesh.T_sc[:, :, elem_idx] @ d_e
    Q_sc_gl = mesh.T_sc_gl[:, :, elem_idx] @ d_e
    Q_c  = mesh.T_c[:, :, elem_idx] @ d_e
    Q_c_gl = mesh.T_c_gl[:, :, elem_idx] @ d_e

    _print_labeled_vector(
        "d_glob[:, elem]  (GLOBAL displacement at the element)",
        d_e,
        labels     = _DOF12_DISP_LABELS,
        units      = _DOF12_DISP_UNITS,
        rad_to_deg = _DOF12_DISP_ROT_IDX,
        dec_places = 4,
    )
    _print_labeled_vector(
        "d_loc [:, elem]  (LOCAL  displacement at the element)",
        d_loc,
        labels     = _DOF12_DISP_LABELS,
        units      = _DOF12_DISP_UNITS,
        rad_to_deg = _DOF12_DISP_ROT_IDX,
        dec_places = 4,
    )
    _print_labeled_vector(
        "Q_sc [:, elem]  (LOCAL  internal forces & moments in SC)",
        Q_sc,
        labels     = _DOF12_FORCE_LABELS,
        units      = _DOF12_FORCE_UNITS,
        dec_places = 0,
    )
    _print_labeled_vector(
        "Q_sc_gl  [:, elem]  (GLOBAL internal forces & moments in SC)",
        Q_sc_gl,
        labels     = _DOF12_FORCE_LABELS,
        units      = _DOF12_FORCE_UNITS,
        dec_places = 0,
    )
    _print_labeled_vector(
        "Q_c [:, elem]  (LOCAL  internal forces & moments in C)",
        Q_c,
        labels     = _DOF12_FORCE_LABELS,
        units      = _DOF12_FORCE_UNITS,
        dec_places = 0,
    )
    _print_labeled_vector(
        "Q_c_gl  [:, elem]  (GLOBAL internal forces & moments in C)",
        Q_c_gl,
        labels     = _DOF12_FORCE_LABELS,
        units      = _DOF12_FORCE_UNITS,
        dec_places = 0,
    )

    print("\n  Nodal displacement (GLOBAL XYZ) - tip node :")
    _print_labeled_vector(
        "d_tip",
        results.d[-6:, cond_idx],
        labels     = _DOF6_DISP_LABELS,
        units      = _DOF6_DISP_UNITS,
        rad_to_deg = _DOF6_DISP_ROT_IDX,
        dec_places = 4,
    )
    print("\n  Reaction at the clamped root (GLOBAL XYZ) :")
    _print_labeled_vector(
        "R_root",
        results.R[:6, cond_idx],
        labels     = _DOF6_FORCE_LABELS,
        units      = _DOF6_FORCE_UNITS,
        dec_places = 0,
    )


# ================================================================================
# Post-processing validation - stresses, Tsai-Wu failure, structural mass
# ================================================================================

def compute_stresses(
    sections : SimpleNamespace,
    mesh     : MeshData,
    results  : FeaResults,
) -> StressData:
    '''Run the production StressRecovery on the solved FEA results.'''
    return StressRecovery(
        sections       = sections,
        element_idx    = mesh.conn,
        fea_results    = results,
        enable_logging = False,
    ).data


def compute_tsw_failure(
    sections    : SimpleNamespace,
    stress      : StressData,
    mesh        : MeshData,
    laminate_db : dict[str, LaminateData],
    ply_db      : dict[str, PlyData],
) -> FailureData:
    '''
    Run the production TsaiWuFailure evaluator on the recovered stresses.

    Builds the (static_data, runtime_data) tuple TsaiWuFailure expects by
    wrapping the loose validate_fea objects in SimpleNamespace.
    '''
    static_ns = SimpleNamespace(
        laminate_db = laminate_db,
        ply_db      = ply_db,
    )
    runtime_ns = SimpleNamespace(
        sections = SimpleNamespace(
            lam_T1 = sections.lam_T1,
            lam_T4 = sections.lam_T4,
        ),
        stress   = stress,
        mesh     = SimpleNamespace(
            conn = mesh.conn,
            n    = mesh.n,
            m    = mesh.m,
            nc   = mesh.nc,
        ),
    )
    return TsaiWuFailure(
        data           = (static_ns, runtime_ns),
        enable_logging = False,
    ).data


def compute_mass(
    sections    : SimpleNamespace,
    mesh        : MeshData,
    laminate_db : dict[str, LaminateData],
) -> ScoreData:
    '''Run the production StructuralMass evaluator on the built sections.'''
    return StructuralMass(
        sections       = sections,
        element_idx    = mesh.conn,
        laminate_db    = laminate_db,
        enable_logging = False,
    ).data


def report_stress(
    stress   : StressData,
    elem_idx : int = _DEBUG_ELEM,
    cond_idx : int = 0,
) -> None:
    '''
    Print the stress state at the debug element for both ends:
        - per-boom normal stress sigma (N_BOOMS = 7)
        - per-panel shear flow q and shear stress tau (N_PANELS = 10)
    '''
    print("\n" + "=" * 80)
    print(f"  STRESS RECOVERY  -  element {elem_idx}, condition {cond_idx}")
    print("=" * 80)

    sigmaA = stress.sigma[0][elem_idx, :, cond_idx]
    sigmaB = stress.sigma[1][elem_idx, :, cond_idx]
    qA     = stress.q    [0][elem_idx, :, cond_idx]
    qB     = stress.q    [1][elem_idx, :, cond_idx]
    tauA   = stress.tau  [0][elem_idx, :, cond_idx]
    tauB   = stress.tau  [1][elem_idx, :, cond_idx]

    df_sig = pd.DataFrame({
        'endA sigma [MPa]' : sigmaA,
        'endB sigma [MPa]' : sigmaB,
    }, index=[f"B{i+1}" for i in range(sigmaA.size)])
    df_pan = pd.DataFrame({
        'endA q [N/mm]'    : qA,
        'endB q [N/mm]'    : qB,
        'endA tau [MPa]'   : tauA,
        'endB tau [MPa]'   : tauB,
    }, index=[f"panel {i+1:>2d}" for i in range(qA.size)])

    with pd.option_context(
        'display.float_format', '{:,.2f}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print("\n  Boom normal stresses :")
        print(df_sig.to_string())
        print("\n  Panel shear flow / shear stress :")
        print(df_pan.to_string())

    sigma_max = float(max(
        np.max(np.abs(stress.sigma[0])),
        np.max(np.abs(stress.sigma[1])),
    ))
    tau_max = float(max(
        np.max(np.abs(stress.tau[0])),
        np.max(np.abs(stress.tau[1])),
    ))
    print(f"\n  Global max |sigma| : {sigma_max:.2f} MPa")
    print(f"  Global max |tau|   : {tau_max:.2f} MPa")


def report_failure(
    failure  : FailureData,
    elem_idx : int = _DEBUG_ELEM,
    cond_idx : int = 0,
) -> None:
    '''
    Print the Tsai-Wu strength ratio R and margin of safety MS at the
    debug element for both ends, plus the global aggregates.
    '''
    print("\n" + "=" * 80)
    print(f"  TSAI-WU FAILURE  -  element {elem_idx}, condition {cond_idx}")
    print("=" * 80)

    R_pan_A  = failure.R_panels [elem_idx, :, 0, cond_idx]
    R_pan_B  = failure.R_panels [elem_idx, :, 1, cond_idx]
    R_bm_A   = failure.R_booms  [elem_idx, :, 0, cond_idx]
    R_bm_B   = failure.R_booms  [elem_idx, :, 1, cond_idx]
    MS_pan_A = failure.MS_panels[elem_idx, :, 0, cond_idx]
    MS_pan_B = failure.MS_panels[elem_idx, :, 1, cond_idx]
    MS_bm_A  = failure.MS_booms [elem_idx, :, 0, cond_idx]
    MS_bm_B  = failure.MS_booms [elem_idx, :, 1, cond_idx]

    df_pan = pd.DataFrame({
        'R  endA' : R_pan_A,
        'R  endB' : R_pan_B,
        'MS endA' : MS_pan_A,
        'MS endB' : MS_pan_B,
    }, index=[f"panel{i+1:>2d}" for i in range(R_pan_A.size)])
    df_bm = pd.DataFrame({
        'R  endA' : R_bm_A,
        'R  endB' : R_bm_B,
        'MS endA' : MS_bm_A,
        'MS endB' : MS_bm_B,
    }, index=[f"B{i+1}" for i in range(R_bm_A.size)])

    with pd.option_context(
        'display.float_format', '{:,.3f}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print("\n  Panels (T2 segments) :")
        print(df_pan.to_string())
        print("\n  Booms (T4 flanges; B2/B4/B6 have no flange -> +inf) :")
        print(df_bm.to_string())

    print(f"\n  Global R_min  : {failure.R_min:.4f}")
    print(f"  Global MS_min : {failure.MS_min:.4f}")
    print(f"  Plies with R < 1 (nv) : {failure.nv}")


def report_mass(score: ScoreData) -> None:
    '''Print the total structural mass plus a per-element breakdown.'''
    print("\n" + "=" * 80)
    print(f"  STRUCTURAL MASS  ({score.m} elements)")
    print("=" * 80)
    print(f"  panels  total : {float(np.sum(score.panels )):>10.4f} kg")
    print(f"  flanges total : {float(np.sum(score.flanges)):>10.4f} kg")
    print(f"  WING    TOTAL : {score.total:>10.4f} kg")

    df = pd.DataFrame({
        'panels [kg]'  : score.panels,
        'flanges [kg]' : score.flanges,
        'total  [kg]'  : score.per_elem,
    })
    df.index = [f"elem{i:>2d}" for i in range(score.m)]
    with pd.option_context(
        'display.float_format', '{:,.4f}'.format,
        'display.max_columns', None,
        'display.width',       200,
    ):
        print("\n  Per-element breakdown :")
        print(df.to_string())


# ================================================================================
# Matplotlib visualization - 2-D cross-section at the debug element
# ================================================================================

def plot_cross_section(
    geom     : GeomData,
    title    : str,
    out_path : Path,
    show     : bool = True,
) -> None:
    '''
    Matplotlib 2-D rendering of one cross-section in the global XZ frame.

    Args:
        geom     : Populated GeomData (output of GeomPropCalculator.run).
        title    : Figure title.
        out_path : PNG output path.
        show     : Pop the matplotlib window after saving.

    Plotted overlays:
        skin panels (T1 seg1..seg5)         - thin black polylines
        spar webs   (T1 seg6, seg7)         - thick black polylines
        booms       (B1..B7)                - filled green circles
        centroid                            - gold diamond
        shear centre                        - green star
    '''
    import matplotlib.pyplot as plt

    T1 = {s['label']: s for s in geom.T1}

    fig, ax = plt.subplots(figsize=(12, 4))

    for label in ('seg1', 'seg2', 'seg3', 'seg4', 'seg5'):
        pts = T1[label]['pts']
        ax.plot(pts[:, 0], pts[:, 1],
                color='#2a2b2c', linewidth=1.2, zorder=2)

    for label in ('seg6', 'seg7'):
        pts = T1[label]['pts']
        ax.plot(pts[:, 0], pts[:, 1],
                color='#2a2b2c', linewidth=2.4, zorder=3)

    bx = geom.boom_Xc + geom.boom_u
    bz = geom.boom_Zc + geom.boom_w
    ax.scatter(bx, bz,
               s=70, color='#1b6511', zorder=5,
               label='booms')

    ax.scatter(geom.C[0], geom.C[2],
               s=110, color='#D2B026', marker='D', zorder=6,
               label='centroid')

    ax.scatter(geom.S_XYZ[0], geom.S_XYZ[2],
               s=160, color='#17B18A', marker='*', zorder=6,
               label='shear centre')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Z [mm]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best', framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ================================================================================
# PyVista visualization
# ================================================================================

def _airfoil_loop_xz(afl: AirfoilData) -> np.ndarray:
    '''Closed airfoil polyline (upper LE->TE + lower TE->LE) adimensional.'''
    xu = np.asarray(afl.x_upper, dtype=float)
    zu = np.asarray(afl.y_upper, dtype=float)
    xl = np.asarray(afl.x_lower, dtype=float)
    zl = np.asarray(afl.y_lower, dtype=float)
    x  = np.concatenate([xu, xl[::-1][1:]])
    z  = np.concatenate([zu, zl[::-1][1:]])
    return np.column_stack([x, z])


def build_wing_surface_points(
    afl       : AirfoilData,
    lerp_wing,
    d_global  : np.ndarray | None = None,
    scale     : float = 1.0,
) -> np.ndarray:
    '''
    Stack the wing surface as an (N_chord, N_span, 3) point cloud, optionally
    translated + rotated by the nodal displacement of each station.

    Args:
        afl       : Airfoil profile.
        lerp_wing : LerpWingData (one station per FEA node).
        d_global  : Optional (6n,) global displacement vector.
        scale     : Magnification factor on the displacement.

    Returns:
        (N_chord, N_span, 3) point cloud in global XYZ [mm].
    '''
    loop  = _airfoil_loop_xz(afl)
    Y_sta = np.asarray(lerp_wing.Y_sta, dtype=float)
    n_c, n_s = loop.shape[0], Y_sta.size

    pts = np.zeros((n_c, n_s, 3))
    for j in range(n_s):
        c   = float(lerp_wing.chord[j])
        tw  = float(lerp_wing.twist[j])
        xle = float(lerp_wing.LE[j, 0])
        zle = float(lerp_wing.LE[j, 2])

        xs = loop[:, 0] * c
        zs = loop[:, 1] * c
        if abs(tw) > 1e-12:
            ca, sa = np.cos(tw), np.sin(tw)
            xs, zs = ca * xs - sa * zs, sa * xs + ca * zs

        X = xs + xle
        Z = zs + zle
        Y = np.full_like(X, Y_sta[j])

        if d_global is not None:
            u,  v,  w  = d_global[6 * j     : 6 * j + 3]
            rx, ry, rz = d_global[6 * j + 3 : 6 * j + 6]
            # Small-rotation translation: dr = theta x r
            X_loc = X - xle
            Z_loc = Z - zle
            Y_loc = np.zeros_like(X_loc)
            dX = ry * Z_loc - rz * Y_loc
            dY = rz * X_loc - rx * Z_loc
            dZ = rx * Y_loc - ry * X_loc
            X = X + scale * (u + dX)
            Y = Y + scale * (v + dY)
            Z = Z + scale * (w + dZ)

        pts[:, j, 0] = X
        pts[:, j, 1] = Y
        pts[:, j, 2] = Z
    return pts


def _displace_per_station_points(
    pts        : np.ndarray,
    le_per_sta : np.ndarray,
    d_global   : np.ndarray | None,
    scale      : float = 1.0,
) -> np.ndarray:
    '''
    Apply per-station rigid-body displacement to a structured point cloud.

    The cross-section is assumed planar (Y_loc = 0) and the rotation
    reference is the LE of each station, matching build_wing_surface_points.

    Args:
        pts        : (n_chord, n_span, 3) point cloud in global XYZ [mm].
        le_per_sta : (n_span, 3) LE coordinates per station [mm].
        d_global   : (6 * n_span,) global displacement vector; None = identity.
        scale      : Magnification factor on the displacement.

    Returns:
        (n_chord, n_span, 3) displaced point cloud.
    '''
    if d_global is None:
        return pts
    out = pts.copy()
    for j in range(out.shape[1]):
        disp       = d_global[6 * j : 6 * j + 6]
        u,  v,  w  = disp[0], disp[1], disp[2]
        rx, ry, rz = disp[3], disp[4], disp[5]
        xle, _, zle = le_per_sta[j]
        X_loc = out[:, j, 0] - xle
        Z_loc = out[:, j, 2] - zle
        Y_loc = np.zeros_like(X_loc)
        dX = ry * Z_loc - rz * Y_loc
        dY = rz * X_loc - rx * Z_loc
        dZ = rx * Y_loc - ry * X_loc
        out[:, j, 0] = out[:, j, 0] + scale * (u + dX)
        out[:, j, 1] = out[:, j, 1] + scale * (v + dY)
        out[:, j, 2] = out[:, j, 2] + scale * (w + dZ)
    return out


def build_structural_components(
    sections  : SimpleNamespace,
    d_global  : np.ndarray | None = None,
    scale     : float = 1.0,
) -> dict:
    '''
    Stack the per-station structural reference points (centroid, shear
    centre, front- and rear-spar webs) into arrays ready for PyVista,
    optionally displaced by the nodal solution.

    Args:
        sections : Output of build_sections (carries sec_data + lerp_wing).
        d_global : Optional (6n,) global displacement vector.
        scale    : Magnification factor on the displacement.

    Returns:
        Dict with keys
            centroid_line  : (n_span, 3) polyline of GeomData.C
            shear_ctr_line : (n_span, 3) polyline of GeomData.S_XYZ
            front_spar     : (2, n_span, 3) bottom/top quad strip (T1 seg6)
            rear_spar      : (2, n_span, 3) bottom/top quad strip (T1 seg7)
    '''
    sec        = sections.sec_data
    n_s        = len(sec)
    le_per_sta = np.asarray(sections.lerp_wing.LE, dtype=float)

    C = np.array([s.C     for s in sec], dtype=float)
    S = np.array([s.S_XYZ for s in sec], dtype=float)

    front = np.zeros((2, n_s, 3))
    rear  = np.zeros((2, n_s, 3))
    for j, gd in enumerate(sec):
        y_sta = float(gd.C[1])
        seg6  = next(s for s in gd.T1 if s['label'] == 'seg6')   # B5 -> B3
        seg7  = next(s for s in gd.T1 if s['label'] == 'seg7')   # B7 -> B1
        front[0, j] = (seg6['pts'][0, 0], y_sta, seg6['pts'][0, 1])
        front[1, j] = (seg6['pts'][1, 0], y_sta, seg6['pts'][1, 1])
        rear [0, j] = (seg7['pts'][0, 0], y_sta, seg7['pts'][0, 1])
        rear [1, j] = (seg7['pts'][1, 0], y_sta, seg7['pts'][1, 1])

    C_pts = _displace_per_station_points(
        C.reshape(1, n_s, 3), le_per_sta, d_global, scale,
    )
    S_pts = _displace_per_station_points(
        S.reshape(1, n_s, 3), le_per_sta, d_global, scale,
    )
    front = _displace_per_station_points(front, le_per_sta, d_global, scale)
    rear  = _displace_per_station_points(rear,  le_per_sta, d_global, scale)

    return {
        'centroid_line'  : C_pts[0],
        'shear_ctr_line' : S_pts[0],
        'front_spar'     : front,
        'rear_spar'      : rear,
    }


def plot_wing_deformation(
    scenario  : str,
    afl       : AirfoilData,
    sections  : SimpleNamespace,
    d         : np.ndarray,
    scale     : float,
    out_path  : Path,
    show      : bool = True,
) -> None:
    '''
    PyVista comparison plot: undeformed wing skin (light wireframe) vs
    deformed skin (translucent, coloured by vertical deflection), with the
    cross-section centroid line, shear-centre line, and front / rear spar
    panels overlaid on the deformed configuration so the structural
    backbone is visible through the transparent skin.

    Args:
        scenario : Free-form label (figure title and screenshot name).
        afl      : Airfoil profile.
        sections : SimpleNamespace returned by build_sections.
        d        : (6n,) global displacement vector.
        scale    : Deformation magnification.
        out_path : PNG output path.
        show     : Pop the interactive window after saving.
    '''
    import pyvista as pv

    lerp_wing = sections.lerp_wing
    pts_undef = build_wing_surface_points(afl, lerp_wing, None, 1.0)
    pts_def   = build_wing_surface_points(afl, lerp_wing, d,    scale)
    n_c, n_s, _ = pts_undef.shape

    undef = pv.StructuredGrid()
    undef.points     = pts_undef.reshape(-1, 3, order='F')
    undef.dimensions = (n_c, n_s, 1)

    defo  = pv.StructuredGrid()
    defo.points      = pts_def.reshape(-1, 3, order='F')
    defo.dimensions  = (n_c, n_s, 1)
    defo['w_global'] = (pts_def[:, :, 2].reshape(-1, order='F')
                       - pts_undef[:, :, 2].reshape(-1, order='F'))

    struct = build_structural_components(sections, d_global=d, scale=scale)

    fs = pv.StructuredGrid()
    fs.points     = struct['front_spar'].reshape(-1, 3, order='F')
    fs.dimensions = (2, n_s, 1)
    rs = pv.StructuredGrid()
    rs.points     = struct['rear_spar' ].reshape(-1, 3, order='F')
    rs.dimensions = (2, n_s, 1)

    pl = pv.Plotter(off_screen=not show, window_size=(1200, 700))
    pl.add_mesh(undef, color='lightgrey', style='wireframe',
                line_width=1, opacity=0.25, label='undeformed')
    pl.add_mesh(defo, scalars='w_global', cmap='turbo',
                show_edges=False, opacity=0.35,
                label=f'deformed (x{scale:g})')

    pl.add_mesh(fs, color='#3a6ea5', opacity=0.85, label='front spar')
    pl.add_mesh(rs, color='#9a3a3a', opacity=0.85, label='rear spar')

    pl.add_mesh(pv.lines_from_points(struct['centroid_line']),
                color='#D2B026', line_width=4, label='centroid line')
    pl.add_mesh(pv.lines_from_points(struct['shear_ctr_line']),
                color='#17B18A', line_width=4, label='shear-centre line')

    pl.add_legend()
    pl.add_axes()
    pl.add_text(f"FEA validation - {scenario}  (deformation x{scale:g})",
                position='upper_left', font_size=10)

    # Stand-alone interactive HTML (vtk.js) for sharing with users that
    # do not have Python installed. Must be exported BEFORE show(),
    # otherwise the plotter is already closed.
    try:
        pl.export_html(str(out_path.with_suffix('.html')))
    except Exception as exc:                                   # pragma: no cover
        print(f"  [warn] HTML export failed: {exc!r}")

    pl.show(screenshot=str(out_path))


# Internal-force component groups toggled on screen.
# Each entry is (label, local DOF index 0..5); the spatial axis a component
# is drawn along is (index % 3): {0,3}->x, {1,4}->y, {2,5}->z.
#     N  = axial   (Fx, idx 0)      Sy = shear y (Fy, idx 1)
#     My = bending (My, idx 4)      Sz = shear z (Fz, idx 2)
#     Mz = bending (Mz, idx 5)      T  = torsion (Mx, idx 3)
_FORCE_GROUPS = {
    'a': (('N', 0), ('My', 4), ('Mz', 5)),
    'b': (('Sy', 1), ('Sz', 2), ('T', 3)),
}
_FORCE_COLORS = {
    'N' : '#d62728', 'My': '#2ca02c', 'Mz': '#1f77b4',
    'Sy': '#ff7f0e', 'Sz': '#9467bd', 'T' : '#8c564b',
}


def plot_internal_forces(
    scenario : str,
    mesh     : MeshData,
    results  : FeaResults,
    afl      : AirfoilData,
    sections : SimpleNamespace,
    out_path : Path,
    show     : bool = True,
    cond_idx : int = 0,
) -> None:
    '''
    PyVista plot of the end-A internal force / moment resultants as arrow
    glyphs rooted at the actual element node coordinates, over a translucent
    wing-outline surface.

    Two side-by-side renders are produced:
        [left]  LOCAL  - components drawn along each element's local axes
                         (from mesh.R), magnitudes from results.Q_sc.
        [right] GLOBAL - components drawn along the world XYZ axes,
                         magnitudes from results.Q_sc_gl.

    An on-screen checkbox toggles which component triad is shown in BOTH
    plots:
        ON  -> group (a): N, My, Mz
        OFF -> group (b): Sy, Sz, T

    Arrow length encodes magnitude on a global log scale (so values spanning
    several orders along the span stay visible); each arrow itself is a plain
    linear glyph. Direction encodes sign and each component has a fixed
    colour (see legend).

    Args:
        scenario : Free-form label (figure title and screenshot name).
        mesh     : MeshData (coord, conn, per-element rotation R).
        results  : FeaResults (Q_sc local, Q_sc_gl global, shape (12, m, nc)).
        afl      : Airfoil profile (for the outline surface).
        sections : SimpleNamespace from build_sections (carries lerp_wing).
        out_path : PNG output path.
        show     : Pop the interactive window (required for the toggle).
        cond_idx : Load-condition index to display.
    '''
    import pyvista as pv

    coord = np.asarray(mesh.coord, dtype=float)        # (n, 3)
    conn  = np.asarray(mesh.conn,  dtype=int)          # (m, 4)
    R_all = np.asarray(mesh.R,     dtype=float)        # (12, 12, m)
    m     = conn.shape[0]

    Q_loc = np.asarray(results.Q_sc   [:, :, cond_idx], dtype=float)   # (12, m)
    Q_gl  = np.asarray(results.Q_sc_gl[:, :, cond_idx], dtype=float)   # (12, m)

    # End-A node coordinates of every element (m, 3).
    pts_a = coord[conn[:, 0]]

    # -------- Length scale: linear glyph, log-compressed magnitude --------
    # Forces (N) and moments (N*mm) get separate log windows of `decades`, so
    # the ~1e3x unit gap between them does not collapse the force arrows. The
    # largest of each kind maps to L_ref; anything <= its floor vanishes.
    span    = float(np.ptp(coord[:, 1]))
    chord   = float(np.ptp(coord[:, 0]))
    L_ref   = max(span, chord, 1.0) * 0.10
    decades = 3.0

    def _kind_max(idx_set: tuple[int, ...]) -> float:
        cols = list(idx_set)
        vals = np.concatenate([np.abs(Q_loc[cols, :]).ravel(),
                               np.abs(Q_gl [cols, :]).ravel()])
        return max(float(np.max(vals)) if vals.size else 1.0, 1e-30)

    v_max_f = _kind_max((0, 1, 2))        # forces  [N]
    v_max_m = _kind_max((3, 4, 5))        # moments [N*mm]

    def _log_len(v_abs: np.ndarray, v_max: float) -> np.ndarray:
        '''Map |value| -> arrow length in [0, L_ref] on a per-kind log scale.'''
        v_floor = v_max * 10.0 ** (-decades)
        out  = np.zeros_like(v_abs, dtype=float)
        mask = v_abs > v_floor
        out[mask] = L_ref * np.log10(v_abs[mask] / v_floor) / decades
        return out

    # Solver global frame -> world geometry frame: X<-y, Y<-x, Z<-z, so
    # global component a is drawn along world axis _GL_AXIS_PERM[a].
    _GL_AXIS_PERM = (1, 0, 2)

    def _axis_dirs(local: bool, a: int) -> np.ndarray:
        '''Unit axis (m, 3) for spatial axis a; local uses per-elem R.'''
        if not local:
            e = np.zeros(3)
            e[_GL_AXIS_PERM[a]] = 1.0
            return np.tile(e, (m, 1))
        return np.array([R_all[0:3, 0:3, j][:, a] for j in range(m)])

    def _add_component(pl, label: str, idx: int, local: bool):
        '''Add one component's end-A arrow glyphs and return the actor.'''
        a     = idx % 3
        Q     = Q_loc if local else Q_gl
        v_max = v_max_f if idx in (0, 1, 2) else v_max_m
        vals  = Q[idx, :]                                       # (m,) end-A
        lens  = np.sign(vals) * _log_len(np.abs(vals), v_max)   # signed length
        vecs  = _axis_dirs(local, a) * lens[:, None]
        pdata = pv.PolyData(pts_a.copy())
        pdata['vec'] = vecs
        glyph = pdata.glyph(
            orient = 'vec', scale = 'vec', factor = 1.0,
            geom   = pv.Arrow(tip_length=0.30, tip_radius=0.09,
                              shaft_radius=0.03),
        )
        return pl.add_mesh(glyph, color=_FORCE_COLORS[label], label=label)

    # -------- Undeformed wing outline surface (rebuilt per subplot) --------
    pts_wing    = build_wing_surface_points(afl, sections.lerp_wing, None, 1.0)
    n_c, n_s, _ = pts_wing.shape

    def _wing_surface():
        g = pv.StructuredGrid()
        g.points     = pts_wing.reshape(-1, 3, order='F')
        g.dimensions = (n_c, n_s, 1)
        return g

    pl = pv.Plotter(off_screen=not show, shape=(1, 2),
                    window_size=(1700, 850))

    actors : dict[str, list] = {'a': [], 'b': []}
    for col, local in enumerate((True, False)):
        pl.subplot(0, col)
        frame = 'LOCAL (element frame)' if local else 'GLOBAL (world frame)'
        pl.add_text(f"{scenario}\n{frame}", position='upper_left',
                    font_size=10)
        pl.add_axes()
        pl.add_mesh(_wing_surface(), color='lightgrey', opacity=0.15,
                    show_edges=False, label='wing outline')
        pl.add_mesh(pv.lines_from_points(coord), color='black', line_width=2)
        pl.add_points(coord, color='black', point_size=8,
                      render_points_as_spheres=True)
        for grp, comps in _FORCE_GROUPS.items():
            for label, idx in comps:
                actors[grp].append(_add_component(pl, label, idx, local))
        pl.add_legend(bcolor='white', size=(0.16, 0.18))

    # -------- Initial state: group (a) visible --------
    for act in actors['b']:
        act.visibility = False

    def _toggle(show_a: bool) -> None:
        for act in actors['a']:
            act.visibility = show_a
        for act in actors['b']:
            act.visibility = not show_a

    if show:
        pl.add_checkbox_button_widget(
            _toggle, value=True, position=(10, 10), size=35,
            color_on='#2ca02c', color_off='#888888',
        )
        pl.add_text(
            "toggle  ON: (a) N, My, Mz    OFF: (b) Sy, Sz, T",
            position=(55, 18), font_size=10, viewport=True,
        )

    pl.link_views()
    pl.show(screenshot=str(out_path))


# ================================================================================
# Per-scenario orchestration
# ================================================================================

def run_scenario(
    name        : str,
    loads       : dict,
    sections    : SimpleNamespace,
    mesh        : MeshData,
    afl         : AirfoilData,
    laminate_db : dict[str, LaminateData],
    ply_db      : dict[str, PlyData],
    out_dir     : Path,
    scale       : float,
    show        : bool = False,
) -> tuple[FeaResults, StressData, FailureData]:
    '''
    Solve one scenario and run the full post-processing chain on it.

    Pipeline
    ----------------
        1. LinearStaticSolver           -> FeaResults
        2. BeamElement matrix dump      -> stdout
        3. StressRecovery               -> StressData
        4. TsaiWuFailure                -> FailureData
        5. PyVista deformation plot     -> PNG + HTML

    Args:
        name        : Short scenario tag for filenames and titles.
        loads       : Loads dict (overrides fem_setup.loads in MeshData).
        sections    : SimpleNamespace returned by build_sections.
        mesh        : MeshData built by MeshBuilder.
        afl         : Airfoil profile.
        laminate_db : 'MAT{k}' -> LaminateData dict.
        ply_db      : ply name -> PlyData dict.
        out_dir     : Where to save the PNG screenshots.
        scale       : Deformation magnification for the mesh plot.
        show        : Pop interactive windows.

    Returns:
        Tuple (results, stress, failure).
    '''
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. Linear static solve --------
    results = LinearStaticSolver(
        mesh           = mesh,
        loads          = loads,
        enable_logging = False,
    ).results

    # -------- 2. Matrix dump --------
    bd = debug_beam_element(sections, mesh.coord, mesh.conn, _DEBUG_ELEM)
    report_matrices(
        scenario = name,
        bd       = bd,
        mesh     = mesh,
        results  = results,
    )

    # -------- 3. Stress recovery --------
    stress = compute_stresses(sections, mesh, results)
    report_stress(stress)

    # -------- 4. Tsai-Wu failure --------
    failure = compute_tsw_failure(sections, stress, mesh, laminate_db, ply_db)
    report_failure(failure)

    # -------- 5. Deformation plot --------
    plot_wing_deformation(
        scenario  = name,
        afl       = afl,
        sections  = sections,
        d         = results.d[:, 0],
        scale     = scale,
        out_path  = out_dir / f"{name}_deformation.png",
        show      = False, #show,
    )

    # -------- 6. Internal force / moment vectors --------
    plot_internal_forces(
        scenario = name,
        mesh     = mesh,
        results  = results,
        afl      = afl,
        sections = sections,
        out_path = out_dir / f"{name}_internal_forces.png",
        show     = show,
    )

    return results, stress, failure


def main(show: bool = _SHOW_PLOTS) -> None:
    '''
    Orchestrate the full validation:  load -> sections -> assemble -> solve.
    '''
    # -------- 1. Inputs --------
    wing, afl, ex_loads = load_inputs()
    laminate_db, ply_db = load_laminate_catalog()
    Y_sta = np.asarray(ex_loads.Y_hf, dtype=float)
    n     = Y_sta.size

    print("=" * 80)
    print("  CL3O FEA VALIDATION - DA62 left wing")
    print("=" * 80)
    print(f"  span (half)   : {abs(Y_sta.min()):.1f} mm")
    print(f"  n nodes       : {n}")
    print(f"  n elements    : {n - 1}")
    print(f"  n cpts        : {int(wing.n_cpts)}")
    print(f"  n laminates   : {len(laminate_db)}")
    print(f"  n plies       : {len(ply_db)}")

    print_design_vector_summary(wing, _DESIGN_VECTOR, laminate_db)

    # -------- 2. Cross-section properties at each station --------
    sections = build_sections(wing, afl, Y_sta, _DESIGN_VECTOR, laminate_db)

    # -------- 2b. Cross-section snapshot at the debug element --------
    _DFLT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_cross_section(
        geom     = sections.sec_data[_DEBUG_ELEM],
        title    = (f"Cross-section at element {_DEBUG_ELEM} "
                    f"(Y = {Y_sta[_DEBUG_ELEM]:.0f} mm,"
                    f" chord = {sections.chord[_DEBUG_ELEM]:.0f} mm)"),
        out_path = _DFLT_OUT_DIR / f"crosssection_elem{_DEBUG_ELEM:02d}.png",
        show     = False, #show,
    )

    # -------- 3. Static FEA pre-processing (mesh topology + nominal loads) --------
    fem_setup = FemSetup(
        exloads_db     = ex_loads,
        lerp_wing_db   = sections.lerp_wing,
        enable_logging = False,
    ).fem_setup

    # -------- 4. Assemble global stiffness K --------
    mesh = MeshBuilder(
        data           = (fem_setup, sections),
        enable_logging = False,
    ).data

    # -------- 5. Structural mass (geometry + materials only, no loads) --------
    score = compute_mass(sections, mesh, laminate_db)
    report_mass(score)

    # -------- 6a. Scenario A: tip point load --------
    # run_scenario(
    #     name        = "scenarioA_tipload",
    #     loads       = loads_dict_tip(n, _TIP_FX, _TIP_FZ, _TIP_MY),
    #     sections    = sections,
    #     mesh        = mesh,
    #     afl         = afl,
    #     laminate_db = laminate_db,
    #     ply_db      = ply_db,
    #     out_dir     = _DFLT_OUT_DIR,
    #     scale       = 1.0,
    #     show        = show,
    # )

    # -------- 6b. Scenario B: distributed loads from ExLoadsData --------
    run_scenario(
        name        = "scenarioB_distloads",
        loads       = loads_dict_from_fem(fem_setup),
        sections    = sections,
        mesh        = mesh,
        afl         = afl,
        laminate_db = laminate_db,
        ply_db      = ply_db,
        out_dir     = _DFLT_OUT_DIR,
        scale       = 1.0,
        show        = show,
    )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    main(show=_SHOW_PLOTS)
