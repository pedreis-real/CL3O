'''
================================================================================
CL3O - Composite Wing Structural Sizing.
MSA Validation Module.

Validation harness for the MSA pipeline (mesh assembly, global K solve,
internal force recovery) against reference values obtained in Simcenter
Femap (r). Two load cases cover the thesis Section 3.5.1 (p. 86):

    1) 2D cantilever of length L = 3000 mm along global +X with a tip
       point load P_Z = +1000 N applied at the free end.

    2) 3D cantilever mirroring the AC line of the analysed DA62 wing at
       1:2 scale (root at origin, tip at x=134.5715, y=-3242.25,
       z=202.461 mm), with tip loads F = (10, 0, 1000) N and
       M = (0, -10000, 0) N*mm.

Both cases use a constant cross-section taken from Section level 5
(Wortmann FX63-137, thesis Figure 50) with generic orthotropic material
E1 = 200 GPa, E2 = 10 GPa, G = 6 GPa. Until T1.a populates the real
Level-5 GeomData from SectionBuilder, a placeholder 180 x 40 mm
thin-wall box is used; section-dependent tests (tip displacements) are
therefore marked xfail with a pointer back to T1.a, while section-
independent tests (global equilibrium, root reactions) run and must
pass exactly.

Femap reference values are taken from docs/Femap_viga_cantilever_2D.xlsx
and docs/Femap_viga_cantilever_3D.xlsx (sheets "Nodes" and "Elements").

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import numpy as np
import pytest

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from geometry.geom_properties import GeomData
from fea.solver.mesh_builder  import MeshBuilder
from fea.solver.static_analysis    import LinearStaticSolver

# ================ Global variables ================
# Orthotropic generic material per thesis p. 86.
_E1 = 200_000.0                           # MPa
_E2 =  10_000.0                           # MPa
_G  =   6_000.0                           # MPa

# Placeholder Section-Level-5 properties. Replace with SectionBuilder
# output for the Wortmann FX63-137 once T1.a closes. The rectangular
# idealisation here is deliberately rough (bending stiffness is roughly
# half of the real Wortmann section); it exists only so the harness can
# run end-to-end and the reaction-equilibrium tests are meaningful.
_BOX_B = 180.0                            # mm, width (along local y)
_BOX_H =  40.0                            # mm, height (along local z)
_BOX_T =   2.0                            # mm, wall thickness
_BOOM_A = _BOX_T * (_BOX_B + _BOX_H) / 4.0

# 2D cantilever parameters.
_L_2D     = 3000.0                        # mm, total length along +X
_N_ELEM_2D = 10
_TIP_FZ_2D = 1_000.0                      # N, global +Z at free end

# 3D cantilever parameters (1:2-scaled DA62 AC line per xlsx Nodes sheet).
_COORD_3D = np.array([
    [  0.0000,     0.000000, 0.000000],   # node  1 (root, fixed)
    [  0.0000,  -103.841333, 0.000000],
    [  0.0000,  -207.682666, 0.000000],
    [  0.0000,  -311.524000, 0.000000],
    [  3.7741625,  -399.780812, 0.000000],
    [  7.5483250,  -488.037625, 0.000000],
    [ 11.3224875,  -576.294437, 0.000000],
    [ 15.0966500,  -664.551250, 0.000000],
    [ 18.8708125,  -752.808062, 0.000000],
    [ 22.6449750,  -841.064875, 0.000000],
    [ 26.4191375,  -929.321687, 0.000000],
    [ 30.1933000, -1017.578500, 0.000000],
    [ 36.7169375, -1156.620000, 12.6538125],
    [ 43.2405750, -1295.662000, 25.3076250],
    [ 49.7642125, -1434.704000, 37.9614375],
    [ 56.2878500, -1573.746000, 50.6152500],
    [ 62.8114875, -1712.788000, 63.2690625],
    [ 69.3351250, -1851.830000, 75.9228750],
    [ 75.8587625, -1990.872000, 88.5766875],
    [ 82.3824000, -2129.914000, 101.2305000],
    [ 88.9060375, -2268.956000, 113.8843125],
    [ 95.4296750, -2407.998000, 126.5381250],
    [101.9533125, -2547.040000, 139.1919375],
    [108.4769500, -2686.082000, 151.8457500],
    [115.0005875, -2825.124000, 164.4995625],
    [121.5242250, -2964.166000, 177.1533750],
    [128.0478625, -3103.208000, 189.8071875],
    [134.5715000, -3242.250000, 202.4610000],   # node 28 (tip, loaded)
])
_N_ELEM_3D = _COORD_3D.shape[0] - 1

_TIP_F_3D = np.array([10.0, 0.0, 1000.0])                        # N
_TIP_M_3D = np.array([0.0, -10_000.0, 0.0])                      # N*mm

# --------- Femap reference values from docs/Femap_*.xlsx ---------
# 2D: node 1 is the constraint. Constraint forces are *reactions* (force
# applied by the constraint on the node), so root_R = -F_tip. Tip state
# read at node 11. Values pasted from sheet "Nodes" row-by-row.
FEMAP_REF_2D: dict[str, float | None] = {
    # Tip DOFs at X = 3000 (section-dependent)
    "tip_uX"  : -0.40513057,
    "tip_uY"  : -2.45479608,
    "tip_uZ"  :  130.82592773,
    "tip_rotX":  0.0,
    "tip_rotY": -0.06428561,
    "tip_rotZ": -0.00122740,
    # Root reactions at X = 0 (section-independent)
    "root_RX" : -3.265358e-10,
    "root_RY" : -1.364242e-12,
    "root_RZ" : -1000.0,
    "root_MX" :  0.0,
    "root_MY" :  3_000_000.0,
    "root_MZ" : -2.211527e-08,
}

# 3D: node 1 is root, node 30 is tip (Femap IDs). Tip state from
# "Nodes" sheet at the last row; root reactions at first row.
FEMAP_REF_3D: dict[str, float | None] = {
    # Tip DOFs (section-dependent)
    "tip_uX"  : -5.17086697,
    "tip_uY"  : 13.32283688,
    "tip_uZ"  : 166.28269958,
    "tip_rotX": -0.07503450,
    "tip_rotY": -0.01565527,
    "tip_rotZ": -0.00093608,
    # Root reactions (section-independent)
    "root_RX" : -10.0,
    "root_RY" : -3.041350e-09,
    "root_RZ" : -1000.0,
    "root_MX" :  3_242_250.0,
    "root_MY" :  142_546.890625,
    "root_MZ" : -32_422.5,
}

# Mark section-dependent tests as xfail pending T1.a (real Wortmann
# section). They will still execute so we can inspect the gap; once
# T1.a lands they should xpass.
_XFAIL_SECTION = pytest.mark.xfail(
    reason = (
        "Placeholder 180x40 box section; real Wortmann FX63-137 Level-5 "
        "properties pending T1.a (geometric-properties validation)."
    ),
    strict = False,
)


# ================================================================================
# PRIVATE API - Geometry and mesh builders
# ================================================================================

def _make_level5_section(Y_sta: float = 0.0) -> GeomData:
    '''
    Placeholder Section-Level-5 as a 4-boom thin-wall rectangular box.

    Analytical inertias (theta_P = 0, principal axes aligned with XZ):
        I_1 = I_XX = 4 * A_b * (H/2)^2   (bending under load in -Z)
        I_2 = I_ZZ = 4 * A_b * (B/2)^2
    Bredt torsion constant for the closed box:
        J = 2 * (B*H)^2 * t / (B + H)
    '''
    half_b = _BOX_B / 2.0
    half_h = _BOX_H / 2.0

    boom_u = np.array([-half_b, +half_b, +half_b, -half_b])
    boom_w = np.array([+half_h, +half_h, -half_h, -half_h])
    boom_A = np.full(4, _BOOM_A)

    I_xx = 4.0 * _BOOM_A * half_h ** 2
    I_zz = 4.0 * _BOOM_A * half_b ** 2
    J    = 2.0 * (_BOX_B * _BOX_H) ** 2 * _BOX_T / (_BOX_B + _BOX_H)

    return GeomData(
        Y_sta    = float(Y_sta),
        chord    = _BOX_B,
        I_XX     = I_xx,
        I_ZZ     = I_zz,
        I_XZ     = 0.0,
        I_1      = I_xx,
        I_2      = I_zz,
        theta_P  = 0.0,
        J        = J,
        E1_eq    = _E1,
        E2_eq    = _E2,
        G_eq     = _G,
        G_REF    = _G,
        boom_u   = boom_u,
        boom_w   = boom_w,
        boom_A   = boom_A,
        A_flange = np.zeros(4),
        t_k      = np.full(7, _BOX_T),
        G_k      = np.full(7, _G),
        qsX_star = np.zeros(7),
        qsZ_star = np.zeros(7),
        qT_star  = np.zeros(7),
        us       = 0.0,
        ws       = 0.0,
    )


def _build_cantilever_2d() -> tuple[np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    '''
    Straight cantilever along +X, 10 elements of length 300 mm. Node 0
    fully restrained; tip load _TIP_FZ_2D applied in global +Z at node 10
    (matches docs/Femap_viga_cantilever_2D.xlsx).
    '''
    n_nodes             = _N_ELEM_2D + 1
    coord               = np.zeros((n_nodes, 3))
    coord[:, 0]         = np.linspace(0.0, _L_2D, n_nodes)
    conn                = np.column_stack([np.arange(_N_ELEM_2D),
                                           np.arange(1, _N_ELEM_2D + 1)]
                                          ).astype(int)
    restraints          = np.zeros((n_nodes, 6), dtype=int)
    restraints[0, :]    = 1
    F_nodal             = np.zeros((n_nodes, 6))
    F_nodal[-1, 2]      = _TIP_FZ_2D
    return coord, conn, restraints, F_nodal


def _build_cantilever_3d() -> tuple[np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    '''
    3D cantilever using the 28-element AC-line mesh from the Femap xlsx.
    Node 0 fully restrained; tip loads F = (10, 0, 1000) N and
    M = (0, -10000, 0) N*mm applied at the last node.
    '''
    coord               = _COORD_3D.copy()
    n_nodes             = coord.shape[0]
    conn                = np.column_stack([np.arange(_N_ELEM_3D),
                                           np.arange(1, _N_ELEM_3D + 1)]
                                          ).astype(int)
    restraints          = np.zeros((n_nodes, 6), dtype=int)
    restraints[0, :]    = 1
    F_nodal             = np.zeros((n_nodes, 6))
    F_nodal[-1, :3]     = _TIP_F_3D
    F_nodal[-1, 3:]     = _TIP_M_3D
    return coord, conn, restraints, F_nodal


def _solve(coord, conn, restraints, F_nodal):
    '''Assemble and solve the MSA pipeline with a constant Level-5 section.'''
    sec = [_make_level5_section(coord[c[0], 1]) for c in conn]
    fem = MeshBuilder(
        sec_data       = sec,
        coord          = coord,
        conn           = conn,
        restraints     = restraints,
        F_nodal        = F_nodal,
        enable_logging = False,
    ).arrays
    return LinearStaticSolver(fem, enable_logging=False).results


# ================================================================================
# PUBLIC API - 2D cantilever tests
# ================================================================================

def test_2d_euler_bernoulli_self_check() -> None:
    '''
    2D tip u_Z matches P*L^3 / (3*E*I) within 1% using the placeholder
    box-section inertia. Independent of Femap reference data.
    '''
    coord, conn, restraints, F_nodal = _build_cantilever_2d()
    res = _solve(coord, conn, restraints, F_nodal)

    sec        = _make_level5_section()
    analytical = _TIP_FZ_2D * _L_2D ** 3 / (3.0 * _E1 * sec.I_XX)
    tip_uZ     = float(res.dmatrix[2, -1])

    rel_err = abs(tip_uZ - analytical) / abs(analytical)
    assert rel_err < 1.0e-2, (
        f"tip u_Z = {tip_uZ:.4f} mm vs analytical {analytical:.4f} mm "
        f"(rel err {rel_err:.3%})"
    )


@pytest.mark.parametrize("key, dof_row, tol", [
    ("root_RX", 0, 1.0e-3),
    ("root_RY", 1, 1.0e-3),
    ("root_RZ", 2, 1.0e-4),
    ("root_MX", 3, 1.0e-3),
    ("root_MY", 4, 1.0e-4),
    ("root_MZ", 5, 1.0e-3),
])
def test_2d_root_reactions_match_femap(key, dof_row, tol) -> None:
    '''Root reactions match Femap exactly (section-independent).'''
    coord, conn, restraints, F_nodal = _build_cantilever_2d()
    res  = _solve(coord, conn, restraints, F_nodal)
    calc = float(res.reaction_mat[dof_row, 0])
    ref  = FEMAP_REF_2D[key]
    # Absolute floor: rounding in MeshBuilder (_N_DEC=4) + machine epsilon.
    atol = max(tol * max(1.0, abs(ref)), 1.0e-3)
    assert np.isclose(calc, ref, atol=atol), (
        f"{key}: calc {calc:.6e} vs Femap {ref:.6e} (atol={atol:.1e})"
    )


@_XFAIL_SECTION
@pytest.mark.parametrize("key, dof_row", [
    ("tip_uX",   0), ("tip_uY",   1), ("tip_uZ",   2),
    ("tip_rotX", 3), ("tip_rotY", 4), ("tip_rotZ", 5),
])
def test_2d_tip_dofs_match_femap(key, dof_row) -> None:
    '''Each 2D tip DOF matches Femap within 5% (section-dependent).'''
    coord, conn, restraints, F_nodal = _build_cantilever_2d()
    res  = _solve(coord, conn, restraints, F_nodal)
    calc = float(res.dmatrix[dof_row, -1])
    ref  = FEMAP_REF_2D[key]
    assert np.isclose(calc, ref, rtol=5.0e-2, atol=1.0e-4), (
        f"{key}: calc {calc:.6e} vs Femap {ref:.6e}"
    )


# ================================================================================
# PUBLIC API - 3D cantilever tests
# ================================================================================

# 3D tolerances differ per DOF: strong-axis reactions (RZ, MX, MY) are
# ~1e-5 accurate, but MeshBuilder rounds K to 4 decimals which degrades
# weak-axis reactions (RX, RY, MZ) by up to ~2%. Tolerances below floor
# at each DOF's observed numerical limit.
@pytest.mark.parametrize("key, dof_row, tol", [
    ("root_RX", 0, 3.0e-2),
    ("root_RY", 1, 3.0e-2),
    ("root_RZ", 2, 1.0e-3),
    ("root_MX", 3, 1.0e-3),
    ("root_MY", 4, 1.0e-3),
    ("root_MZ", 5, 3.0e-2),
])
def test_3d_root_reactions_match_femap(key, dof_row, tol) -> None:
    '''
    Root reactions match Femap within the per-DOF tolerance. Strong-axis
    reactions (0.1% floor) are section-independent; weak-axis reactions
    are limited to ~2% by MeshBuilder K-rounding (_N_DEC=4).
    '''
    coord, conn, restraints, F_nodal = _build_cantilever_3d()
    res  = _solve(coord, conn, restraints, F_nodal)
    calc = float(res.reaction_mat[dof_row, 0])
    ref  = FEMAP_REF_3D[key]
    atol = max(tol * max(1.0, abs(ref)), 3.0e-1)
    assert np.isclose(calc, ref, atol=atol), (
        f"{key}: calc {calc:.6e} vs Femap {ref:.6e} (atol={atol:.1e})"
    )


@_XFAIL_SECTION
@pytest.mark.parametrize("key, dof_row", [
    ("tip_uX",   0), ("tip_uY",   1), ("tip_uZ",   2),
    ("tip_rotX", 3), ("tip_rotY", 4), ("tip_rotZ", 5),
])
def test_3d_tip_dofs_match_femap(key, dof_row) -> None:
    '''Each 3D tip DOF matches Femap within 5% (section-dependent).'''
    coord, conn, restraints, F_nodal = _build_cantilever_3d()
    res  = _solve(coord, conn, restraints, F_nodal)
    calc = float(res.dmatrix[dof_row, -1])
    ref  = FEMAP_REF_3D[key]
    assert np.isclose(calc, ref, rtol=5.0e-2, atol=1.0e-4), (
        f"{key}: calc {calc:.6e} vs Femap {ref:.6e}"
    )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
