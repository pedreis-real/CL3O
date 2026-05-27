'''
================================================================================
CWSS - Composite Wing Structural Sizing.
FEM Pipeline Test Fixtures Module.

Shared pytest fixtures and helper builders for the FEM pipeline smoke
tests. Provides a minimal symmetric 4-boom box-beam GeomData, a two-
element cantilever mesh, and a single-ply carbon/epoxy LaminateBundle
so each phase-specific test file can exercise the pipeline without
duplicating the fixture setup.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import numpy as np
import pytest

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from geometry.geom_properties import GeomData
from materials.laminate       import PlyData, LaminateData
from fem.post.tsw_failure     import LaminateBundle

# ================ Global variables ================
# Symmetric 100 x 100 mm box beam with 4 booms of 100 mm^2 each.
# I_1 = I_2 = 4 * 100 * 50^2 = 1e6 mm^4; with E = 70000 MPa and
# L = 1000 mm under P = -100 N the tip deflection is -0.47619 mm.
E_ALU   = 70000.0                     # MPa
G_ALU   = 27000.0                     # MPa
BOOM_A  = 100.0                       # mm^2 per boom
HALF    = 50.0                        # mm, boom offset from centroid
I_BEAM  = 4.0 * BOOM_A * HALF**2      # mm^4
TIP_P   = -100.0                      # N, tip load in global Z
L_TOT   = 1000.0                      # mm, total cantilever length

# Ply strengths chosen so FI = 1.0 exactly at sigma_1 = Xt.
_XT = 2000.0                          # MPa
_XC = -1500.0                         # MPa
_YT =   50.0
_YC = -200.0
_S  =  100.0


# ================================================================================
# PRIVATE API - Section and mesh builders
# ================================================================================

def _make_box_section(Y_sta: float = 0.0) -> GeomData:
    '''Symmetric 4-boom box section aligned with centroidal XZ axes.'''
    boom_u = np.array([-HALF, +HALF, +HALF, -HALF])
    boom_w = np.array([+HALF, +HALF, -HALF, -HALF])
    boom_A = np.full(4, BOOM_A)

    return GeomData(
        Y_sta    = float(Y_sta),
        chord    = 100.0,
        theta_P  = 0.0,
        I_1      = I_BEAM,
        I_2      = I_BEAM,
        J        = 2.0 * I_BEAM,
        E1_eq    = E_ALU,
        E2_eq    = E_ALU,
        G_eq     = G_ALU,
        boom_u   = boom_u,
        boom_w   = boom_w,
        boom_A   = boom_A,
        A_flange = np.zeros(4),
        t_k      = np.full(7, 1.0),
        G_k      = np.full(7, G_ALU),
        qsX_star = np.zeros(7),
        qsZ_star = np.zeros(7),
        qT_star  = np.zeros(7),
        us       = 0.0,
        ws       = 0.0,
    )


def _make_single_ply() -> PlyData:
    '''Build a minimal ply with on-axis stiffness and Tsai-Wu factors.'''
    Ex    = 130000.0
    Ey    = 10000.0
    Es    = 5000.0
    nux   = 0.3
    nuy   = nux * Ey / Ex
    denom = 1.0 - nux * nuy
    Qxys  = np.array([
        [Ex / denom,        nuy * Ex / denom, 0.0],
        [nuy * Ex / denom,  Ey / denom,       0.0],
        [0.0,               0.0,              Es ],
    ])

    Fxx = 1.0 / (_XT * (-_XC))
    Fyy = 1.0 / (_YT * (-_YC))
    Fss = 1.0 / (_S ** 2)
    Fx  = 1.0 / _XT + 1.0 / _XC
    Fy  = 1.0 / _YT + 1.0 / _YC
    Fxy = -0.5 * np.sqrt(Fxx * Fyy)

    return PlyData(
        mat_name = "carbon",
        name     = "p0",
        thick    = 1.0,
        core     = False,
        rho      = 1600.0,
        gms      = 200.0,
        angle    = 0.0,
        Ex=Ex, Ey=Ey, Es=Es, nux=nux, nuy=nuy,
        Xt=_XT, Xc=_XC, Yt=_YT, Yc=_YC, S=_S,
        Qxys   = Qxys,
        Te_p   = np.eye(3),
        Fxx=Fxx, Fyy=Fyy, Fss=Fss, Fxy=Fxy, Fx=Fx, Fy=Fy,
    )


def _make_single_ply_bundle() -> LaminateBundle:
    '''Pair the single ply with a LaminateData exposing compl_a.'''
    ply     = _make_single_ply()
    t       = ply.thick
    compl_a = np.linalg.inv(ply.Qxys) / t

    lam = LaminateData(
        name       = "single_ply",
        plies      = ["p0"],
        np    = 1,
        thick      = t,
        rho        = ply.rho,
        gms        = ply.gms,
        stiff_A    = ply.Qxys * t,
        stiff_B    = np.zeros((3, 3)),
        stiff_D    = ply.Qxys * (t ** 3) / 12.0,
        stiff_ABD  = np.zeros((6, 6)),
        compl_a    = compl_a,
        compl_b    = np.zeros((3, 3)),
        compl_c    = np.zeros((3, 3)),
        compl_d    = np.zeros((3, 3)),
        compl_abcd = np.zeros((6, 6)),
        E1=ply.Ex, E2=ply.Ey, G12=ply.Es,
        nu12=ply.nux, nu21=ply.nuy,
        eta16=0.0, eta26=0.0, eta61=0.0, eta62=0.0,
        eng_compl = np.zeros((3, 3)),
        eng_stiff = np.zeros((3, 3)),
    )
    return LaminateBundle(laminate=lam, plies=[ply])


# ================================================================================
# PUBLIC API - Pytest fixtures
# ================================================================================

@pytest.fixture
def box_section() -> GeomData:
    '''Symmetric 4-boom box GeomData fixture.'''
    return _make_box_section()


@pytest.fixture
def cantilever_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Two-element cantilever along +Y with tip point load.

    Node 0 at Y=0 (fixed), node 1 at Y=500, node 2 at Y=1000 (tip).
    TIP_P is applied at node 2 in global -Z.
    '''
    coord = np.array([
        [0.0,     0.0, 0.0],
        [0.0,   500.0, 0.0],
        [0.0,  L_TOT,  0.0],
    ])
    conn             = np.array([[0, 1], [1, 2]], dtype=int)
    restraints       = np.zeros((3, 6), dtype=int)
    restraints[0, :] = 1
    F_nodal          = np.zeros((3, 6))
    F_nodal[2, 2]    = TIP_P
    return coord, conn, restraints, F_nodal


@pytest.fixture
def single_ply() -> PlyData:
    '''Carbon/epoxy-like single ply with Tsai-Wu factors.'''
    return _make_single_ply()


@pytest.fixture
def single_ply_bundle() -> LaminateBundle:
    '''LaminateBundle wrapping a single carbon/epoxy-like ply.'''
    return _make_single_ply_bundle()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
