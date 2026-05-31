'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Geometry Properties Tests.

Drives GeomPropCalculator end-to-end on the WortmannFX63137 airfoil with
representative DA62 station parameters and asserts the structural section
properties land in physically sensible ranges (positive inertias, cell
areas, boom areas, torsion constant and equivalent moduli).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.Constants import N_BOOMS, N_SEG_T1, N_FLANGES
from cl3o.geometry.geom_properties import GeomPropCalculator, GeomData

pytestmark = pytest.mark.integration


# ================ Global variables ================
_CHORD = 1500.0


# ================================================================================
# PRIVATE API - Calculator builder
# ================================================================================

def _build_calculator(airfoil_arrays) -> GeomPropCalculator:
    '''Assemble a GeomPropCalculator with uniform-material segment arrays.'''
    afl_pts = (
        airfoil_arrays["x_upper"], airfoil_arrays["y_upper"],
        airfoil_arrays["x_lower"], airfoil_arrays["y_lower"],
    )

    # T1 base-segment material arrays (7 segments).
    t_seg       = np.full(N_SEG_T1, 1.0)
    E1_seg      = np.full(N_SEG_T1, 60000.0)
    E2_seg      = np.full(N_SEG_T1, 5000.0)
    G_seg       = np.full(N_SEG_T1, 5000.0)
    E1_bend_seg = np.full(N_SEG_T1, 60000.0)
    E2_bend_seg = np.full(N_SEG_T1, 5000.0)
    T1_props = (t_seg, E1_seg, E2_seg, G_seg, E1_bend_seg, E2_bend_seg)

    # T4 flange material arrays (4 flanges); bf in absolute mm.
    t_flange       = np.full(N_FLANGES, 2.0)
    E1_flange      = np.full(N_FLANGES, 60000.0)
    E2_flange      = np.full(N_FLANGES, 5000.0)
    G_flange       = np.full(N_FLANGES, 5000.0)
    bf             = np.full(N_FLANGES, 20.0)
    E1_bend_flange = np.full(N_FLANGES, 60000.0)
    E2_bend_flange = np.full(N_FLANGES, 5000.0)
    T4_props = (t_flange, E1_flange, E2_flange, G_flange, bf,
                E1_bend_flange, E2_bend_flange)

    return GeomPropCalculator(
        afl_pts  = afl_pts,
        chord    = _CHORD,
        twist    = 0.0,
        Y_sta    = 2000.0,
        xw1      = 0.25,
        xw2      = 0.65,
        T1_props = T1_props,
        T4_props = T4_props,
        LE_xz    = np.array([0.0, 0.0]),
        enable_logging = False,
    )


# ================================================================================
# PUBLIC API - Section property tests
# ================================================================================

def test_geom_properties_smoke(airfoil_arrays) -> None:
    '''Full pipeline returns a GeomData with sane section properties.'''
    gd = _build_calculator(airfoil_arrays).run()

    assert isinstance(gd, GeomData)
    assert gd.chord == _CHORD

    # Centroid X within the chord-length bounding box.
    assert 0.0 < float(gd.C[0]) < _CHORD

    # Inertias and torsion constant must be strictly positive.
    for name in ("I_XX", "I_ZZ", "I_1", "I_2", "J"):
        assert float(getattr(gd, name)) > 0.0, f"{name} not positive"


def test_geom_properties_cells_and_booms(airfoil_arrays) -> None:
    '''Three closed cells, seven booms, and positive section measures.'''
    gd = _build_calculator(airfoil_arrays).run()

    assert gd.A_cells.shape == (3,)
    assert np.all(gd.A_cells > 0.0)

    assert gd.boom_A.shape == (N_BOOMS,)
    assert np.all(gd.boom_A > 0.0)

    assert np.all(np.asarray(gd.s_k, dtype=float) > 0.0)
    assert len(gd.T1) == N_SEG_T1


def test_geom_properties_equivalent_moduli(airfoil_arrays) -> None:
    '''Equivalent axial and shear moduli are positive.'''
    gd = _build_calculator(airfoil_arrays).run()

    assert float(gd.E1_eq) > 0.0
    assert float(gd.G_eq) > 0.0


def test_geom_local_boom_frame_diagonalizes(airfoil_arrays) -> None:
    '''
    boom_y / boom_z (boom_u / boom_w rotated about the centroid by c_rad)
    express the booms in the beam-local principal axes, so the boom
    inertia tensor diagonalizes onto the principal inertias:
        sum(A * z^2) = I_2 (local y, minor)
        sum(A * y^2) = I_1 (local z, major)
        sum(A * y * z) ~ 0  (cross term vanishes)
    This is the frame-consistency condition that lets StressRecovery use
    local-frame moments against local boom coordinates.
    '''
    gd = _build_calculator(airfoil_arrays).run()

    assert gd.boom_y.shape == (N_BOOMS,)
    assert gd.boom_z.shape == (N_BOOMS,)

    A, y, z = gd.boom_A, gd.boom_y, gd.boom_z
    Iyy = float(np.sum(A * z**2))    # about local y -> minor I_2
    Izz = float(np.sum(A * y**2))    # about local z -> major I_1
    Iyz = float(np.sum(A * y * z))   # cross term -> must vanish

    np.testing.assert_allclose(Iyy, gd.I_2, rtol=1e-9)
    np.testing.assert_allclose(Izz, gd.I_1, rtol=1e-9)
    assert abs(Iyz) <= 1e-9 * gd.I_1


def test_geom_local_stress_constants(airfoil_arrays) -> None:
    '''
    Local-frame direct-stress constants match the principal-axis closed
    form: IXstar_loc = z / I_2 (coeff of local My) and
    IZstar_loc = y / I_1 (coeff of local Mz).
    '''
    gd = _build_calculator(airfoil_arrays).run()

    assert gd.IXstar_loc.shape == (N_BOOMS,)
    assert gd.IZstar_loc.shape == (N_BOOMS,)
    np.testing.assert_allclose(gd.IXstar_loc, gd.boom_z / gd.I_2, rtol=1e-12)
    np.testing.assert_allclose(gd.IZstar_loc, gd.boom_y / gd.I_1, rtol=1e-12)
