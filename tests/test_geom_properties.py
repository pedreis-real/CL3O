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

# ================ Module imports ================
from cl3o.Constants import N_BOOMS, N_SEG_T1, N_FLANGES
from cl3o.geometry.geom_properties import GeomPropCalculator, GeomData


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
