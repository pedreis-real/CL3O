'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing Module Tests.

Smoke tests for the WingData / Wing public surface, focused on the
lerp_wing_geometry interpolation path. The current implementation folds
any Y_sta input onto the left wing (Y <= 0) and stacks root -> tip, so
the assertions here target convention-independent invariants (shapes,
constant chord recovery, twist span, positive chord).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np
import pytest

# ================ Module imports ================
from cl3o.geometry.wing import Wing, WingData, LerpWingData


# ================================================================================
# PRIVATE API - Fixtures
# ================================================================================

def _make_wing_data() -> WingData:
    '''Straight, constant-chord wing with two control points and a
    linear -2 deg washout from root to tip.'''
    return WingData(
        n_cpts   = 2,
        b        = 10000.0,
        cr       = 1000.0,
        root_off = 0.0,
        afl_lst  = ["Airfoil0", "Airfoil1"],
        area     = 5000000.0,
        AR       = 20.0,
        mgc      = 500.0,
        mac      = 500.0,
        taper    = np.array([1.0, 1.0]),
        pci      = np.array([0.0, 1.0]),
        chords   = np.array([1000.0, 1000.0]),
        pos      = np.array([0.0, 5000.0]),
        twist    = np.array([0.0, -2.0]),
        sweep    = np.array([0.0, 0.0]),
        sweep_LE = np.array([0.0, 0.0]),
        dihedral = np.array([0.0, 0.0]),
        ds       = np.array([5000.0]),
        dc       = np.array([0.0]),
        sec_span = np.array(10000.0),
        x_le     = np.array([0.0, 500.0]),
        z_le     = np.array([0.0, 0.0]),
        x_te     = np.array([1000.0, 1500.0]),
        z_te     = np.array([0.0, 0.0]),
    )


class _WingStub:
    '''Minimal object exposing the .wng_data attribute Wing reads.'''
    def __init__(self, wng_data: WingData) -> None:
        self.wng_data = wng_data


# ================================================================================
# PUBLIC API - Interpolation tests
# ================================================================================

def test_lerp_returns_well_shaped_container() -> None:
    '''lerp_wing_geometry returns a LerpWingData with consistent shapes.'''
    wng = _WingStub(_make_wing_data())
    Y = np.array([0.0, -5000.0])

    data = Wing.lerp_wing_geometry(wng, Y)

    assert isinstance(data, LerpWingData)
    assert data.n_sta == 2
    assert data.LE.shape == (2, 3)
    assert data.TE.shape == (2, 3)
    assert data.chord.shape == (2,)
    assert data.twist.shape == (2,)


def test_lerp_constant_chord_recovered() -> None:
    '''A constant-chord wing must interpolate to a constant chord.'''
    wng = _WingStub(_make_wing_data())
    Y = np.linspace(-5000.0, 0.0, 11)

    data = Wing.lerp_wing_geometry(wng, Y)

    np.testing.assert_allclose(data.chord, 1000.0)
    # Trailing edge sits aft of the leading edge at every station.
    assert np.all(data.TE[:, 0] > data.LE[:, 0])


def test_lerp_twist_span_preserved() -> None:
    '''Interpolated twist spans the full root -> tip washout (in rad).'''
    wng = _WingStub(_make_wing_data())
    Y = np.linspace(-5000.0, 0.0, 11)

    data = Wing.lerp_wing_geometry(wng, Y)

    assert np.isclose(data.twist.max(), 0.0, atol=1e-6)
    assert np.isclose(data.twist.min(), np.radians(-2.0), atol=1e-4)


def test_lerp_midspan_between_endpoints() -> None:
    '''A single midspan station lands strictly between the cpt outlines.'''
    wng = _WingStub(_make_wing_data())
    data = Wing.lerp_wing_geometry(wng, np.array([-2500.0]))

    assert data.n_sta == 1
    assert 0.0 <= data.LE[0, 0] <= 500.0
    assert 1000.0 <= data.TE[0, 0] <= 1500.0
    np.testing.assert_allclose(data.chord, [1000.0])
