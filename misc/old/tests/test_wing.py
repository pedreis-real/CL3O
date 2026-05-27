'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Wing Module Tests.

Smoke tests for WingData/Wing public surface, focused on the
lerp_wing_geometry interpolation path that was rewritten during the
pre-Phase-I audit.

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
from geometry.wing import Wing, WingData, LerpWingData


# ================================================================================
# PRIVATE API - Fixture builder
# ================================================================================

def _make_wing_data() -> WingData:
    '''
    Tapered straight wing, two control points at Y=0 mm and Y=5000 mm.

    LE at (0, 0, 0) and (500, 5000, 0); TE at (1000, 0, 0) and
    (1500, 5000, 0). Chord drops linearly from 1000 mm to 1000 mm.
    '''
    return WingData(
        n_cpts   = 2,
        b     = 10000.0,
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


# ================================================================================
# PUBLIC API - Wing interpolation tests
# ================================================================================

class _WingStub:
    '''Minimal stub that mimics Wing just enough for lerp_wing_geometry.'''
    def __init__(self, wng_data: WingData) -> None:
        self.wng_data = wng_data


def test_lerp_wing_geometry_endpoints() -> None:
    '''
    lerp_wing_geometry must return the exact control-point values at
    Y_sta = [0, span] without error.
    '''
    wng = _WingStub(_make_wing_data())
    Y_sta = np.array([0.0, 5000.0])

    data = Wing.lerp_wing_geometry(wng, Y_sta)

    assert isinstance(data, LerpWingData)
    assert data.n_sta == 2
    assert data.LE.shape == (2, 3)
    assert data.TE.shape == (2, 3)
    assert data.chord.shape == (2,)
    assert data.twist.shape == (2,)

    np.testing.assert_allclose(data.LE[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.LE[1], [500.0, 5000.0, 0.0])
    np.testing.assert_allclose(data.TE[0], [1000.0, 0.0, 0.0])
    np.testing.assert_allclose(data.TE[1], [1500.0, 5000.0, 0.0])
    np.testing.assert_allclose(data.chord, [1000.0, 1000.0])


def test_lerp_wing_geometry_midspan_interpolation() -> None:
    '''Midspan station must recover the linearly-interpolated outline.'''
    wng = _WingStub(_make_wing_data())
    Y_sta = np.array([2500.0])

    data = Wing.lerp_wing_geometry(wng, Y_sta)

    np.testing.assert_allclose(data.LE[0], [250.0, 2500.0, 0.0])
    np.testing.assert_allclose(data.TE[0], [1250.0, 2500.0, 0.0])
    np.testing.assert_allclose(data.chord, [1000.0])
    np.testing.assert_allclose(data.twist, [np.radians(-1.0)], atol=1e-4)


def test_lerp_wing_geometry_many_stations() -> None:
    '''Interpolating to n_sta >> n_cpts must preserve monotonic twist.'''
    wng = _WingStub(_make_wing_data())
    Y_sta = np.linspace(0.0, 5000.0, 11)

    data = Wing.lerp_wing_geometry(wng, Y_sta)

    assert data.LE.shape == (11, 3)
    assert data.TE.shape == (11, 3)
    assert np.all(np.diff(data.twist) <= 1e-9)
    np.testing.assert_allclose(data.twist[0],  0.0)
    np.testing.assert_allclose(data.twist[-1], np.radians(-2.0), atol=1e-4)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
