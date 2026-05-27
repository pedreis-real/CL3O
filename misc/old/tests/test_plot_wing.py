'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Wing Outline Plot Tests Module.

Phase I regression tests for results.plot_wing:

  - Smoke-test 2-D LE/TE planform rendering from a synthetic WingData.
  - Smoke-test 3-D outline with and without a LerpWingData overlay.
  - PlotWingHelper.set_equal_aspect_3d equalises x/y/z span around the
    centroid of the input point cloud.

Notes
-----
The disk JSON at data/wings/da62_WingData.json is stale (missing
root_off); this suite therefore constructs WingData in-memory. If the
database is refreshed with the current Wing builder, the fixture
builder here can be swapped for io.read_json without changing the
assertions.

@ CWSS Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ================ Module imports ================
from geometry.wing     import WingData, LerpWingData
from results.plot_wing import (
    plot_wing_outline_2d, plot_wing_outline_3d, PlotWingHelper,
)


# ================================================================================
# PRIVATE API - Synthetic wing builders
# ================================================================================

def _make_wing_data(n: int = 4) -> WingData:
    '''Trapezoidal wing with two taper breaks roughly matching DA62.'''
    pos    = np.array([0.0, 623.0, 2035.0, 6484.0])
    chords = np.array([1728.0, 1728.0, 1316.0, 915.0])
    x_le   = np.array([0.0, 0.0, 200.0, 600.0])
    x_te   = x_le + chords
    z_le   = np.array([0.0, 0.0, 100.0, 500.0])
    z_te   = z_le.copy()

    return WingData(
        n_cpts   = n,
        b     = 12969.0,
        cr       = 1728.0,
        root_off = 0.0,
        afl_lst  = ["w"] * n,
        area     = 2.0e7,
        AR       = 8.0,
        mgc      = 1400.0,
        mac      = 1400.0,
        taper    = chords / chords[0],
        pci      = pos / pos[-1],
        chords   = chords,
        pos      = pos,
        twist    = np.zeros(n),
        sweep    = np.zeros(n),
        sweep_LE = np.zeros(n),
        dihedral = np.zeros(n),
        ds       = np.zeros(n),
        dc       = np.zeros(n),
        sec_span = np.zeros(n),
        x_le     = x_le,
        z_le     = z_le,
        x_te     = x_te,
        z_te     = z_te,
    )


def _make_lerp_wing(wing: WingData, n_sta: int = 20) -> LerpWingData:
    '''Densely interpolate the cpt polylines for the 3-D overlay.'''
    Y = np.linspace(wing.pos[0], wing.pos[-1], n_sta)
    LE = np.column_stack([
        np.interp(Y, wing.pos, wing.x_le),
        Y,
        np.interp(Y, wing.pos, wing.z_le),
    ])
    TE = np.column_stack([
        np.interp(Y, wing.pos, wing.x_te),
        Y,
        np.interp(Y, wing.pos, wing.z_te),
    ])
    chord = np.interp(Y, wing.pos, wing.chords)
    twist = np.zeros(n_sta)
    return LerpWingData(
        Y_cp    = wing.pos,
        Y_sta   = Y,
        LE      = LE,
        TE      = TE,
        chord   = chord,
        twist   = twist,
        afl_lst = wing.afl_lst,
    )


# ================================================================================
# PUBLIC API - Tests
# ================================================================================

def test_plot_wing_2d_renders(tmp_path: Path) -> None:
    '''2-D planform renders and writes a non-trivial PNG.'''
    wing = _make_wing_data()
    out  = tmp_path / "wing2d.png"

    fig = plot_wing_outline_2d(
        wing_data      = wing,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_plot_wing_3d_without_lerp(tmp_path: Path) -> None:
    '''3-D outline works even when LerpWingData is not supplied.'''
    wing = _make_wing_data()
    out  = tmp_path / "wing3d.png"

    fig = plot_wing_outline_3d(
        wing_data      = wing,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_plot_wing_3d_with_lerp_overlay(tmp_path: Path) -> None:
    '''3-D outline accepts a LerpWingData and renders the interp curves.'''
    wing = _make_wing_data()
    lerp = _make_lerp_wing(wing, n_sta=25)
    out  = tmp_path / "wing3d_lerp.png"

    fig = plot_wing_outline_3d(
        wing_data      = wing,
        lerp_wing      = lerp,
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_equal_aspect_3d_centers_limits() -> None:
    '''The helper centers x/y/z limits on the data centroid with equal span.'''
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 4.0, 0.0],
        [0.0,  4.0, 2.0],
    ])
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    PlotWingHelper.set_equal_aspect_3d(ax, pts)

    xl = ax.get_xlim(); yl = ax.get_ylim(); zl = ax.get_zlim()
    span = float(max(
        np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]),
    ))
    for lim in (xl, yl, zl):
        assert np.isclose(lim[1] - lim[0], span)
    plt.close(fig)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # pytest.main([__file__, "-q"])
    wing = _make_wing_data()
    lerp = _make_lerp_wing(wing, n_sta=25)
    plot_wing_outline_2d(wing_data=wing, enable_logging=False)
    plot_wing_outline_3d(wing_data=wing, enable_logging=False)
    plot_wing_outline_3d(wing_data=wing, lerp_wing=lerp, enable_logging=False)
    plt.show()
