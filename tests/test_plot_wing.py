'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing Plot Smoke Tests.

Headless smoke tests for the 2-D and 3-D wing-outline plotters against the
real DA62 WingData / LerpWingData.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import matplotlib.pyplot as plt
import pytest

# ================ Module imports ================
from cl3o.results.plot_wing import plot_wing_outline_2d, plot_wing_outline_3d

pytestmark = pytest.mark.slow


# ================================================================================
# PUBLIC API - Plot smoke tests
# ================================================================================

def test_plot_wing_2d(static) -> None:
    '''2-D planform plotter returns a Figure.'''
    fig = plot_wing_outline_2d(static.wing_db, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_wing_3d(static) -> None:
    '''3-D outline plotter returns a Figure (with interpolated stations).'''
    fig = plot_wing_outline_3d(
        static.wing_db,
        lerp_wing      = static.lerp_wing_db,
        show           = False,
        enable_logging = False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
