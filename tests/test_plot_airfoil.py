'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Airfoil Plot Smoke Tests.

Headless (Agg) smoke test that plot_airfoil returns a Matplotlib Figure
for a real AirfoilData record without opening a window.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import matplotlib.pyplot as plt

# ================ Module imports ================
from cl3o.geometry.airfoil   import AirfoilData
from cl3o.results.plot_airfoil import plot_airfoil


# ================================================================================
# PUBLIC API - Plot smoke test
# ================================================================================

def test_plot_airfoil_returns_figure(airfoil_arrays) -> None:
    '''plot_airfoil produces a Figure and does not block.'''
    afl = AirfoilData(**airfoil_arrays)
    fig = plot_airfoil(afl, show=False, enable_logging=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
