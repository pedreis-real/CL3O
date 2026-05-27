'''
================================================================================
CWSS - Composite Wing Structural Sizing.
Airfoil Plot Tests Module.

Phase I regression tests for results.plot_airfoil.plot_airfoil:

  - A synthetic NACA-style AirfoilData renders without error and the
    saved PNG is non-empty.
  - Equal-aspect axes are preserved (ax.get_aspect() == 'equal').

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
from geometry.airfoil     import AirfoilData
from results.plot_airfoil import plot_airfoil


# ================================================================================
# PRIVATE API - Synthetic NACA-like airfoil
# ================================================================================

def _make_naca_like(n: int = 40, t: float = 0.12) -> AirfoilData:
    '''
    Build a symmetric NACA-like AirfoilData from the thickness polynomial
    yt = 5t * (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3
              - 0.1015 x^4).
    '''
    x  = np.linspace(0.0, 1.0, n)
    yt = 5.0 * t * (
          0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    return AirfoilData(
        x_upper  = x.copy(),
        y_upper  = +yt,
        x_lower  = x.copy(),
        y_lower  = -yt,
        x_camber = x.copy(),
        y_camber = np.zeros_like(x),
    )


# ================================================================================
# PUBLIC API - Tests
# ================================================================================

def test_plot_airfoil_renders(tmp_path: Path) -> None:
    '''Synthetic airfoil renders and writes a non-trivial PNG.'''
    afl = _make_naca_like()
    out = tmp_path / "airfoil.png"

    fig = plot_airfoil(
        afl_data       = afl,
        title          = "NACA 0012-like",
        save_path      = out,
        show           = False,
        enable_logging = False,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    plt.close(fig)


def test_plot_airfoil_uses_equal_aspect() -> None:
    '''Aspect ratio must be 'equal' to avoid visual thickness distortion.'''
    afl = _make_naca_like()
    fig = plot_airfoil(
        afl_data       = afl,
        show           = False,
        enable_logging = False,
    )
    ax = fig.axes[0]
    # matplotlib normalises "equal" to 1.0 on 2D axes
    aspect = ax.get_aspect()
    assert aspect == "equal" or np.isclose(float(aspect), 1.0)
    plt.close(fig)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # pytest.main([__file__, "-q"])
    afl = _make_naca_like()
    fig = plot_airfoil(afl_data=afl, title="NACA 0012-like", enable_logging=False)
    plt.show()
