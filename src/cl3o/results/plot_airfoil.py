'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Airfoil Plot Module.

Overlays the upper, lower and camber polylines of an AirfoilData record
on a single equal-aspect 2-D figure. Produces a screen figure and
optionally writes the rendering to disk.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# ================ Pathing ================


# ================ Module imports ================
from cl3o.utils import io_utils as io
from cl3o.geometry.airfoil import AirfoilData

# ================ Global variables ================
_COLOR_UPPER  = "#48aa1a"
_COLOR_LOWER  = "#ab44ab"
_COLOR_CAMBER = "#04aafb"


# ================================================================================
# PUBLIC API - Plot an AirfoilData record
# ================================================================================

def plot_airfoil(
    afl_data       : AirfoilData,
    title          : Optional[str]        = None,
    save_path      : Optional[str | Path] = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Plot an AirfoilData record with upper, lower and camber polylines.

    Args:
        afl_data : Parsed airfoil data container.
        title    : Optional figure title. Defaults to a blank title.
        save_path: Optional path to write the figure as PNG/PDF.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_airfoil", enable_logging)

    xu = np.asarray(afl_data.x_upper,  dtype=float)
    yu = np.asarray(afl_data.y_upper,  dtype=float)
    xl = np.asarray(afl_data.x_lower,  dtype=float)
    yl = np.asarray(afl_data.y_lower,  dtype=float)
    xc = np.asarray(afl_data.x_camber, dtype=float)
    yc = np.asarray(afl_data.y_camber, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(
        xu, yu,
        color      = _COLOR_UPPER,
        marker     = "s",
        markersize = 3,
        linewidth  = 1.2,
        label      = "Upper",
    )
    ax.plot(
        xl, yl,
        color      = _COLOR_LOWER,
        marker     = "s",
        markersize = 3,
        linewidth  = 1.2,
        label      = "Lower",
    )
    ax.plot(
        xc, yc,
        color     = _COLOR_CAMBER,
        linestyle = "-.",
        linewidth = 1.0,
        label     = "Camber line",
    )

    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Airfoil figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
