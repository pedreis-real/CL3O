'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing Outline Plot Module.

Renders the wing planform as either a 2-D LE/TE outline or a 3-D
visualization (leading edge, trailing edge, and chord lines at each
control point). All coordinates follow the CL3O convention: +Y span,
+X chord, +Z vertical.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                         # noqa: F401

# ================ Pathing ================


# ================ Module imports ================
from cl3o.utils import io_utils as io
from cl3o.geometry.wing import WingData, LerpWingData

# ================ Global variables ================
_COLOR_LE    = "#1a3a6b"
_COLOR_TE    = "#6b1a1a"
_COLOR_CHORD = "#555555"
_COLOR_QC    = "#e88b00"


# ================================================================================
# Internal Helpers
# ================================================================================

class PlotWingHelper:
    def __init__(self):
        pass

    @staticmethod
    def set_equal_aspect_3d(ax, pts: np.ndarray) -> None:
        '''Equalize the three axis ranges around the data centroid.'''
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = float((mx - mn).max())
        if span <= 0.0:
            span = 1.0
        mid  = 0.5 * (mn + mx)
        r    = 0.5 * span
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)


# ================================================================================
# PUBLIC API - 2-D LE/TE planform
# ================================================================================

def plot_wing_outline_2d(
    wing_data      : WingData,
    title          : str | None        = None,
    save_path      : str | Path | None = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    2-D planform view (XY plane) of the leading and trailing edges.

    Args:
        wing_data: WingData container (expects x_le, z_le, x_te, z_te, pos).
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_wing_2d", enable_logging)

    pos  = np.asarray(wing_data.pos,  dtype=float)
    x_le = np.asarray(wing_data.x_le, dtype=float)
    x_te = np.asarray(wing_data.x_te, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(pos, x_le, color=_COLOR_LE, linewidth=1.5, label="LE")
    ax.plot(pos, x_te, color=_COLOR_TE, linewidth=1.5, label="TE")

    for k in range(pos.size):
        ax.plot(
            [pos[k], pos[k]],
            [x_le[k], x_te[k]],
            color=_COLOR_CHORD,
            linewidth=0.6,
            alpha=0.6,
        )

    ax.set_xlabel("Span Y [mm]")
    ax.set_ylabel("Chord X [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.grid(which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Wing 2-D figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - 3-D wing outline
# ================================================================================

def plot_wing_outline_3d(
    wing_data      : WingData,
    lerp_wing      : LerpWingData | None = None,
    title          : str | None          = None,
    save_path      : str | Path | None   = None,
    show           : bool                   = True,
    enable_logging : bool                   = True,
) -> plt.Figure:
    '''
    3-D wing outline: LE/TE polylines, chord lines at each cpt and
    (when provided) the interpolated LE/TE polyline from LerpWingData.

    Args:
        wing_data: WingData with LE/TE coordinates at cpts.
        lerp_wing: Optional LerpWingData for the dense interpolated mesh.
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_wing_3d", enable_logging)

    pos  = np.asarray(wing_data.pos,  dtype=float)
    x_le = np.asarray(wing_data.x_le, dtype=float)
    z_le = np.asarray(wing_data.z_le, dtype=float)
    x_te = np.asarray(wing_data.x_te, dtype=float)
    z_te = np.asarray(wing_data.z_te, dtype=float)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    # -------- 1. LE and TE polylines at cpts --------
    ax.plot(
        x_le, pos, z_le,
        color     = _COLOR_LE,
        linewidth = 1.8,
        label     = "LE",
    )
    ax.plot(
        x_te, pos, z_te,
        color     = _COLOR_TE,
        linewidth = 1.8,
        label     = "TE",
    )

    # -------- 2. Chord lines at each cpt --------
    for k in range(pos.size):
        ax.plot(
            [x_le[k], x_te[k]],
            [pos[k],  pos[k] ],
            [z_le[k], z_te[k]],
            color     = _COLOR_CHORD,
            linewidth = 0.8,
            alpha     = 0.7,
        )

    # -------- 3. Optional dense LerpWingData outline --------
    if lerp_wing is not None:
        LE = np.asarray(lerp_wing.LE, dtype=float)
        TE = np.asarray(lerp_wing.TE, dtype=float)
        ax.plot(
            LE[:, 0], LE[:, 1], LE[:, 2],
            color     = _COLOR_LE,
            linewidth = 0.8,
            linestyle = "--",
            alpha     = 0.6,
            label     = "LE (interp)",
        )
        ax.plot(
            TE[:, 0], TE[:, 1], TE[:, 2],
            color     = _COLOR_TE,
            linewidth = 0.8,
            linestyle = "--",
            alpha     = 0.6,
            label     = "TE (interp)",
        )

    # -------- 4. Quarter-chord line at cpts --------
    x_qc = x_le + 0.25 * (x_te - x_le)
    z_qc = z_le + 0.25 * (z_te - z_le)
    ax.plot(
        x_qc, pos, z_qc,
        color     = _COLOR_QC,
        linewidth = 1.2,
        linestyle = "-.",
        label     = "c/4",
    )

    # -------- 5. Styling --------
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend(loc="best")
    if title is not None:
        ax.set_title(title)

    all_pts = np.column_stack([
        np.concatenate([x_le, x_te]),
        np.concatenate([pos,  pos ]),
        np.concatenate([z_le, z_te]),
    ])
    PlotWingHelper.set_equal_aspect_3d(ax, all_pts)

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Wing 3-D figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
