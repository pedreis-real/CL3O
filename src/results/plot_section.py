'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Cross-Section Plot Module.

Renders a 3-cell idealized cross-section at a single spanwise station
from a GeomData record. Annotates the four booms (with marker size
proportional to area), the centroid, and the shear centre, all in the
global XZ frame.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent            # src/results/
_SRC  = _HERE.parent                               # src/

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Module imports ================
from utils import io_utils as io
from geometry.geom_properties import GeomData

# ================ Global variables ================
_COLOR_SKIN     = "#1a3a6b"
_COLOR_WEB      = "#6b1a1a"
_COLOR_BOOM     = "#e88b00"
_COLOR_CENTROID = "#2da84a"
_COLOR_SC       = "#b03f8f"
_COLOR_CELL_EDG = "#bfbfbf"


# ================================================================================
# Internal Helpers
# ================================================================================

class PlotSectionHelper:
    def __init__(self):
        pass

    @staticmethod
    def boom_marker_sizes(
        boom_A   : np.ndarray,
        min_size : float = 20.0,
        max_size : float = 240.0,
    ) -> np.ndarray:
        '''Scale the 4 boom areas to marker sizes on [min_size, max_size].'''
        A = np.asarray(boom_A, dtype=float)
        A_max = float(A.max()) if A.size > 0 else 0.0
        if A_max <= 0.0:
            return np.full_like(A, min_size)
        return min_size + (max_size - min_size) * (A / A_max)

    @staticmethod
    def draw_T1_skins(ax, T1: list, color: str) -> None:
        '''Plot every T1 segment as a polyline in the XZ frame.'''
        for seg in T1:
            pts = np.asarray(seg["pts"], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2)

    @staticmethod
    def draw_T3_cells(ax, T3: list, color: str) -> None:
        '''Plot the three closed-cell polygons as dashed outlines.'''
        for cell in T3:
            pts = np.asarray(cell["pts"], dtype=float)
            if pts.size == 0:
                continue
            x_cl = np.append(pts[:, 0], pts[0, 0])
            z_cl = np.append(pts[:, 1], pts[0, 1])
            ax.plot(
                x_cl, z_cl,
                color     = color,
                linewidth = 0.6,
                linestyle = "--",
                alpha     = 0.9,
            )


# ================================================================================
# PUBLIC API - Plot a single station
# ================================================================================

def plot_section(
    geom_data      : GeomData,
    ax             : Optional[plt.Axes]   = None,
    title          : Optional[str]        = None,
    save_path      : Optional[str | Path] = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Draw a cross-section at a single station in the global XZ frame.

    Args:
        geom_data: Per-station geometry container.
        ax       : Optional pre-existing Axes to draw into. If None, a
            new figure is created.
        title    : Optional figure title.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_section", enable_logging)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # -------- 1. Skin polylines (T1) and cell outlines (T3) --------
    PlotSectionHelper.draw_T3_cells(ax, geom_data.T3, _COLOR_CELL_EDG)
    PlotSectionHelper.draw_T1_skins(ax, geom_data.T1, _COLOR_SKIN)

    # Spar webs are already part of T1 (seg6, seg7) and rendered by
    # draw_T1_skins above, so no separate web overlay is needed.

    # -------- 3. Booms (global XZ) --------
    Xc = float(geom_data.C[0])
    Zc = float(geom_data.C[2])

    bu = np.asarray(geom_data.boom_u, dtype=float)
    bw = np.asarray(geom_data.boom_w, dtype=float)
    bA = np.asarray(geom_data.boom_A, dtype=float)

    x_boom = Xc + bu
    z_boom = Zc + bw

    sizes = PlotSectionHelper.boom_marker_sizes(bA)
    ax.scatter(
        x_boom, z_boom,
        s      = sizes,
        color  = _COLOR_BOOM,
        edgecolor = "black",
        linewidth = 0.6,
        zorder = 5,
        label  = "Booms",
    )
    for k in range(x_boom.size):
        ax.annotate(
            f"B{k + 1}",
            xy         = (x_boom[k], z_boom[k]),
            xytext     = (6, 6),
            textcoords = "offset points",
            fontsize   = 8,
        )

    # -------- 4. Centroid and shear centre --------
    ax.scatter(
        [Xc], [Zc],
        marker    = "+",
        color     = _COLOR_CENTROID,
        s         = 120,
        linewidth = 2.0,
        zorder    = 6,
        label     = "Centroid",
    )

    us = float(geom_data.S_uvw[0])
    ws = float(geom_data.S_uvw[2])
    x_sc = Xc + us
    z_sc = Zc + ws
    ax.scatter(
        [x_sc], [z_sc],
        marker    = "x",
        color     = _COLOR_SC,
        s         = 110,
        linewidth = 2.0,
        zorder    = 6,
        label     = "Shear center",
    )

    # -------- 5. Styling --------
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Station Y = {float(geom_data.C[1]):.1f} mm"
            f"  |  chord = {float(geom_data.chord):.1f} mm"
        )

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Section figure saved to {target}")

    if show and created_fig:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - Plot a full spanwise sweep of sections
# ================================================================================

def plot_all_sections(
    sec_data       : list[GeomData],
    n_cols         : int                  = 2,
    title          : Optional[str]        = None,
    save_path      : Optional[str | Path] = None,
    show           : bool                 = True,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Arrange all stations on a grid of subplots, one cross-section each.

    Args:
        sec_data : Ordered list of GeomData (root -> tip).
        n_cols   : Number of subplot columns.
        title    : Optional figure suptitle.
        save_path: Optional PNG/PDF output path.
        show     : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_all_sections", enable_logging)

    n = len(sec_data)
    if n == 0:
        raise ValueError("[CL3O] plot_all_sections received an empty list.")

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize = (6.0 * n_cols, 3.5 * n_rows),
        squeeze = False,
    )

    for k, gd in enumerate(sec_data):
        r, c = divmod(k, n_cols)
        plot_section(
            geom_data      = gd,
            ax             = axes[r, c],
            show           = False,
            enable_logging = False,
        )

    # Hide any unused subplots
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Multi-section figure saved to {target}")

    if show:
        plt.show()

    return fig


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
