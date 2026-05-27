'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Load Distribution Plot Module.

Renders external (lift, drag, aerodynamic moment) and internal (shear
forces and bending / torsion moments) spanwise load distributions read
from the JSON databases produced by fem.loads.load_mapper.LoadMapper.

Each quantity is drawn as a 2-D plot of value versus spanwise coordinate
Y, with one curve per flight condition stored in the database.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================ Default Database Paths ================
from cl3o.paths import LOADS_DIR as _DFLT_LDS_DIR, OUTPUTS_DIR as _OUT
_DFLT_OUT_DIR = _OUT / "loads"

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# Loads
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData

# ================ Global variables ================
_EXL_TAG = "ExLoadsData"
_INL_TAG = "InLoadsData"

_FIGSIZE     = (12, 4)
_LINEWIDTH   = 1.6
_MARKERSIZE  = 4
_GRID_ALPHA  = 0.35
_SHADE_ALPHA = 0.10
_LABEL_FS    = 10
_TITLE_FS    = 11
_YPAD        = 0.1

_COLOR_CYCLE = [
    "#3350b7",
    "#d63b3b",
    "#2da84a",
    "#e88b00",
    "#6a1bbf",
    "#1a9b9b",
    "#b7337a",
    "#555555",
]

_MARKER_CYCLE = [
    "s",
    "o",
    "^",
    "v",
    "8",
    "p",
    "*",
    "d"
]


# ================================================================================
# Internal Helpers
# ================================================================================

class PlotLoadsHelper:

    def __init__(self):
        pass

    @staticmethod
    def style_axes(
        ax: plt.Axes,
        values: np.ndarray,
        xlabel: str,
        ylabel: str,
        b_mm: float,
    ) -> None:
        '''Applies consistent styling to a 2-D spanwise distribution axes.'''
        y_max = float(np.max(np.abs(values))) if values.size else 1.0
        if y_max <= 0.0:
            y_max = 1.0

        all_pos = bool(np.all(values >= 0.0))
        all_neg = bool(np.all(values <= 0.0))

        if all_pos:
            ax.set_ylim(-_YPAD * y_max, (1.0 + 2.0 * _YPAD) * y_max)
        elif all_neg:
            ax.set_ylim(-(1.0 + 2.0 * _YPAD) * y_max, _YPAD * y_max)
        else:
            ax.set_ylim(-(1.0 + _YPAD) * y_max, (1.0 + _YPAD) * y_max)

        ax.set_xlim(-b_mm / 2.0, b_mm / 2.0)

        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.axvline(0.0, color="black", linewidth=0.7, linestyle="-.", alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
        ax.grid(True, which="both", alpha=_GRID_ALPHA)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        save_path: str | Path,
        logger,
    ) -> None:
        '''Saves a figure to disk, creating parent directories as needed.'''
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        logger.info(f"Load distribution figure saved to {target}")


# ================================================================================
# PUBLIC API - 2-D spanwise distribution (single quantity, multi-condition)
# ================================================================================

def plot_distribution(
    Y              : np.ndarray,
    values         : np.ndarray,
    conditions     : list[str],
    b_mm           : float,
    ylabel         : str,
    title          : Optional[str]        = None,
    save_path      : Optional[str | Path] = None,
    show           : bool                 = False,
    shade          : bool                 = False,
    enable_logging : bool                 = True,
) -> plt.Figure:
    '''
    Plot a single spanwise quantity, overlaying one curve per condition.

    Args:
        Y         : Spanwise coordinate [mm], shape (n,).
        values    : Quantity values, shape (nc, n) or (n,).
        conditions: List of condition tags, length nc.
        b_mm      : Full wing span [mm] for symmetric x-axis limits.
        ylabel    : Y-axis label string (include units).
        title     : Optional figure title.
        save_path : Optional PNG/PDF output path.
        show      : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        The matplotlib Figure handle.
    '''
    logger = io.setup_logger("plot_distribution", enable_logging)

    Y   = np.asarray(Y, dtype=float)
    arr = np.atleast_2d(np.asarray(values, dtype=float))

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for k in range(arr.shape[0]):
        color  = _COLOR_CYCLE[k % len(_COLOR_CYCLE)]
        marker = _MARKER_CYCLE[k % len(_MARKER_CYCLE)]
        label = conditions[k] if k < len(conditions) else f"cond_{k}"
        ax.plot(
            Y, arr[k],
            color           = color,
            linewidth       = _LINEWIDTH,
            marker          = marker,
            markersize      = _MARKERSIZE,
            markerfacecolor = "white",
            markeredgecolor = color,
            markeredgewidth = 1.2,
            label           = label,
        )
        if shade:
            ax.fill_between(Y, arr[k], 0.0, color=color, alpha=_SHADE_ALPHA)

    PlotLoadsHelper.style_axes(
        ax,
        arr,
        xlabel = r"Posicao ao longo da envergadura  [mm]",
        ylabel = ylabel,
        b_mm   = b_mm,
    )

    if title is not None:
        ax.set_title(title, fontsize=_TITLE_FS)

    if not len(conditions) == 1:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    
    fig.tight_layout()

    if save_path is not None:
        PlotLoadsHelper.save_figure(fig, save_path, logger)

    if show:
        plt.show()

    return fig


# ================================================================================
# PUBLIC API - External loads
# ================================================================================

def plot_external_loads(
    exl_data       : ExLoadsData,
    b_mm           : float,
    out_dir        : Optional[str | Path] = None,
    show           : bool                 = False,
    enable_logging : bool                 = True,
) -> list[plt.Figure]:
    '''
    Render the three spanwise external load distributions stored in
    ExLoadsData: lift [N], drag [N] and aerodynamic moment [N*mm].

    Args:
        exl_data      : ExLoadsData container from LoadMapper.
        b_mm          : Full wing span [mm] for axis limits.
        out_dir       : Optional output directory. Files are saved as
                        "exloads_<quantity>.pdf" when provided.
        show          : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        List of matplotlib Figure handles [lift, drag, moment].
    '''
    logger = io.setup_logger("plot_external_loads", enable_logging)

    Y          = np.asarray(exl_data.Y, dtype=float)
    conditions = list(exl_data.conditions)

    specs = [
        (exl_data.lift,   "L", "Sustentacao",           r"[N]"),
        (exl_data.drag,   "D", "Arrasto",               r"[N]"),
        (exl_data.moment, "Ma", "Momento Aerodinamico", r"[N$\cdot$mm]"),
    ]

    figures: list[plt.Figure] = []
    for arr, name, title_tag, unit in specs:
        save_path = None
        if out_dir is not None:
            save_path = Path(out_dir) / f"{name}.pdf"

        fig = plot_distribution(
            Y              = Y,
            values         = arr,
            conditions     = conditions,
            b_mm           = b_mm,
            ylabel         = f"{title_tag} ${name}_i$ {unit}",
            title          = f"Distribuicao de {title_tag}",
            save_path      = save_path,
            show           = False,
            enable_logging = enable_logging,
        )
        figures.append(fig)

    logger.info(f"Rendered {len(figures)} external load distribution figures")

    if show:
        plt.show()

    return figures


# ================================================================================
# PUBLIC API - Internal loads
# ================================================================================

def plot_internal_loads(
    inl_data       : InLoadsData,
    b_mm           : float,
    out_dir        : Optional[str | Path] = None,
    show           : bool                 = False,
    enable_logging : bool                 = True,
) -> list[plt.Figure]:
    '''
    Render the six spanwise internal load distributions stored in
    InLoadsData:

        Ny  [N]      Axial force along Y
        Vx  [N]      Shear force along X (drag direction)
        Vz  [N]      Shear force along Z (lift direction)
        Mfx [N*mm]   Out-of-plane bending moment (flapwise)
        Mfz [N*mm]   In-plane bending moment (edgewise)
        Mty [N*mm]   Torsion about spanwise axis

    Args:
        inl_data      : InLoadsData container from LoadMapper.
        b_mm          : Full wing span [mm] for axis limits.
        out_dir       : Optional output directory. Files are saved as
                        "inloads_<quantity>.pdf" when provided.
        show          : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        List of matplotlib Figure handles, in spec order.
    '''
    logger = io.setup_logger("plot_internal_loads", enable_logging)

    Y          = np.asarray(inl_data.Y, dtype=float)
    conditions = list(inl_data.conditions)

    specs = [
        (inl_data.Ny,  "Ny",  r"Forca axial", r"$N_y$  [N]"),
        (inl_data.Vx,  "Vx",  r"Cortante em $X$", r"$V_X$  [N]"),
        (inl_data.Vz,  "Vz",  r"Cortante em $Z$", r"$V_Z$  [N]"),
        (inl_data.Mfx, "Mfx", r"Momento fletor em torno de $X$", r"$M_{f_X}$  [N$\cdot$mm]"),
        (inl_data.Mfz, "Mfz", r"Momento fletor em torno de $Z$", r"$M_{f_Z}$  [N$\cdot$mm]"),
        (inl_data.Mty, "Mty", r"Torque", r"$T$  [N$\cdot$mm]"),
    ]

    figures: list[plt.Figure] = []
    for arr, name, title, ylabel in specs:
        save_path = None
        if out_dir is not None:
            save_path = Path(out_dir) / f"{name}.pdf"

        fig = plot_distribution(
            Y              = Y,
            values         = arr,
            conditions     = conditions,
            b_mm           = b_mm,
            ylabel         = f"{ylabel}",
            title          = f"{title}",
            save_path      = save_path,
            show           = False,
            enable_logging = enable_logging,
        )
        figures.append(fig)

    logger.info(f"Rendered {len(figures)} internal load distribution figures")

    if show:
        plt.show()

    return figures


# ================================================================================
# PUBLIC API - Convenience loader
# ================================================================================

def plot_loads_from_database(
    aircraft_name  : str,
    b_mm           : float,
    lds_dir        : Optional[str | Path] = None,
    out_dir        : Optional[str | Path] = None,
    show           : bool                 = False,
    enable_logging : bool                 = True,
) -> tuple[list[plt.Figure], list[plt.Figure]]:
    '''
    Load both ExLoadsData and InLoadsData for an aircraft and render
    the full set of external and internal distribution plots.

    Args:
        aircraft_name : Aircraft tag, matching the JSON filename prefix
                        (e.g. "da62" -> "da62_ExLoadsData.json").
        b_mm          : Full wing span [mm].
        lds_dir       : Directory where the load JSONs live.
                        Defaults to <ROOT>/data/loads.
        out_dir       : Output directory for the generated PDFs.
                        Defaults to <ROOT>/outputs/loads/<aircraft>.
        show          : Whether to call plt.show() at the end.
        enable_logging: Toggle logger.

    Returns:
        Tuple (external_figures, internal_figures).
    '''
    logger = io.setup_logger("plot_loads_from_database", enable_logging)

    lds_dir = Path(lds_dir) if lds_dir is not None else _DFLT_LDS_DIR
    out_dir = Path(out_dir) if out_dir is not None else _DFLT_OUT_DIR / aircraft_name

    exl_path = lds_dir / f"{aircraft_name.lower()}_{_EXL_TAG}.json"
    inl_path = lds_dir / f"{aircraft_name.lower()}_{_INL_TAG}.json"

    logger.info(f"Reading external loads from: {exl_path}")
    exl_data = io.read_json(filepath=exl_path, dcls=ExLoadsData)

    logger.info(f"Reading internal loads from: {inl_path}")
    inl_data = io.read_json(filepath=inl_path, dcls=InLoadsData)

    ex_figs = plot_external_loads(
        exl_data       = exl_data,
        b_mm           = b_mm,
        out_dir        = out_dir,
        show           = False,
        enable_logging = enable_logging,
    )
    in_figs = plot_internal_loads(
        inl_data       = inl_data,
        b_mm           = b_mm,
        out_dir        = out_dir,
        show           = False,
        enable_logging = enable_logging,
    )

    if show:
        plt.show()

    return ex_figs, in_figs


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    aircraft_name = "da62"
    b_mm          = 12969.0

    plot_loads_from_database(
        aircraft_name = aircraft_name,
        b_mm          = b_mm,
        show          = True,
    )
