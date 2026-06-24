'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Cross-Section Validation Module.

Instantiates GeomPropCalculator with a reference cross-section, runs the
full pipeline, prints a property summary, and plots the idealized section
with centroid, shear center, boom positions, and T1 segments.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
import json
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# ================ Default Database Paths ================
from cl3o.paths import AIRFOILS_DIR as _DFLT_AFL_PATH, OUTPUTS_DIR as _DFLT_OUT_PATH

# ================ Module imports ================
# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.geom_properties import GeomPropCalculator, GeomData

# ================ Global variables ================
_SHOW_PLOTS = False   # default headless; figures are saved to disk. Set True to display.
_ENBLE_LOG = True
_RECALC_PROPS = True
_USE_BOOM_CENTROID = False
_LOAD_FROM_DB = True

_XW1_RANGE = [0.10, 0.40]
_XW2_RANGE = [0.41, 0.75]
_N_TEST_SAMPLE = 100

_DFLT_FACTORY = np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE])

_FILEPATH = _DFLT_OUT_PATH / "cross_section_validation.json"

# Seção 1
_AFL = "e169_AirfoilData.json"
_CHORD  = 300.0
_TWIST  = 0.0
_Y_STA  = 0.0
_XW1    = 60.0  / _CHORD
# _XW1    = np.linspace(_XW1_RANGE[0], _XW1_RANGE[1], _N_TEST_SAMPLE)
_XW2    = 210.0 / _CHORD
# _XW2    = np.linspace(_XW2_RANGE[0], _XW2_RANGE[1], _N_TEST_SAMPLE)
_E1     = 200_000.0
_E2     =  10_000.0
_E1B    = _E1        # homogeneous: bending == membrane
_E2B    = _E2
_G      =   6_000.0
_T_SEG     = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 4.0, 2.0])
_T_FLANGE  = np.array([3.0, 2.5, 3.0, 2.0])
# _T_FLANGE  = np.full(4, 2.0)
_BF        = np.array([12.0, 10.0, 8.0, 8.0])
# _BF        = np.full(4, 10.0)

# Seção 2
# _AFL = "wortmannfx63137_AirfoilData.json"
# _CHORD  = 300.0
# _TWIST  = 0.0
# _Y_STA  = 0.0
# _XW1    = 90.0 / _CHORD
# # _XW1    = np.linspace(_XW1_RANGE[0], _XW1_RANGE[1], _N_TEST_SAMPLE)
# _XW2    = 170.0 / _CHORD
# # _XW2    = np.linspace(_XW2_RANGE[0], _XW2_RANGE[1], _N_TEST_SAMPLE)
# _E1     = 200_000.0
# _E2     =  10_000.0
# _G      =   6_000.0
# _E1B    = _E1        # homogeneous: bending == membrane
# _E2B    = _E2
# _T_SEG     = np.array([4, 2, 2, 2, 2, 3, 2], dtype=float)
# _T_FLANGE  = np.array([4, 2.5, 2, 2], dtype=float)
# # _T_FLANGE  = np.full(4, 2.0)
# _BF        = np.array([16, 12, 7, 12], dtype=float)
# # _BF        = np.full(4, 10.0)

_PROP_LABELS: dict[str, str] = {
    # 'A'   : 'A [mm^2]',
    'Xc'  : 'Xc [mm]',
    'Zc'  : 'Zc [mm]',
    'I_XX': 'I_XX [mm^4]',
    'I_ZZ': 'I_ZZ [mm^4]',
    'I_XZ': 'I_XZ [mm^4]',
    'J'   : 'J [mm^4]',
    # 'I_1' : 'I_1 [mm^4]',
    # 'I_2' : 'I_2 [mm^4]',
    # 'tP'  : 'theta_P [deg]',
    'Xs'  : 'Xs [mm]',
    'Zs'  : 'Zs [mm]',
}



# ================================================================================
# Data persistence - All geometrical sample data
# ================================================================================

@dataclass
class AcumulatedGeomData:
    '''
    Sample data for geometrical properties validation

    Property    Size        Description                                     Unit
    --------    --------    ----------------------------------------    --------
    A           (1,)        Total cross-sectional area                  mm^2
    Xc          (1,)        Centroid X in global frame                  mm
    Zc          (1,)        Centroid Z in global frame                  mm
    I_XX        (1,)        Area moment of inertia about X              mm^4
    I_ZZ        (1,)        Area moment of inertia about Z              mm^4
    I_XZ        (1,)        Product of inertia                          mm^4
    I_1         (1,)        Principal inertia 1                         mm^4
    I_2         (1,)        Principal inertia 2                         mm^4
    theta_P     (1,)        Principal axis angle                        rad
    A_cells     (3,)        Enclosed areas of cells I, II, III          mm^2
    J           (1,)        Torsional constant                          mm^4
    Xs          (1,)        Shear centre X in global frame              mm
    Zs          (1,)        Shear centre Z in global frame              mm
    '''
    A    : np.array = field(default_factory=lambda: _DFLT_FACTORY),

    Xc   : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    Zc   : np.array = field(default_factory=lambda: _DFLT_FACTORY),

    I_XX : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    I_ZZ : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    I_XZ : np.array = field(default_factory=lambda: _DFLT_FACTORY),

    J    : np.array = field(default_factory=lambda: _DFLT_FACTORY),

    I_1  : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    I_2  : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    tP   : np.array = field(default_factory=lambda: _DFLT_FACTORY),

    Xs   : np.array = field(default_factory=lambda: _DFLT_FACTORY),
    Zs   : np.array = field(default_factory=lambda: _DFLT_FACTORY),


# ================================================================================
# Internal Helpers and auxiliary calculations
# ================================================================================

def _norm(arr: list):
    data = np.array(arr)
    xmax = np.max(data)
    xmin = np.min(data)

    xnorm = []
    for x in data:
        xnorm.append(2 * ( (x - xmin) / (xmax - xmin) ) - 1)
    return xnorm


# ================================================================================
# PUBLIC API - Validation
# ================================================================================

def run_reference_section(
    afl_filename: str,
    xw1: float = _XW1,
    xw2: float = _XW2,
    recalculate_props: bool = True,
    use_boom_centroid: bool = True,
    enable_logging: bool = True
) -> GeomData:
    '''
    Run the geometric properties pipeline on the reference cross-section.

    Args:
        afl_path: Path to a *_AirfoilData.json file.

    Returns:
        GeomData with all cross-section properties populated.
    '''
    path = Path(_DFLT_AFL_PATH / f"{afl_filename}")
    with open(path) as fh:
        raw = json.load(fh)
    afl = {k: np.array(v) for k, v in raw.items()}

    afl_pts = (afl['x_upper'], afl['y_upper'],
               afl['x_lower'], afl['y_lower'])
    T1_mat  = (_T_SEG,
               np.full(7, _E1), np.full(7, _E2), np.full(7, _G),
               np.full(7, _E1B), np.full(7, _E2B))
    T4_mat  = (_T_FLANGE,
               np.full(4, _E1), np.full(4, _E2), np.full(4, _G),
               _BF,
               np.full(4, _E1B), np.full(4, _E2B))

    calc = GeomPropCalculator(
        afl_pts           = afl_pts,
        chord             = _CHORD,
        twist             = _TWIST,
        Y_sta             = _Y_STA,
        xw1               = xw1,
        xw2               = xw2,
        T1_props            = T1_mat,
        T4_props            = T4_mat,
        LE_xz             = np.array([0.0,0.0]),
        recalculate_props = recalculate_props,
        use_boom_centroid = use_boom_centroid,
        enable_logging    = enable_logging,
    )
    return calc.run()


def print_summary(gd: GeomData) -> None:
    '''
    Print a formatted property table for the given GeomData.

    Args:
        gd: Populated GeomData instance.
    '''
    c  = gd.chord

    print("=" * 60)
    print("  Cross-Section Property Summary")
    print("=" * 60)
    print(f"  Chord            : {c:.1f} mm")
    print(f"  xw1 / xw2        : {gd.xw1*100:.1f}% / {gd.xw2*100:.1f}% chord")
    print()
    print(f"  Centroid  X      : {gd.C[0]:.4f} mm  ({gd.C[0]/c*100:.1f}% chord)")
    print(f"  Centroid  Z      : {gd.C[2]:.4f} mm")
    print()
    print(f"  A                : {gd.A:.2f} mm^2")
    print()
    print(f"  I_XX             : {gd.I_XX:.0f} mm^4")
    print(f"  I_ZZ             : {gd.I_ZZ:.0f} mm^4")
    print(f"  I_XZ             : {gd.I_XZ:.0f} mm^4")
    print(f"  I_1 (principal)  : {gd.I_1:.0f} mm^4")
    print(f"  I_2 (principal)  : {gd.I_2:.0f} mm^4")
    print(f"  theta_P          : {np.degrees(gd.theta_P):.3f} deg")
    print()
    print(f"  J (torsional)    : {gd.J:.0f} mm^4")
    print(f"  G_REF            : {gd.G_REF:.1f} MPa")
    print()
    print(f"  Shear Center X   : {gd.S_XYZ[0]:.4f} mm  ({gd.S_XYZ[0]/c*100:.1f}% chord)")
    print(f"  Shear Center Z   : {gd.S_XYZ[2]:.4f} mm")
    print(f"  SC centroidal u  : {gd.S_uvw[0]:.4f} mm")
    print(f"  SC centroidal w  : {gd.S_uvw[2]:.4f} mm")
    print()
    print(f"  A_cells I/II/III : {gd.A_cells[0]:.1f} / "
          f"{gd.A_cells[1]:.1f} / {gd.A_cells[2]:.1f} mm^2")
    print()
    print("  Boom positions (global XZ) and areas:")
    labels = gd.boom_lbls
    for i, lbl in enumerate(labels):
        bx = gd.C[0] + gd.boom_u[i]
        bz = gd.C[2] + gd.boom_w[i]
        print(f" : X={bx:7.2f}  Z={bz:7.2f}  A={gd.boom_A[i]:.3f} mm^2")
    print()
    print("  T2 sub-panel properties (s_k, delta_k):")
    for i in range(int(gd.s_k.size)):
        print(f"    panel{i+1:>2d}: s={gd.s_k[i]:.2f} mm  "
              f"delta={gd.delta_k[i]:.4f}")


def plot_section(gd: GeomData) -> None:
    '''
    Plot the cross-section: airfoil outline (T1 skin segments), spar webs,
    boom positions, centroid, and shear center.

    Args:
        gd: Populated GeomData instance.
    '''
    T1 = {s['label']: s for s in gd.T1}
    Xc, Zc = float(gd.C[0]), float(gd.C[2])

    # Skin outline (seg1..seg5) and spar webs (seg6, seg7)
    skin_segs  = ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
    spar_segs  = ['seg6', 'seg7']

    # Boom positions in global XZ
    bx = gd.boom_Xc + gd.boom_u
    bz = gd.boom_Zc + gd.boom_w

    # Flange and stringer positions (from extracted T4 vector arrays)
    flsx = gd.T4_xz[:, 0]
    flsz = gd.T4_xz[:, 1]

    fig, ax = plt.subplots(figsize=(13, 3))

    # -------- Skin contour --------
    for label in skin_segs:
        pts = T1[label]['pts']
        ax.plot(pts[:, 0], pts[:, 1],
                color='#2a2b2c', linewidth=1.2, zorder=2)

    # -------- Spar webs --------
    spar_colors = ['#2a2b2c', '#2a2b2c']
    for label, col in zip(spar_segs, spar_colors):
        pts = T1[label]['pts']
        ax.plot(pts[:, 0], pts[:, 1],
                color=col, linewidth=2.0, zorder=3)

    # -------- Booms --------
    ax.scatter(bx, bz,s=50, color='#1b6511', zorder=5, label='Booms')

    # -------- Flanges and stringers --------
    ax.scatter(flsx, flsz,s=20, color='#b7213e', marker='*', zorder=5, label='Flanges e reforçadores')

    # -------- Centroid --------
    ax.scatter(Xc, Zc,
               s=80, color='#D2B026', zorder=6, marker='D', label='Centroide')

    # -------- Shear center --------
    ax.scatter(gd.S_XYZ[0], gd.S_XYZ[2],
               s=80, color='#17B18A', zorder=6, marker='*',
               label=f'Centro de cisalhamento')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Z [mm]')
    # ax.set_title('Cross-Section Validation')
    ax.set_aspect('equal')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.92,
              edgecolor='#cccccc')

    plt.tight_layout()
    fig.savefig(f"{_DFLT_OUT_PATH / _AFL.removesuffix("_AirfoilData.json")}.pdf", dpi=300, bbox_inches="tight")
    if _SHOW_PLOTS:
        plt.show()


def geom_sample_analysis(
    filepath: str | Path,
    load_from_db: bool,
    enable_logging: bool = True,
) -> dict:
    # Acumulator
    geom_data_acum = {
        'A'    : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'Xc'   : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'Zc'   : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'I_XX' : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'I_ZZ' : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'I_XZ' : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'J'    : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'I_1'  : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'I_2'  : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'tP'   : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'Xs'   : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
        'Zs'   : np.zeros([_N_TEST_SAMPLE, _N_TEST_SAMPLE]),
    }

    logger = io.setup_logger(
        obj=__name__,
        enable_logging=enable_logging
    )

    if load_from_db:
        geom_data_acum_cls = io.read_json(
            filepath=Path(filepath),
            dcls=AcumulatedGeomData,
        )
        geom_data_acum = geom_data_acum_cls.__dict__
    else:
        for i, xw1 in enumerate(_XW1):
            for j, xw2 in enumerate(_XW2):
                logger.info(
                    f"Running XW1 = {xw1 * _CHORD:.0f} mm \t"
                    f"|\t XW2 = {xw2 * _CHORD:.0f} mm"
                )
                gd = run_reference_section(
                    afl_filename=_AFL,
                    xw1=xw1,
                    xw2=xw2,
                    recalculate_props=_RECALC_PROPS,
                    use_boom_centroid=_USE_BOOM_CENTROID,
                    enable_logging=False,
                )

                geom_data_acum['A'][i,j]    = gd.A
                geom_data_acum['Xc'][i,j]   = gd.C[0]
                geom_data_acum['Zc'][i,j]   = gd.C[2]
                geom_data_acum['I_XX'][i,j] = gd.I_XX
                geom_data_acum['I_ZZ'][i,j] = gd.I_ZZ
                geom_data_acum['I_XZ'][i,j] = gd.I_XZ
                geom_data_acum['J'][i,j]    = gd.J
                geom_data_acum['I_1'][i,j]  = gd.I_1
                geom_data_acum['I_2'][i,j]  = gd.I_2
                geom_data_acum['tP'][i,j]   = gd.theta_P
                geom_data_acum['Xs'][i,j]   = gd.S_XYZ[0]
                geom_data_acum['Zs'][i,j]   = gd.S_XYZ[2]

        io.write_json(
            obj=geom_data_acum,
            filepath=_FILEPATH,
        )
        logger.info(f"Data saved to: {_FILEPATH}")
    
    return geom_data_acum


def plot_geom_sweep(
    acum: dict,
    xw1_arr: np.ndarray,
    xw2_arr: np.ndarray,
) -> None:
    '''
    Plot accumulated geometric properties as heatmaps over the
    (xw1, xw2) parameter grid.

    Args:
        acum: Dict of (N, N) arrays keyed by property name.
        xw1_arr: Front spar positions (fraction of chord), shape (N,).
        xw2_arr: Rear spar positions (fraction of chord), shape (N,).
    '''
    keys   = list(_PROP_LABELS.keys())
    n_cols = 3
    n_rows = int(np.ceil(len(keys) / n_cols))

    x1_pct = xw1_arr * 100.0
    x2_pct = xw2_arr * 100.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3),
    )
    axes = axes.ravel()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        Z  = _norm(acum[key])
        if key == 'tP':
            Z = np.degrees(Z)

        pcm = ax.pcolormesh(
            x2_pct, x1_pct, Z,
            cmap='turbo', shading='auto',
        )
        plt.colorbar(pcm, ax=ax, pad=0.02)
        ax.set_title(f'{key}', fontsize=9)
        ax.set_xlabel(r'xw2 [% c]', fontsize=8)
        ax.set_ylabel(r'xw1 [% c]', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if _SHOW_PLOTS:
        plt.show()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    # geom_data_acum = geom_sample_analysis(
    #     filepath=_FILEPATH,
    #     load_from_db=_LOAD_FROM_DB,
    #     enable_logging=_ENBLE_LOG,
    # )
    # plot_geom_sweep(geom_data_acum, _XW1, _XW2)
    gd = run_reference_section(
        afl_filename=_AFL,
        recalculate_props=_RECALC_PROPS,
        use_boom_centroid=_USE_BOOM_CENTROID,
        enable_logging=True,
    )
    print_summary(gd)
    plot_section(gd)
