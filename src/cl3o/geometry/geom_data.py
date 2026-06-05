'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Cross-Section Geometry Data Container Module.

GeomData: the per-station container holding the geometric and mechanical
properties of a wing cross-section (topology dicts, centroid, inertia, shear
centre, torsion constants and stress-recovery constants). Split out of
geom_properties.py so the schema imports without pulling in the calculator;
geom_properties re-exports GeomData for backward compatibility (including pickle
resolution of archived section snapshots).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from dataclasses import dataclass, field

import numpy as np

# ================ Module imports ================

# Constants
from cl3o.Constants import N_BOOMS, BOOM_LBLS


@dataclass
class GeomData:
    '''
    Container for storing geometric and mechanical parameters
    of a cross section at a single spanwise station.

    All coordinates are in the global XZ frame unless noted as uvw
    (centroidal). Units follow CL3O convention: mm, mm^2, mm^4,
    1/mm for fluxes.

    Property        Size        Description                             Units
    --------        --------    ------------------------------------    --------
    chord           (1,)        Chord length                            mm
    xw1             (1,)        Front spar chord fraction               -
    xw2             (1,)        Rear spar chord fraction                -
    t_k             (10,)       Panel thickness per T2 sub-panel        mm
    s_k             (10,)       Arc length per T2 sub-panel             mm
    A_k             (10,)       Area per T2 sub-panel                   mm^2
    G_k             (10,)       Shear modulus per T2 sub-panel          MPa
    delta_k         (10,)       Flexibility parameter per panel         1/mm
    C               (3,)        Centroid vector [Xc, Y_sta, Zc]         mm
    A               (1,)        Total cross-sectional area              mm^2
    I_XX            (1,)        Moment of inertia about X               mm^4
    I_ZZ            (1,)        Moment of inertia about Z               mm^4
    I_XZ            (1,)        Product of inertia                      mm^4
    I_1             (1,)        Principal inertia 1                     mm^4
    I_2             (1,)        Principal inertia 2                     mm^4
    theta_P         (1,)        Principal axis angle                    rad
    c_rad           (1,)        Beam-frame rotation angle (corrected)   rad
    J               (1,)        Torsional constant                      mm^4
    A_cells         (3,)        Enclosed areas of cells I-III           mm^2
    delta_mat       (3, 3)      Cell flexibility matrix                 1/mm
    Delta_mat       (4, 4)      Combined delta matrix                   -
    boom_lbls       (7,)        Boom labels [B1..B7]                    -
    boom_Xc         (1,)        Boom centroid X                         mm
    boom_Zc         (1,)        Boom centroid Z                         mm
    boom_u          (7,)        Boom u-coords (centroidal)              mm
    boom_w          (7,)        Boom w-coords (centroidal)              mm
    boom_y          (7,)        Boom y-coords (beam-local, c_rad rot)   mm
    boom_z          (7,)        Boom z-coords (beam-local, c_rad rot)   mm
    boom_A          (7,)        Boom areas                              mm^2
    stress_ratios   dict        Per-boom Megson stress ratio lists      -
    IXstar          (N_BOOMS,)  Per-boom coefficient of MX in sigma     mm^-4
    IZstar          (N_BOOMS,)  Per-boom coefficient of MZ in sigma     mm^-4
    IXstar_loc      (N_BOOMS,)  Per-boom coeff of local My in sigma     mm^-4
    IZstar_loc      (N_BOOMS,)  Per-boom coeff of local Mz in sigma     mm^-4
    S_XYZ           (3,)        Shear centre [Xs, Y_sta, Zs]            mm
    S_uvw           (3,)        Shear centre [us, 0, ws] (centroid)     mm
    qsX_star        (10,)       Shear flux per unit S_X                 1/mm
    qsZ_star        (10,)       Shear flux per unit S_Z                 1/mm
    qT_star         (10,)       Shear flux per unit torque T            1/mm
    qbX_star        (10,)       Open shear flux per unit S_X            1/mm
    qbZ_star        (10,)       Open shear flux per unit S_Z            1/mm
    qs0X_star       (3,)        Cell constants per unit S_X             1/mm
    qs0Z_star       (3,)        Cell constants per unit S_Z             1/mm
    qs0T_star       (3,)        Cell constants per unit torque T        1/mm
    xi0             (1,)        Moment arm LE to SC (u dir)             mm
    eta0            (1,)        Moment arm LE to SC (w dir)             mm
    G_REF           (1,)        Reference shear modulus                 MPa
    E1_eq           (1,)        Equivalent membrane modulus E1          MPa
    E2_eq           (1,)        Equivalent membrane modulus E2          MPa
    G_eq            (1,)        Equivalent shear modulus                MPa
    E1_bend_eq      (1,)        Equivalent flexural modulus E1          MPa
    E2_bend_eq      (1,)        Equivalent flexural modulus E2          MPa
    A_flange        (4,)        Flange areas [F1..F4]                   mm^2
    A_stringer      (2,)        Stringer areas [S1..S2]                 mm^2
    T1              list        T1 segment dicts (7 entries)            -
    T2              list        T2 sub-panel dicts (10 entries)         -
    T3              list        T3 cell polygon dicts (3 entries)       -
    T4              list        T4 flange/stringer dicts (6 entries)    -
    T4_xz           (6,2)       Flange/stringer positions               mm
    T4_A            (6,)        Flange/stringer areas                   mm^2
    T2_boom_idx     (10,2)      Boom pair indices per T2 panel          -
    P_XZ            (4,2)       Key-point positions (global XZ)         mm
    P_uvw           (4,2)       Key-point positions (centroidal)        mm
    skew_matrix     (3,3)       Antissimetrical matrix (C -> SC)        mm
    skew_matrix_ac  (3,3)       Antissimetrical matrix (AC -> SC)       mm
    cache_key       tuple       Content key in geom_cache (or None)     -
    '''
    chord   : float = 0.0
    xw1     : float = 0.0
    xw2     : float = 0.0

    t_k     : np.ndarray = field(default_factory=lambda: np.zeros(10))
    s_k     : np.ndarray = field(default_factory=lambda: np.zeros(10))
    A_k     : np.ndarray = field(default_factory=lambda: np.zeros(10))
    G_k     : np.ndarray = field(default_factory=lambda: np.zeros(10))
    delta_k : np.ndarray = field(default_factory=lambda: np.zeros(10))

    C  : np.ndarray = field(default_factory=lambda: np.zeros(3))

    A  : float = 0.0

    I_XX    : float = 0.0
    I_ZZ    : float = 0.0
    I_XZ    : float = 0.0
    I_1     : float = 0.0
    I_2     : float = 0.0
    theta_P : float = 0.0
    c_rad   : float = 0.0
    J       : float = 0.0

    A_cells : np.ndarray = field(default_factory=lambda: np.zeros(3))

    delta_mat : np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
    Delta_mat : np.ndarray = field(default_factory=lambda: np.zeros((4,4)))

    boom_lbls : tuple  = BOOM_LBLS
    boom_Xc   : float = 0.0
    boom_Zc   : float = 0.0
    boom_u    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_w    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_y    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_z    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_A    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IXstar    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IZstar    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IXstar_loc : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IZstar_loc : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    stress_ratios  : dict[str, list] = field(
        default_factory=lambda: {f'B{i}': [] for i in range(1, N_BOOMS+1)}
    )

    S_XYZ   : np.ndarray = field(default_factory=lambda: np.zeros(3))
    S_uvw   : np.ndarray = field(default_factory=lambda: np.zeros(3))

    qsX_star : np.ndarray = field(default_factory=lambda: np.zeros(10))
    qsZ_star : np.ndarray = field(default_factory=lambda: np.zeros(10))
    qT_star  : np.ndarray = field(default_factory=lambda: np.zeros(10))
    qbX_star  : np.ndarray = field(default_factory=lambda: np.zeros(10))
    qbZ_star  : np.ndarray = field(default_factory=lambda: np.zeros(10))
    qs0X_star : np.ndarray = field(default_factory=lambda: np.zeros(3))
    qs0Z_star : np.ndarray = field(default_factory=lambda: np.zeros(3))
    qs0T_star : np.ndarray = field(default_factory=lambda: np.zeros(3))

    xi0     : float = 0.0
    eta0    : float = 0.0

    G_REF      : float = 0.0
    E1_eq      : float = 0.0
    E2_eq      : float = 0.0
    G_eq       : float = 0.0
    E1_bend_eq : float = 0.0
    E2_bend_eq : float = 0.0

    A_flange   : np.ndarray = field(default_factory=lambda: np.zeros(4))
    A_stringer : np.ndarray = field(default_factory=lambda: np.zeros(2))

    T1 : list = field(default_factory=list)
    T2 : list = field(default_factory=list)
    T3 : list = field(default_factory=list)
    T4 : list = field(default_factory=list)

    # Extracted vector arrays for T4 and T2 boom topology
    T4_xz       : np.ndarray = field(
        default_factory=lambda: np.zeros((6,2))
    )
    T4_A        : np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    T2_boom_idx : np.ndarray = field(
        default_factory=lambda: np.zeros((10,2), dtype=int)
    )

    P_XZ  : np.ndarray = field(default_factory=lambda: np.zeros((4,2)))
    P_uvw : np.ndarray = field(default_factory=lambda: np.zeros((4,2)))

    skew_matrix    : np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
    skew_matrix_ac : np.ndarray = field(default_factory=lambda: np.zeros((3,3)))

    # Content key under which this object is stored in StaticData.geom_cache.
    # Set by SectionBuilder; lets the beam cache key on geometry content rather
    # than id(), so both caches can evict independently without false hits.
    cache_key : tuple | None = None
