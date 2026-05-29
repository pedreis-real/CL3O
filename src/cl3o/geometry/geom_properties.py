'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Geometrical Properties Module.

Computes all geometric and structural properties of a wing cross-section
at a single spanwise station.

Pipeline (executed by GeomPropCalculator.run()):
  1. Blend airfoil profiles from adjacent control points
  2. Apply twist rotation and chord scaling
  3. Insert spar / key points and build the 7-boom canonical set:
        B1........upper-rear-spar
        B2........upper-mid
        B3........upper-front-spar
        B4........LE
        B5........lower-front-spar
        B6........lower-mid
        B7........lower-rear-spar
    and segment into T1 (7 panels, straight spar webs) and
    T2 (10 sub-panels).  TE is a geometric endpoint
    only (stored as P4) and does not carry a boom index.
  4. Compute T2 segment lengths, areas, delta parameters, G_REF
  5. Pre-compute per-panel first and second moments (for vectorized
     centroid and inertia)
  6. Compute centroid, moments of inertia, principal axes
  7. Compute cell enclosed areas, delta matrix, torsional constant J
  8. Compute boom areas (structural idealization, 8-boom T2' set)
  9. Compute shear center (open + closed section shear flows)
  10. Compute adimensional shear flux per T2 sub-panel

Topology coordinate convention:
    T1 / T2 / T3 dicts store coordinates as a single (N, 2) array
    under key 'pts', where pts[:, 0] = x and pts[:, 1] = z.
    T4 dicts store a single point as a (2,) array under 'pts'.
    Global arrays T4_xz (6, 2) and T4_A (6,) are extracted from T4
    for use in vectorized centroid / inertia operations.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================


# ================ Module imports ================

# Constants
from cl3o.Constants import N_BOOMS, BOOM_LBLS, TOL

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils import math_utils as mthu

# Geometry
from cl3o.geometry.structural_idealization import StructuralIdealization


# ================================================================================
# Data persistence - Geometrical properties for single station
# ================================================================================

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
    J               (1,)        Torsional constant                      mm^4
    A_cells         (3,)        Enclosed areas of cells I-III           mm^2
    delta_mat       (3, 3)      Cell flexibility matrix                 1/mm
    Delta_mat       (4, 4)      Combined delta matrix                   -
    boom_lbls       (7,)        Boom labels [B1..B7]                    -
    boom_Xc         (1,)        Boom centroid X                         mm
    boom_Zc         (1,)        Boom centroid Z                         mm
    boom_u          (7,)        Boom u-coords (centroidal)              mm
    boom_w          (7,)        Boom w-coords (centroidal)              mm
    boom_A          (7,)        Boom areas                              mm^2
    stress_ratios   dict        Per-boom Megson stress ratio lists      -
    IXstar          (N_BOOMS,)  Per-boom coefficient of MX in sigma     mm^-4
    IZstar          (N_BOOMS,)  Per-boom coefficient of MZ in sigma     mm^-4
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
    J       : float = 0.0

    A_cells : np.ndarray = field(default_factory=lambda: np.zeros(3))

    delta_mat : np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
    Delta_mat : np.ndarray = field(default_factory=lambda: np.zeros((4,4)))

    boom_lbls : tuple  = BOOM_LBLS
    boom_Xc   : float = 0.0
    boom_Zc   : float = 0.0
    boom_u    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_w    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_A    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IXstar    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    IZstar    : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
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


# ================================================================================
# PUBLIC API - Geometrical properties calculator
# ================================================================================

class GeomPropCalculator:
    '''
    Computes all geometric and structural properties of a wing
    cross-section at a single spanwise station.
    '''
    def __init__(
        self,
        afl_pts : tuple[np.ndarray, ...],
        chord : float,
        twist : float,
        Y_sta : float,
        xw1 : float,
        xw2 : float,
        T1_props : tuple,
        T4_props : tuple,
        LE_xz : np.ndarray,
        recalculate_props : bool = True,
        use_boom_centroid : bool = False,
        enable_logging    : bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)
        self.recalculate_props = recalculate_props
        self.use_boom_centroid = use_boom_centroid
        self.LE_xz = (
            np.zeros(2, dtype=float) if LE_xz is None
            else np.asarray(LE_xz, dtype=float).reshape(2).copy()
        )

        # Store inputs
        _a = np.asarray
        self.x_u_adim = _a(afl_pts[0])
        self.y_u_adim = _a(afl_pts[1])
        self.x_l_adim = _a(afl_pts[2])
        self.y_l_adim = _a(afl_pts[3])

        self.chord = float(chord)
        self.twist = float(twist)
        self.Y_sta = float(Y_sta)

        self.xw1_frac = float(xw1)
        self.xw2_frac = float(xw2)

        self.t_seg       = _a(T1_props[0])
        self.E1_seg      = _a(T1_props[1])
        self.E2_seg      = _a(T1_props[2])
        self.G_seg       = _a(T1_props[3])
        self.E1_bend_seg = _a(T1_props[4])
        self.E2_bend_seg = _a(T1_props[5])

        self.t_flange       = _a(T4_props[0])
        self.E1_flange      = _a(T4_props[1])
        self.E2_flange      = _a(T4_props[2])
        self.G_flange       = _a(T4_props[3])
        self.bf             = _a(T4_props[4])
        self.E1_bend_flange = _a(T4_props[5])
        self.E2_bend_flange = _a(T4_props[6])

        self.A_flange = self.bf * self.t_flange

        self.A_stringer = np.zeros(2, dtype=float)

    # ----------------------------------------------------------------
    # Private - Section construction
    # ----------------------------------------------------------------

    def _scale_airfoil(self):
        '''
        Apply twist rotation and chord scaling to the adimensional
        airfoil. Returns upper and lower surface arrays in dimensional
        coordinates [mm].
        '''
        c = self.chord

        xu = self.x_u_adim * c
        yu = self.y_u_adim * c
        xl = self.x_l_adim * c
        yl = self.y_l_adim * c

        if abs(self.twist) > TOL:
            xu, yu = mthu.rotate_points(xu, yu, self.twist)
            xl, yl = mthu.rotate_points(xl, yl, self.twist)

        self.xu = xu
        self.yu = yu
        self.xl = xl
        self.yl = yl

    def _find_spar_intersections(self):
        '''
        Find where the front and rear spars intersect the upper/lower
        surfaces and assemble the 7-boom canonical set:

            B1 = upper rear-spar cap     (xw2, zu_w2)
            B2 = upper mid-skin point    (x_mid, zu_mid)
            B3 = upper front-spar cap    (xw1, zu_w1)
            B4 = leading edge (LE)
            B5 = lower front-spar cap    (xw1, zl_w1)
            B6 = lower mid-skin point    (x_mid, zl_mid)
            B7 = lower rear-spar cap     (xw2, zl_w2)

        Auxiliary key points P1=LE, P4=TE, P5=B3, P6=B7 are kept for
        downstream plotting / diagnostics.
        '''
        xw1 = self.xw1_frac * self.chord
        xw2 = self.xw2_frac * self.chord

        zu_w1 = float(np.interp(xw1, self.xu, self.yu))
        zu_w2 = float(np.interp(xw2, self.xu, self.yu))
        zl_w1 = float(np.interp(xw1, self.xl, self.yl))
        zl_w2 = float(np.interp(xw2, self.xl, self.yl))

        x_mid  = 0.5 * (xw1 + xw2)
        zu_mid = float(np.interp(x_mid, self.xu, self.yu))
        zl_mid = float(np.interp(x_mid, self.xl, self.yl))

        TE = np.array([self.xu[-1], self.yu[-1]])
        LE = np.array([self.xu[ 0], self.yu[ 0]])
        self.boom_pos = np.array([
            [xw2,   zu_w2 ],    # B1: upper rear spar
            [x_mid, zu_mid],    # B2: upper mid skin
            [xw1,   zu_w1 ],    # B3: upper front spar
            LE,                  # B4: LE
            [xw1,   zl_w1 ],    # B5: lower front spar
            [x_mid, zl_mid],    # B6: lower mid skin
            [xw2,   zl_w2 ],    # B7: lower rear spar
        ], dtype=float)

        self.P1 = LE.copy()
        self.P4 = TE.copy()
        self.P5 = self.boom_pos[2].copy()    # = B3 (upper front spar)
        self.P6 = self.boom_pos[6].copy()    # = B7 (lower rear spar)

        self.xw1_dim = xw1
        self.xw2_dim = xw2

    def _segment_T1(self):
        '''
        Build the 7 T1 segments. Each segment dict contains:
            label : str
            pts   : (N, 2) float array, columns [x, z]
            t     : float, panel thickness [mm]

        T1 topology (boom indices in self.boom_pos, 0=B1..6=B7):
          seg1: B5 -> LE -> B3    (nose skin, wraps around LE)
          seg2: B3 -> B2 -> B1    (upper middle skin)
          seg3: B1 -> TE          (upper rear skin)
          seg4: TE -> B7          (lower rear skin)
          seg5: B7 -> B6 -> B5    (lower middle skin)
          seg6: B5 -> B3          (front spar web, straight)
          seg7: B7 -> B1          (rear spar web, straight)
        '''
        xu, yu = self.xu, self.yu
        xl, yl = self.xl, self.yl
        xw1 = self.xw1_dim
        xw2 = self.xw2_dim
        bp  = self.boom_pos

        (xu_le_w1, yu_le_w1), (xu_w1_te, yu_w1_te) = \
            mthu.split_curve_at_x(xu, yu, xw1)
        (xu_w1_w2, yu_w1_w2), (xu_w2_te, yu_w2_te) = \
            mthu.split_curve_at_x(xu_w1_te, yu_w1_te, xw2)

        (xl_le_w1, yl_le_w1), (xl_w1_te, yl_w1_te) = \
            mthu.split_curve_at_x(xl, yl, xw1)
        (xl_w1_w2, yl_w1_w2), (xl_w2_te, yl_w2_te) = \
            mthu.split_curve_at_x(xl_w1_te, yl_w1_te, xw2)

        # seg1: lower B5 to LE (reversed), then upper LE to B3
        s1_x = np.concatenate([xl_le_w1[::-1], xu_le_w1[1:]])
        s1_z = np.concatenate([yl_le_w1[::-1], yu_le_w1[1:]])

        s2_x = xu_w1_w2.copy();   s2_z = yu_w1_w2.copy()
        s3_x = xu_w2_te.copy();   s3_z = yu_w2_te.copy()
        s4_x = xl_w2_te[::-1];    s4_z = yl_w2_te[::-1]
        s5_x = xl_w1_w2[::-1];    s5_z = yl_w1_w2[::-1]

        s6_x = np.linspace(bp[4, 0], bp[2, 0], 6)  # B5 -> B3
        s6_z = np.linspace(bp[4, 1], bp[2, 1], 6)
        s7_x = np.linspace(bp[6, 0], bp[0, 0], 6)  # B7 -> B1
        s7_z = np.linspace(bp[6, 1], bp[0, 1], 6)

        def _seg(label, x, z, t_idx):
            return {
                'label': label,
                'pts'  : np.column_stack([
                    np.asarray(x, dtype=float),
                    np.asarray(z, dtype=float),
                ]),
                't'    : float(self.t_seg[t_idx]),
            }

        self.T1 = [
            _seg('seg1', s1_x, s1_z, 0),
            _seg('seg2', s2_x, s2_z, 1),
            _seg('seg3', s3_x, s3_z, 2),
            _seg('seg4', s4_x, s4_z, 3),
            _seg('seg5', s5_x, s5_z, 4),
            _seg('seg6', s6_x, s6_z, 5),
            _seg('seg7', s7_x, s7_z, 6),
        ]

    def _segment_T2(self):
        '''
        Build the 10 T2 sub-panels. Each sub-panel
        dict contains:
            label  : str
            pts    : (N, 2) float array, columns [x, z]
            t      : float, panel thickness [mm]
            s      : float, arc length [mm]
            G      : float, shear modulus [MPa]
            E1     : float, in-plane modulus 1 [MPa]
            E2     : float, in-plane modulus 2 [MPa]
            boomA  : int, first boom index (0-based)
            boomB  : int, second boom index (0-based)

        T2 panel topology (7-boom, boom indices 0-based):
            r1  (panel  1): B1 -> TE    upper rear skin       booms (0, 6)
            r2  (panel  2): B2 -> B1    upper mid right       booms (1, 0)
            r3  (panel  3): B3 -> B2    upper mid left        booms (2, 1)
            r4  (panel  4): B4 -> B3    upper LE skin         booms (3, 2)
            r5  (panel  5): B4 -> B5    lower LE skin         booms (3, 4)
            r6  (panel  6): B5 -> B6    lower mid left        booms (4, 5)
            r7  (panel  7): B6 -> B7    lower mid right       booms (5, 6)
            r8  (panel  8): TE -> B7    lower rear skin       booms (0, 6)
            r9  (panel  9): B3 -> B5    front spar web        booms (2, 4)
            r10 (panel 10): B1 -> B7    rear spar web         booms (0, 6)
        '''
        T1    = {s['label']: s for s in self.T1}
        x_mid = 0.5 * (self.xw1_dim + self.xw2_dim)

        def _xy(seg):
            return seg['pts'][:, 0], seg['pts'][:, 1]

        x1, z1 = _xy(T1['seg1'])
        x2, z2 = _xy(T1['seg2'])
        x3, z3 = _xy(T1['seg3'])
        x4, z4 = _xy(T1['seg4'])
        x5, z5 = _xy(T1['seg5'])
        x6, z6 = _xy(T1['seg6'])
        x7, z7 = _xy(T1['seg7'])

        # -------- seg1 split at LE --------
        ba   = int(np.argmin(x1))
        r4_x, r4_z = x1[ba:],    z1[ba:]
        r5_x, r5_z = x1[ba::-1], z1[ba::-1]

        # -------- seg2 split at B2 (upper surface @ x_mid) --------
        (r3_xz, r2_xz) = mthu.split_curve_at_x(x2, z2, x_mid)
        r3_x, r3_z = r3_xz
        r2_x, r2_z = r2_xz

        # -------- seg3: upper rear skin B1 -> TE --------
        r1_x, r1_z = x3.copy(), z3.copy()

        # -------- seg4: lower rear skin TE -> B7 --------
        r8_x, r8_z = x4.copy(), z4.copy()

        # -------- seg5 split at B6 (lower surface @ x_mid) --------
        (r6_xz, r7_xz) = mthu.split_curve_at_x(x5[::-1], z5[::-1], x_mid)
        r6_x, r6_z = r6_xz
        r7_x, r7_z = r7_xz

        # -------- spar webs: straight 2-point segments --------
        r9_x,  r9_z  = x6.copy(), z6.copy()
        r10_x, r10_z = x7.copy(), z7.copy()

        def _r(
            label  : str,
            x      : np.ndarray,
            z      : np.ndarray,
            T1_seg : int,
            n_idx  : int,
            m_idx  : int,
        ) -> dict:
            '''Build a T2 sub-panel dict. n_idx/m_idx are 0-based boom idx.'''
            pts = np.column_stack([
                np.asarray(x, dtype=float),
                np.asarray(z, dtype=float),
            ])
            return {
                'label'  : label,
                'pts'    : pts,
                't'      : float(self.t_seg[T1_seg]),
                's'      : mthu.arc_length(pts[:, 0], pts[:, 1]),
                'G'      : float(self.G_seg[T1_seg]),
                'E1'     : float(self.E1_seg[T1_seg]),
                'E2'     : float(self.E2_seg[T1_seg]),
                'E1_bend': float(self.E1_bend_seg[T1_seg]),
                'E2_bend': float(self.E2_bend_seg[T1_seg]),
                'boomA'  : n_idx,
                'boomB'  : m_idx,
            }

        self.T2 = [
            _r('r1',  r1_x,  r1_z,  2, 0, 6),  # B1->TE,  upper rear skin
            _r('r2',  r2_x,  r2_z,  1, 1, 0),  # B2->B1,  upper mid right
            _r('r3',  r3_x,  r3_z,  1, 2, 1),  # B3->B2,  upper mid left
            _r('r4',  r4_x,  r4_z,  0, 3, 2),  # B4->B3,  upper LE skin
            _r('r5',  r5_x,  r5_z,  0, 3, 4),  # B4->B5,  lower LE skin
            _r('r6',  r6_x,  r6_z,  4, 4, 5),  # B5->B6,  lower mid left
            _r('r7',  r7_x,  r7_z,  4, 5, 6),  # B6->B7,  lower mid right
            _r('r8',  r8_x,  r8_z,  3, 0, 6),  # TE->B7,  lower rear skin
            _r('r9',  r9_x,  r9_z,  5, 2, 4),  # B3->B5,  front spar web
            _r('r10', r10_x, r10_z, 6, 0, 6),  # B1->B7,  rear spar web
        ]

    def _segment_T3(self):
        '''
        Build the 3 T3 closed cell polygons from T1 segments. Each dict
        contains 'label' and 'pts' (N, 2), columns [x, z]. Cells are
        oriented clockwise.

        T3 topology (clockwise traversal):
          Cell I   : seg1(B5->LE->B3) + seg6(B3->B5)
          Cell II  : seg2(B3->B2->B1) + seg7(B1->B7 rev)
                   + seg5(B7->B6->B5) + seg6(rev B5->B3)
          Cell III : seg3(B1->TE) + seg4(TE->B7) + seg7(rev B7->B1)
        '''
        T1 = {s['label']: s for s in self.T1}

        def _cat(*pieces):
            xs, zs = [], []
            for x, z in pieces:
                if not xs:
                    xs.append(x);  zs.append(z)
                else:
                    xs.append(x[1:]); zs.append(z[1:])
            return np.concatenate(xs), np.concatenate(zs)

        def _s(label, rev=False):
            pts = T1[label]['pts']
            if rev:
                return pts[::-1, 0], pts[::-1, 1]
            return pts[:, 0], pts[:, 1]

        xI,   zI   = _cat(_s('seg1'), _s('seg6'))
        xII,  zII  = _cat(
            _s('seg2'), _s('seg7'),
            _s('seg5'), _s('seg6', True),
        )
        xIII, zIII = _cat(_s('seg3'), _s('seg4'), _s('seg7', True))

        def _cell(label, x, z):
            return {
                'label': label,
                'pts'  : np.column_stack([
                    np.asarray(x, dtype=float),
                    np.asarray(z, dtype=float),
                ]),
            }

        self.T3 = [
            _cell('cell_I',   xI,   zI  ),
            _cell('cell_II',  xII,  zII ),
            _cell('cell_III', xIII, zIII),
        ]

    def _segment_T4(self):
        '''
        Build the 6 T4 elements (4 flanges + 2 stringer placeholders).
        Each dict contains 'label', 'A', and 'pts' (2,) = [x, z].

        After building, T4_xz (6, 2) and T4_A (6,) are extracted for
        vectorized centroid and inertia computations.

        Flange / Stringer       Associated boom     T1 seg
        ------------------      ----------------    -------
        F1                      B3 (idx=2)          1 (idx=0)
        F2                      B5 (idx=4)          1 (idx=0)
        F3                      B1 (idx=0)          2 (idx=1)
        F4                      B7 (idx=6)          5 (idx=4)
        S1                      B2 (idx=1)          2 (idx=1)
        S2                      B6 (idx=5)          5 (idx=4)
        '''
        tk = self.t_seg
        tf = self.t_flange
        Af = self.A_flange
        As = self.A_stringer
        bp = self.boom_pos

        self.T4 = [
            {'label': 'F1',
             'A': Af[0],
             'pts': np.array([bp[2,0], bp[2,1]-tk[0]-tf[0]/2])},
            {'label': 'F2',
             'A': Af[1],
             'pts': np.array([bp[4,0], bp[4,1]+tk[0]+tf[1]/2])},
            {'label': 'F3',
             'A': Af[2],
             'pts': np.array([bp[0,0], bp[0,1]-tk[1]-tf[2]/2])},
            {'label': 'F4',
             'A': Af[3],
             'pts': np.array([bp[6,0], bp[6,1]+tk[4]+tf[3]/2])},
            {'label': 'S1',
             'A': As[0],
             'pts': np.array([bp[1,0], bp[1,1]-tk[1]])},
            {'label': 'S2',
             'A': As[1],
             'pts': np.array([bp[5,0], bp[5,1]+tk[4]])},
        ]

        self.T4_xz = np.stack([r['pts'] for r in self.T4])   # (6, 2)
        self.T4_A  = np.array([r['A']   for r in self.T4])   # (6,)


    # ----------------------------------------------------------------
    # Private - Primary geometric properties
    # ----------------------------------------------------------------

    def _compute_segment_properties(self):
        '''
        Compute per-T2-sub-panel scalars: arc length s_k, area A_k,
        thickness t_k, shear modulus G_k, normalized t*_k, delta_k.

        All per-panel scalars are extracted directly from the T2 dicts
        and assembled into (10,) arrays without explicit Python loops
        over individual coordinate segments.
        '''
        t_panel = np.array([r['t'] for r in self.T2])
        G_panel = np.array([r['G'] for r in self.T2])
        s_panel = np.array([r['s'] for r in self.T2])
        A_panel = s_panel * t_panel

        # G_REF: skin/web + flange contributions
        total_GA = (float(np.dot(G_panel, A_panel))
                    + float(np.dot(self.G_flange, self.A_flange)))
        total_A  = (float(np.sum(A_panel))
                    + float(np.sum(self.A_flange)))
        G_REF = total_GA / total_A

        # Normalized thickness t*_k = (G_k / G_REF) * t_k
        t_star      = (G_panel / G_REF) * t_panel
        delta_panel = s_panel / (t_star + TOL)

        self.s_k     = s_panel
        self.A_k     = A_panel
        self.A       = total_A
        self.t_k     = t_panel
        self.G_k     = G_panel
        self.delta_k = delta_panel
        self.G_REF   = G_REF

    def _precompute_panel_moments(self):
        '''
        Pre-compute arc-length weighted first and second spatial moments
        for each T2 sub-panel. Stored as (10,) arrays; consumed by
        _compute_centroid and _compute_inertia via the parallel-axis
        theorem to avoid repeated coordinate loops.

        For panel i with sub-segments j:
            dA_j = t_i * ||dr_j||
            _T2_Sx [i] = sum_j ( x_mid_j * dA_j )
            _T2_Sz [i] = sum_j ( z_mid_j * dA_j )
            _T2_Sx2[i] = sum_j ( x_mid_j^2 * dA_j )
            _T2_Sz2[i] = sum_j ( z_mid_j^2 * dA_j )
            _T2_Sxz[i] = sum_j ( x_mid_j * z_mid_j * dA_j )
        '''
        n   = len(self.T2)
        Sx  = np.zeros(n)
        Sz  = np.zeros(n)
        Sx2 = np.zeros(n)
        Sz2 = np.zeros(n)
        Sxz = np.zeros(n)

        for i, r in enumerate(self.T2):
            pts = r['pts']             # (Nk, 2)
            x   = pts[:, 0]
            z   = pts[:, 1]
            t   = r['t']
            xm  = 0.5 * (x[:-1] + x[1:])
            zm  = 0.5 * (z[:-1] + z[1:])
            ds  = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
            dA  = t * ds
            Sx[i]  = float(np.dot(xm,      dA))
            Sz[i]  = float(np.dot(zm,      dA))
            Sx2[i] = float(np.dot(xm**2,   dA))
            Sz2[i] = float(np.dot(zm**2,   dA))
            Sxz[i] = float(np.dot(xm * zm, dA))

        self._T2_Sx  = Sx
        self._T2_Sz  = Sz
        self._T2_Sx2 = Sx2
        self._T2_Sz2 = Sz2
        self._T2_Sxz = Sxz

    # ----------------------------------------------------------------
    # Private - Delta matrix and cell properties
    # ----------------------------------------------------------------

    def _compute_delta_matrix(self):
        '''
        Build [delta] (3x3) cell matrix from the per-T2-panel deltas.

        Cell membership of each T2 panel (0-indexed):
            Cell I   (nose)  : panels 4, 5, 9
            Cell II  (mid)   : panels 3, 2, 10, 7, 6, 9
            Cell III (rear)  : panels 1, 8, 10
        Shared walls:
            I-II  : panel 9  (front spar)
            II-III: panel 10 (rear spar)
        '''
        d = self.delta_k    # (10,) 0-indexed

        dI      = float(np.sum(d[[3, 4, 8]]))
        dI_II   = float(d[8])
        dII     = float(np.sum(d[[1, 2, 5, 6, 8, 9]]))
        dII_III = float(d[9])
        dIII    = float(np.sum(d[[0, 7, 9]]))

        self.delta_mat = np.array([
            [ dI,     -dI_II,    0.0     ],
            [-dI_II,   dII,     -dII_III ],
            [ 0.0,    -dII_III,  dIII    ],
        ])
        self.d_cells = np.array([dI, dII, dIII, dI_II, dII_III])

    def _compute_cell_areas(self):
        '''
        Compute enclosed area for each T3 cell using the shoelace
        formula.
        '''
        self.A_cells = np.array([
            mthu.polygon_area(c['pts'][:, 0], c['pts'][:, 1])
            for c in self.T3
        ])

    def _compute_Delta_and_J(self):
        '''
        Build [Delta] (4x4) and compute torsional constant J.
        '''
        A_vec = 2.0 * self.A_cells   # [2*AI, 2*AII, 2*AIII]

        Delta = np.zeros((4, 4))
        Delta[:3, 1:] = self.delta_mat
        Delta[:3,  0] = -A_vec
        Delta[3,  1:] =  A_vec
        Delta[3,   0] = 0.0

        inv_D = np.linalg.inv(Delta)
        self.Delta_mat = Delta
        self.inv_D     = inv_D
        self.J         = float(1.0 / inv_D[0, -1])

    # ----------------------------------------------------------------
    # Private - Centroid and inertia
    # ----------------------------------------------------------------

    def _compute_centroid(self):
        '''
        Centroid of the cross section from the precomputed T2 first
        moments plus T4 flange and stringer point contributions.

        Uses the parallel-axis form:
            Xc = (sum_T2(Sx) + sum_T4(x_f * A_f)) / A_total
        '''
        T4_xz = self.T4_xz   # (6, 2)
        T4_A  = self.T4_A    # (6,)

        sum_xA = float(np.sum(self._T2_Sx) + np.dot(T4_xz[:, 0], T4_A))
        sum_zA = float(np.sum(self._T2_Sz) + np.dot(T4_xz[:, 1], T4_A))
        sum_A  = float(np.sum(self.A_k)    + np.sum(T4_A))

        self.Xc = sum_xA / (sum_A + TOL)
        self.Zc = sum_zA / (sum_A + TOL)

    def _compute_inertia(self):
        '''
        Moments of inertia I_XX, I_ZZ, I_XZ about centroid via the
        parallel-axis theorem applied to precomputed panel moment sums.

        For panels:
            I_XX = sum(Sz2) - 2*Zc*sum(Sz) + Zc^2 * sum(A_k)
        For flanges/stringers (point areas):
            I_XX += sum( (z_f - Zc)^2 * A_f )
        '''
        Xc   = self.Xc
        Zc   = self.Zc
        A_k  = self.A_k
        T4_x = self.T4_xz[:, 0] - Xc
        T4_z = self.T4_xz[:, 1] - Zc
        T4_A = self.T4_A

        sum_Ak = float(np.sum(A_k))

        IXX = (float(np.sum(self._T2_Sz2))
               - 2.0 * Zc * float(np.sum(self._T2_Sz))
               + Zc**2 * sum_Ak
               + float(np.dot(T4_z**2, T4_A)))

        IZZ = (float(np.sum(self._T2_Sx2))
               - 2.0 * Xc * float(np.sum(self._T2_Sx))
               + Xc**2 * sum_Ak
               + float(np.dot(T4_x**2, T4_A)))

        IXZ = (float(np.sum(self._T2_Sxz))
               - Xc * float(np.sum(self._T2_Sz))
               - Zc * float(np.sum(self._T2_Sx))
               + Xc * Zc * sum_Ak
               + float(np.dot(T4_x * T4_z, T4_A)))

        self.I_XX = IXX
        self.I_ZZ = IZZ
        self.I_XZ = IXZ

    def _compute_principal_inertia(self):
        '''Principal inertias and angle.'''
        (self.I_1, self.I_2, self.theta_P) = mthu.mohr_circle(
            Ixx=self.I_XX,
            Iyy=self.I_ZZ,
            Ixy=self.I_XZ,
        )

    # ----------------------------------------------------------------
    # Private - Shear center
    # ----------------------------------------------------------------

    def _swept_double_area_T2(self, panel_idx: int) -> float:
        '''
        Twice the area swept by the radius from the LE (B4) to the
        T2 sub-panel polyline. Used as the moment arm for open-section
        shear flux about the LE.
        '''
        pts = self.T2[panel_idx]['pts']
        BA  = self.boom_pos[3]            # B4 = LE (index 3 in 7-boom)
        return mthu.swept_double_area(
            pts[:, 0] - BA[0],
            pts[:, 1] - BA[1],
        )

    def _compute_shear_center(self):
        '''
        Compute shear center coordinates and the open-section flux per
        T2 sub-panel for the 8-boom canonical set. Cuts are taken at
        T2 panels 1 (B2->B1, upper rear), 3 (B4->B3, upper mid left)
        and 4 (B5->B4, upper LE), leaving seven non-zero open flows.
        '''
        IXX, IZZ, IXZ = self.I_XX, self.I_ZZ, self.I_XZ
        D = IXX * IZZ - IXZ**2

        u, w, B = self.boom_u, self.boom_w, self.boom_A

        # Per-boom dq* coefficients
        dqX = (-IXX * u + IXZ * w) / D * B
        dqZ = ( IXZ * u - IZZ * w) / D * B

        # Open-section flux accumulation (cuts at r1, r3, r4)
        def _open(dq):
            qb_21 = dq[1]                   # dq_B2
            qb_17 = qb_21 + dq[0]           # dq_B2 + dq_B1
            qb_35 = dq[2]                   # dq_B3
            qb_45 = dq[3]                   # dq_B4
            qb_56 = qb_35 + qb_45 + dq[4]   # dq_B3 + dq_B4 + dq_B5
            qb_67 = qb_56 + dq[5]           # dq_B3 + dq_B4 + dq_B5 + dq_B6
            return np.array([qb_21, qb_17, qb_35, qb_45, qb_56, qb_67])

        qbX = _open(dqX)
        qbZ = _open(dqZ)

        self.logger.debug(f"\n[CLEO] Residual: qb,67 + qb,17 + dq_B7 = \n"
                          f"|  {qbX[5] + qbX[1] + dqX[6]:.2e} in X\n"
                          f"|  {qbZ[5] + qbZ[1] + dqZ[6]:.2e} in Z\n"
                          f".. For reference, qb,35 = ({qbX[2]:.2e}, {qbZ[2]:.2e})")
        
        (qbX_21, qbX_17, qbX_35, qbX_45, qbX_56, qbX_67) = qbX
        (qbZ_21, qbZ_17, qbZ_35, qbZ_45, qbZ_56, qbZ_67) = qbZ

        d  = self.delta_k
        d2 = d[1];  d5 = d[4];  d6 = d[5]
        d7 = d[6];  d8 = d[8];  d9 = d[9]

        OX = np.array([
            qbX_45 * d5  - qbX_35 * d8,
            qbX_35 * d8 + qbX_56 * d6 + qbX_67 * d7
                - qbX_17 * d9 - qbX_21 * d2,
            qbX_17 * d9,
        ])
        OZ = np.array([
            qbZ_45 * d5  - qbZ_35 * d8,
            qbZ_35 * d8 + qbZ_56 * d6 + qbZ_67 * d7
                - qbZ_17 * d9 - qbZ_21 * d2,
            qbZ_17 * d9,
        ])

        A_21 = self._swept_double_area_T2(1)
        A_45 = self._swept_double_area_T2(4)
        A_56 = self._swept_double_area_T2(5)
        A_67 = self._swept_double_area_T2(6)
        A_35 = self._swept_double_area_T2(8)
        A_17 = self._swept_double_area_T2(9)

        def _moment(qb):
            qb_21, qb_17, qb_35, qb_45, qb_56, qb_67 = qb
            return (  qb_45 * A_45
                    - qb_35 * A_35
                    + qb_56 * A_56
                    + qb_67 * A_67
                    - qb_17 * A_17
                    - qb_21 * A_21)

        MqbX = _moment(qbX)
        MqbZ = _moment(qbZ)

        delta_inv = np.linalg.inv(self.delta_mat)
        A_vec     = 2.0 * self.A_cells

        A_dinv_OX = float(A_vec @ delta_inv @ OX)
        A_dinv_OZ = float(A_vec @ delta_inv @ OZ)

        xi_s  = -MqbZ + A_dinv_OZ
        eta_s =  MqbX - A_dinv_OX

        Xc, Zc = self.Xc, self.Zc
        BA = self.boom_pos[3]
        self.Xs = float(BA[0] - xi_s)
        self.Zs = float(BA[1] - eta_s)
        self.us = self.Xs - Xc
        self.ws = self.Zs - Zc

        u_BA = float(BA[0]) - Xc
        w_BA = float(BA[1]) - Zc
        self.xi0  = u_BA - self.us
        self.eta0 = w_BA - self.ws

        self._qbX  = qbX
        self._qbZ  = qbZ
        self._OX   = OX
        self._OZ   = OZ
        self._MqbX = MqbX
        self._MqbZ = MqbZ

    def _compute_shear_flux(self):
        '''
        Compute closed-section shear flux per T2 sub-panel (size 10)
        for unit S_X, S_Z and torque T.
        '''
        Delta_inv = self.inv_D

        rhs_T         = np.array([0.0, 0.0, 0.0, 1.0])
        qs0_T         = (Delta_inv @ rhs_T)[1:4]

        rhs_X         = np.zeros(4)
        rhs_X[:3]     = -self._OX
        rhs_X[3]      = self.eta0 - self._MqbX
        qs0_X         = (Delta_inv @ rhs_X)[1:4]

        rhs_Z         = np.zeros(4)
        rhs_Z[:3]     = -self._OZ
        rhs_Z[3]      = -self.xi0 - self._MqbZ
        qs0_Z         = (Delta_inv @ rhs_Z)[1:4]

        def _assemble(qs0: np.ndarray, qb: np.ndarray) -> np.ndarray:
            '''
            Assemble total shear flux for all 10 T2 panels from the
            cell constants qs0 (size 3) and open-section flux qb (6).
            '''
            qb_21, qb_17, qb_35, qb_45, qb_56, qb_67 = qb
            q = np.empty(10)
            q[0] = qs0[2]
            q[1] = qs0[1] - qb_21
            q[2] = qs0[1]
            q[3] = qs0[0]
            q[4] = qs0[0] + qb_45
            q[5] = qs0[1] + qb_56
            q[6] = qs0[1] + qb_67
            q[7] = qs0[2]
            q[8] = -qs0[0] + qs0[1] + qb_35
            q[9] = qs0[1] - qs0[2] - qb_17
            return q

        self.qsX_star  = _assemble(qs0_X, self._qbX)
        self.qsZ_star  = _assemble(qs0_Z, self._qbZ)
        self.qT_star   = _assemble(qs0_T, np.zeros(6))
        self.qs0X_star = qs0_X
        self.qs0Z_star = qs0_Z
        self.qs0T_star = qs0_T

        def _expand_qb(qb: np.ndarray) -> np.ndarray:
            '''
            Expand 6-element open-flux vector to all 10 T2 panels.
            Cuts at r1, r3, r4 and unloaded r8 remain zero.
            '''
            qb_21, qb_17, qb_35, qb_45, qb_56, qb_67 = qb
            out = np.zeros(10)
            out[1] = -qb_21
            out[4] =  qb_45
            out[5] =  qb_56
            out[6] =  qb_67
            out[8] =  qb_35
            out[9] =  qb_17
            return out

        self.qbX_star = _expand_qb(self._qbX)
        self.qbZ_star = _expand_qb(self._qbZ)

    # ----------------------------------------
    # Private - Equivalent elastic moduli
    # ----------------------------------------

    def _compute_equivalent_moduli(self):
        '''
        Area-weighted average membrane (E1, E2) and bending (E1_bend,
        E2_bend) moduli over the 10 T2 sub-panels plus the 4 flanges.

        Membrane moduli (E1_eq, E2_eq) are used for axial stiffness EA.
        Bending moduli (E1_bend_eq, E2_bend_eq) are used for bending
        stiffness EIy and EIz in the beam element.
        '''
        E1_arr      = np.array([r['E1']      for r in self.T2])
        E2_arr      = np.array([r['E2']      for r in self.T2])
        E1_bend_arr = np.array([r['E1_bend'] for r in self.T2])
        E2_bend_arr = np.array([r['E2_bend'] for r in self.T2])
        total_A = float(np.sum(self.A_k)) + float(np.sum(self.A_flange))

        self.E1_eq = float(
            (np.dot(E1_arr, self.A_k)
             + np.dot(self.E1_flange, self.A_flange)) / total_A
        )
        self.E2_eq = float(
            (np.dot(E2_arr, self.A_k)
             + np.dot(self.E1_flange, self.A_flange)) / total_A
        )
        self.E1_bend_eq = float(
            (np.dot(E1_bend_arr, self.A_k)
             + np.dot(self.E1_bend_flange, self.A_flange)) / total_A
        )
        self.E2_bend_eq = float(
            (np.dot(E2_bend_arr, self.A_k)
             + np.dot(self.E2_bend_flange, self.A_flange)) / total_A
        )
        self.G_eq = self.G_REF

    # ----------------------------------------
    # Private - Skew matrix
    # ----------------------------------------

    def _get_skew_matrix(self) -> np.ndarray:
        '''
        Build the cross-product matrix [r_C_to_SC]_x expressed in the beam
        local frame (intermediate, before the R_c web-angle rotation).

        The section's u-axis maps to the beam local y-axis and the
        section's w-axis to the beam local z-axis, so
            r_C_to_SC (in beam local frame) = (0, us, ws)
        and the corresponding cross-product matrix is
            [r]_x = [[0, -ws,  us ],
                    [ws,  0,   0  ],
                    [-us, 0,   0  ]].
        This matrix is consumed by BeamElement._offset_matrix as the
        rigid-offset transformation block (i.e., delta-u = -[r]_x @
        delta-theta), so it must be the *signed* [r_C_to_SC]_x, not
        [r_SC_to_C]_x.
        '''
        ry, rz = self.us, self.ws
        self.skew_matrix = np.array([
            [ 0.0,  -rz,   ry ],
            [ rz,   0.0,   0.0 ],
            [-ry,   0.0,   0.0 ],
        ])

    def _get_skew_matrix_ac(self) -> None:
        '''
        Build the cross-product matrix [r_AC_to_SC]_x in the beam local frame,
        used to translate applied aerodynamic loads from the aerodynamic
        centre (AC = LE + 0.25 * chord) to the shear centre (SC).

        AC is the natural reference for the external aerodynamic loads
        (lift, drag, pitching moment). The global stiffness matrix is
        assembled at the SC, so the load vector must first be translated
        from AC to SC via the force-couple rule
            F_SC = F_AC
            M_SC = M_AC - [r_AC_to_SC]_x @ F_AC.

        The (u, w) components of r_AC_to_SC are stored here using the same
        layout as _get_skew_matrix, so the downstream consumer
        (MeshBuilder._assemble) can build the 6x6 force-couple block
        mechanically.
        '''
        LE_local = np.array([self.xu[ 0], self.yu[ 0]])
        TE_local = np.array([self.xu[-1], self.yu[-1]])
        AC_xz    = LE_local + 0.25 * (TE_local - LE_local)

        u_AC = AC_xz[0] - self.Xc
        w_AC = AC_xz[1] - self.Zc

        ry = self.us - u_AC
        rz = self.ws - w_AC

        self.skew_matrix_ac = np.array([
            [ 0.0,  -rz,   ry  ],
            [ rz,   0.0,   0.0 ],
            [-ry,   0.0,   0.0 ],
        ])

    # ----------------------------------------
    # Private - Direct stress aux
    # ----------------------------------------
    
    def _get_constants_direct_stress(self):
        '''
        Per-boom coefficients that multiply MX and MZ in the direct
        stress decomposition

            sigma_i = N / A + IXstar_i * MX + IZstar_i * MZ

        with

            IXstar_i = (IZZ * w_i - IXZ * u_i) / den
            IZstar_i = (IXX * u_i - IXZ * w_i) / den
            den      = IXX * IZZ - IXZ**2

        where (u_i, w_i) are the boom coordinates relative to the
        centroid. Both arrays have shape (N_BOOMS,).
        '''
        IXX = self.I_XX
        IZZ = self.I_ZZ
        IXZ = self.I_XZ

        den = IXX * IZZ - IXZ * IXZ
        self.IXstar = (IZZ * self.boom_w - IXZ * self.boom_u) / den
        self.IZstar = (IXX * self.boom_u - IXZ * self.boom_w) / den


    # ----------------------------------------------------------------
    # Private - Pack geometrical properties
    # ----------------------------------------------------------------

    def _pack_geom_data(self) -> GeomData:
        dX = float(self.LE_xz[0])
        dZ = float(self.LE_xz[1])

        # -------- Translated copies of coordinate topology dicts --------
        def _tr_pts(pts: np.ndarray) -> np.ndarray:
            p = pts.copy()
            p[:, 0] += dX
            p[:, 1] += dZ
            return p

        def _tr_seg(seg: dict) -> dict:
            return {**seg, 'pts': _tr_pts(seg['pts'])}

        T1_gl = [_tr_seg(s) for s in self.T1]
        T2_gl = [_tr_seg(r) for r in self.T2]
        T3_gl = [_tr_seg(c) for c in self.T3]
        T4_gl = [
            {**f, 'pts': f['pts'] + np.array([dX, dZ])}
            for f in self.T4
        ]

        T4_xz_gl = self.T4_xz.copy()
        T4_xz_gl[:, 0] += dX
        T4_xz_gl[:, 1] += dZ

        P_XZ_gl = self.P_XZ.copy()
        P_XZ_gl[:, 0] += dX
        P_XZ_gl[:, 1] += dZ

        return GeomData(
            chord     = self.chord,
            xw1       = self.xw1_frac,
            xw2       = self.xw2_frac,
            C         = np.array([self.Xc + dX, self.Y_sta, self.Zc + dZ]),
            A         = self.A,
            I_XX      = self.I_XX,
            I_ZZ      = self.I_ZZ,
            I_XZ      = self.I_XZ,
            I_1       = self.I_1,
            I_2       = self.I_2,
            theta_P   = self.theta_P,
            A_cells   = self.A_cells,
            J         = self.J,
            G_REF     = self.G_REF,
            s_k       = self.s_k,
            A_k       = self.A_k,
            delta_k   = self.delta_k,
            delta_mat = self.delta_mat,
            Delta_mat = self.Delta_mat,
            boom_lbls = self.boom_labels,
            boom_Xc   = self.boom_Xc + dX,
            boom_Zc   = self.boom_Zc + dZ,
            boom_u    = self.boom_u,
            boom_w    = self.boom_w,
            boom_A    = self.boom_A,
            IXstar    = self.IXstar,
            IZstar    = self.IZstar,
            stress_ratios = self.stress_ratios,
            S_XYZ     = np.array([self.Xs + dX, self.Y_sta, self.Zs + dZ]),
            S_uvw     = np.array([self.us, 0.0, self.ws]),
            qsX_star  = self.qsX_star,
            qsZ_star  = self.qsZ_star,
            qT_star   = self.qT_star,
            qbX_star  = self.qbX_star,
            qbZ_star  = self.qbZ_star,
            qs0X_star = self.qs0X_star,
            qs0Z_star = self.qs0Z_star,
            qs0T_star = self.qs0T_star,
            xi0       = self.xi0,
            eta0      = self.eta0,
            E1_eq      = self.E1_eq,
            E2_eq      = self.E2_eq,
            G_eq       = self.G_eq,
            E1_bend_eq = self.E1_bend_eq,
            E2_bend_eq = self.E2_bend_eq,
            t_k       = self.t_k,
            G_k       = self.G_k,
            A_flange  = self.A_flange,
            A_stringer = self.A_stringer,
            T1        = T1_gl,
            T2        = T2_gl,
            T3        = T3_gl,
            T4        = T4_gl,
            T4_xz     = T4_xz_gl,
            T4_A      = self.T4_A,
            T2_boom_idx = np.array(
                [[r['boomA'], r['boomB']] for r in self.T2],
                dtype=int,
            ),
            P_XZ      = P_XZ_gl,
            P_uvw     = self.P_uvw,
            skew_matrix    = self.skew_matrix,
            skew_matrix_ac = self.skew_matrix_ac,
        )


    # ----------------------------------------------------------------
    # Public - Run full pipeline
    # ----------------------------------------------------------------

    def run(self) -> GeomData:
        '''Execute the full geometric properties pipeline.'''
        self.logger.info(f"Computing geom properties at Y={self.Y_sta:.1f} mm")

        # Step 1: Build dimensional airfoil
        self._scale_airfoil()

        # Step 2: Find spar intersections (B1-B7, P1-P6)
        self._find_spar_intersections()

        # Step 3: Segment into T1, T2, T3 and T4
        self._segment_T1()
        self._segment_T2()
        self._segment_T3()
        self._segment_T4()

        # Step 4: Segment properties (s_k, A_k, G_REF, delta_k)
        self._compute_segment_properties()

        # Step 4b: Pre-compute panel moment sums
        self._precompute_panel_moments()

        # Step 5: Delta matrix
        self._compute_delta_matrix()

        # Step 6: Cell areas
        self._compute_cell_areas()

        # Step 7: Combined Delta matrix and torsional constant J
        self._compute_Delta_and_J()

        # Step 8: Centroid
        self._compute_centroid()

        # Step 9: Moments of inertia
        self._compute_inertia()

        # Step 10: Principal inertias
        self._compute_principal_inertia()

        # Steps 11-12: Structural idealization (boom areas + T4')
        si        = StructuralIdealization(self, enable_logging=False)
        boom_data = si.run()

        self.boom_labels   = boom_data.boom_labels
        self.boom_A        = boom_data.boom_A
        self.boom_u        = boom_data.boom_u
        self.boom_w        = boom_data.boom_w
        self.stress_ratios = boom_data.boom_rat

        self.boom_Xc = boom_data.Xc
        self.boom_Zc = boom_data.Zc
        if self.use_boom_centroid:
            self.Xc = self.boom_Xc
            self.Zc = self.boom_Zc

        self.I_XX    = boom_data.I_XX
        self.I_ZZ    = boom_data.I_ZZ
        self.I_XZ    = boom_data.I_XZ
        self.I_1     = boom_data.I_1
        self.I_2     = boom_data.I_2
        self.theta_P = boom_data.theta_P
        if self.recalculate_props:
            self.A = float(np.sum(self.boom_A))

        # Step 13: Shear center
        self._compute_shear_center()

        # Step 14: Adimensional shear flux
        self._compute_shear_flux()

        # Step 15: Equivalent moduli
        self._compute_equivalent_moduli()

        # Step 16: Build skew matrix
        self._get_skew_matrix()

        # Step 16b: AC->SC skew matrix (load translation block)
        self._get_skew_matrix_ac()

        # Step 17: Constants in direct stress calculations
        self._get_constants_direct_stress()

        # Pack results
        self.P_XZ  = np.array([self.P1, self.P4, self.P5, self.P6])
        self.P_uvw = self.P_XZ - np.array([self.Xc, self.Zc])  # (4,2) XZ offset

        return self._pack_geom_data()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
