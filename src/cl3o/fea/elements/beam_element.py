'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Beam Element Module.

Computes the local 12x12 stiffness matrix, fixed-end force vector, shear-
centre offset transformation, and member-release modification for a single
3-D Euler-Bernoulli beam element at one spanwise station.

The formulation follows MSA.m (Rahami, 2010), extended with a rigid-offset
congruence transformation that shifts all stiffness quantities from the
cross-section centroid C to the SC SC before the release
modification is applied.

All matrices are expressed in the LOCAL coordinate system aligned with the
principal inertia axes of the cross-section:
    local y....aligned with MINOR inertia I2
    local z....aligned with MAJOR inertia I1

Pipeline:
----------------
    1. Extract section properties from GeomData.
    2. Centroidal stiffness [k]
    3. Offset matrix [G]
    4. Offset transform:
        [k_sc]  = [G]^T [k]  [G]
    5. Release matrix [M]
    6. Release application:
        [k_sc_r]  = [M] [k_sc]
    7. Rotation matrix [T]
    8. Global transformation:
        [k_gl] = [T]^T [k_sc_r] [T]

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
from cl3o.Constants import TOL, SPHERICAL_HINGE

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils import math_utils as mthu

# Geometry
from cl3o.geometry.geom_properties import GeomData


# ================================================================================
# Data persistence - Beam element matrices for a single element
# ================================================================================

@dataclass
class BeamData:
    '''
    Container for all matrices produced for a single 3-D Euler-Bernoulli
    beam element, expressed in the LOCAL coordinate system aligned with
    the principal inertia axes of the cross-section.

    Property    Size        Description                                 Units
    --------    --------    ----------------------------------------    -------------
    L           (1,)        Element length                              mm
    EA          (1,)        Axial member stiffness                      N
    EIy         (1,)        Bending along y-axis member stiffness       N*mm^2
    EIz         (1,)        Benting along z-axis member stiffness       N*mm^2
    GJ          (1,)        Torsional member stiffness                  N*mm^2

    k           (12, 12)    Local stiffness at centroid,                N/mm, N, N*mm
    k_sc        (12, 12)    Local stiffness at SC,                      N/mm, N, N*mm
    k_sc_r      (12, 12)    Released local stiffness at SC              N/mm, N, N*mm
    k_gl        (12, 12)    Local stiffness in global coordinates,      N/mm, N, N*mm

    Gmatrix     (12, 12)    Rigid-offset transformation matrix          mm, -
    Mmatrix     (12, 12)    Member-release modification matrix          -
    R_a         (3, 3)      Rotation matrix along w                     - [rad]
    R_b         (3, 3)      Rotation matrix along y                     - [rad]
    R_c         (3, 3)      Rotation matrix along x                     - [rad]
    c_rad
    Rmatrix     (12, 12)    DOF rotation matrix (local to global)       - [rad]
    '''
    L   : float = 0.0
    EA  : float = 0.0
    EIy : float = 0.0
    EIz : float = 0.0
    GJ  : float = 0.0

    k       : np.ndarray = field(default_factory=lambda: np.zeros((12, 12)))
    k_sc    : np.ndarray = field(default_factory=lambda: np.zeros((12, 12)))
    k_sc_r  : np.ndarray = field(default_factory=lambda: np.zeros((12, 12)))
    k_gl    : np.ndarray = field(default_factory=lambda: np.zeros((12, 12)))

    Gmatrix : np.ndarray = field(default_factory=lambda: np.eye(12))
    Mmatrix : np.ndarray = field(default_factory=lambda: np.eye(12))
    R_a     : np.ndarray = field(default_factory=lambda: np.eye(3))
    R_b     : np.ndarray = field(default_factory=lambda: np.eye(3))
    R_c     : np.ndarray = field(default_factory=lambda: np.eye(3))
    c_rad   : float = 0.0
    Rmatrix : np.ndarray = field(default_factory=lambda: np.eye(12))


# ================================================================================
# PUBLIC API - 3D Euler-Bernoulli beam element builder
# ================================================================================

class BeamElement:
    '''
    Builds all local matrices for a single 3-D Euler-Bernoulli beam element
    at one spanwise station from a GeomData cross-section.

    Outputs are stored as self.data : BeamData.

    Inputs are all geometric paremeters for both ends of the beam element,
    so that a implementation of tapered beam feels like a natural path to
    work along. (commit to us!)
    '''
    def __init__(
        self,
        geomA : GeomData,
        geomB : GeomData,
        coord_vector : np.ndarray,
        release_type : int,
        enable_logging : bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)

        # %-format defers the to-string conversion until the handler actually
        # decides to emit. The DE inner loop creates ~NP*(1+k_max) beams per
        # generation and each one paid the f-string cost otherwise.
        self.logger.info("Building element at Y = %s mm", geomA.C[1])

        # Unpack inputs
        self.code = release_type

        self.C = coord_vector
        self.a, self.b, self.L = mthu.cart2sph(self.C)
        self.c = geomA.c_rad

        self.E1  = geomA.E1_eq        # membrane: used for EA
        self.E2  = geomA.E2_eq        # membrane: used for EA
        self.E1b = geomA.E1_bend_eq   # bending:  used for EIy
        self.E2b = geomA.E2_bend_eq   # bending:  used for EIz
        self.G   = geomA.G_eq
        self.A  = geomA.A
        self.Iy = geomA.I_2     # y-axis is the MINOR inertia
        self.Iz = geomA.I_1     # z-axis is the MAJOR inertia
        self.J  = geomA.J

        self.skewA = geomA.skew_matrix
        self.skewB = geomB.skew_matrix

        # Develop stiffness matrices
        self._build_matrices()


    # ----------------------------------------------------------------
    # Private - Matrices development
    # ----------------------------------------------------------------

    def _local_stiffness_matrix(self) -> np.ndarray:
        '''
        Assemble the 12x12 local stiffness at the centroid for an
        Euler-Bernoulli beam without transverse shear deformation.

        Returns:
            k_c : Centroidal stiffness matrix (12, 12)
        '''
        E1, E2, G        = self.E1, self.E2, self.G
        E1b, E2b         = self.E1b, self.E2b
        A, Iy, Iz, J, L  = self.A, self.Iy, self.Iz, self.J, self.L

        _coef_arr = np.array([12.0, 6.0 * L, 4.0 * L**2, 2.0 * L**2])
        _k_ax  = E1  * A  * L**2
        _k_fy  = E1b * Iy * _coef_arr
        _k_fz  = E2b * Iz * _coef_arr
        _k_rot = G   * J  * L**2

        # 3x3 sub-matrix block
        k1 = np.diag([_k_ax, _k_fz[0], _k_fy[0]])
        k2 = np.array([
            [0.0,  0.0,      0.0     ],
            [0.0,  0.0,      _k_fz[1]],
            [0.0, -_k_fy[1], 0.0     ],
        ])
        k3 = np.diag([ _k_rot, _k_fy[2], _k_fz[2]])
        k4 = np.diag([-_k_rot, _k_fy[3], _k_fz[3]])

        # 12x12 local stiffness matrix
        k_c = np.block([
            [ k1,    k2,   -k1,    k2],
            [ k2.T,  k3,   -k2.T,  k4],
            [-k1,   -k2,    k1,   -k2],
            [ k2.T,  k4,   -k2.T,  k3],
        ]) / (L**3)

        return k_c


    def _offset_matrix(self) -> np.ndarray:
        '''
        Assemble the 12x12 rigid-offset transformation that shifts the
        beam stiffness from the centroidal axis to the shear-centre axis.

        Each 6x6 nodal block is the standard kinematic transform
            d_SC = T_C_to_SC @ d_C,
        with
            T_C_to_SC = [[ I,  -[r_C_to_SC]_x ],
                         [ 0,   I             ]].
        Since skew_matrix already stores [r_C_to_SC]_x (see
        GeomPropCalculator._get_skew_matrix), and the G matrix is the
        inverse of T_C_to_SC, so no minus sign is needed when defining
        G directly.

        Returns:
            G_mat : Offset transformation matrix (12, 12)
        '''
        X_r1, X_r2 = self.skewA, self.skewB

        G_mat = np.eye(12)
        G_mat[0:3, 3:6 ] = X_r1
        G_mat[6:9, 9:12] = X_r2

        return G_mat
    

    def _release_matrix(self) -> np.ndarray:
        '''
        Build the 12x12 member-release modification matrix [M].

        Release code = 2 * conn[2] + conn[3]:
            3 : both ends fixed
            2 : released at end
            1 : released at beggining
            0 : both ends released

        Returns:
            M_mat : Release modification matrix (12, 12)
        '''
        L = self.L

        if SPHERICAL_HINGE:
            X = np.diag([1.0, -0.5, -0.5])
            W = np.diag([0.0, 0.0, 0.0])
        else:
            X = np.diag([0.0, -0.5, -0.5])
            W = np.diag([1.0, 0.0, 0.0])
        
        Y = np.array([
            [0.0,  0.0, 0.0],
            [0.0,  0.0, 1.5],
            [0.0, -1.5, 0.0],
        ]) / L
        Z = np.zeros((3,3))

        M_mat = np.eye(12)
        release_block = np.block([
            [-Y],
            [ W],
            [ Y],
            [ X],
        ])

        match self.code:
            case 3:
                return M_mat
            case 2: # Released at end
                M_mat[:, 9:12] = release_block
            case 1: # Released at beggining
                M_mat[:, 3:6] = release_block
            case 0: # Both ends released
                Y_scaled = (2.0 / 3.0) * Y
                release_block_both = np.block([
                    [-Y_scaled, -Y_scaled],
                    [ W,         Z       ],
                    [ Y_scaled,  Y_scaled],
                    [ Z,         W       ]
                ])
                M_mat[:, 3:6]  = release_block_both[:, 0:3]
                M_mat[:, 9:12] = release_block_both[:, 3:6]
            case _:
                raise ValueError(f"Value not expected."
                                 f" Failed with code: {self.code}")

        return M_mat


    def _rotation_matrix(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Build the 3x3 element rotation.

        The web-angle rotation R_c uses (theta_P - pi/2) rather than theta_P itself:
        mohr_circle returns the angle of the MAJOR principal axis from the section
        x-axis, while local-y in the beam frame must be aligned with the MINOR
        principal axis (to match the assignment self.Iy = I_2 in __init__).
        Subtracting pi/2 rotates local-y from y' to the minor direction.

        The principal angle theta_P is defined only modulo pi, so the minor-axis
        (local y) direction is ambiguous up to a 180deg flip. For a wing section
        I_XX < I_ZZ drives theta_P to ~+-90deg, and the (often noise-level) sign
        of I_XZ then decides whether it snaps to +90 or -90 -- flipping local y
        and inverting every recovered local bending moment. We resolve the
        ambiguity by pinning local y to point toward the trailing edge (+X), so
        the local frame and the Q_sc / Q_c moment signs stay stable along the
        span and consistent with the global X-Z stress-recovery frame. Flipping
        local y by 180deg co-flips local z (rotation about local x), leaving the
        global stiffness k_gl invariant.

        Return:
            R_a : Rotation matrix along centroidal axis w == z''
            R_b : Rotation matrix along intermediate axis y'
            R_c : Rotation matrix along local axis x == x'
        '''
        a, b, c = self.a, self.b, self.c

        R_a, R_b, R_c = mthu.rot3(a), mthu.rot2(b), mthu.rot3(c)

        gamma_rot = R_c @ R_b @ R_a

        T_mat = np.kron(np.eye(4), gamma_rot)

        return R_a, R_b, R_c, T_mat


    # ----------------------------------------------------------------
    # Private - Full pipeline
    # ----------------------------------------------------------------

    def _build_matrices(self) -> None:
        '''Execute the full 6-step pipeline and store results in self.data.'''

        # Step 2. Centroidal stiffness
        k_c = self._local_stiffness_matrix()

        # Steps 3. Offset matrix
        G_mat = self._offset_matrix()
        # G_mat = np.eye(12)

        # Steps 4. Offset transformation
        k_sc = G_mat.T @ k_c  @ G_mat

        # Steps 5. Member-release matrix
        M_mat = self._release_matrix()

        # Steps 6. Member-release transformation
        k_sc_r = M_mat @ k_sc

        # Steps 7. Rotation matrix
        R_a, R_b, R_c, R_mat = self._rotation_matrix()

        # Steps 8. Rotation transformation
        k_gl = R_mat @ k_sc_r @ R_mat.T

        # Pack results
        self.data = BeamData(
            L        = float(self.L),
            EA       = float(self.E1  * self.A),
            EIy      = float(self.E1b * self.Iy),
            EIz      = float(self.E2b * self.Iz),
            GJ       = float(self.G * self.J),
            k        = k_c.astype(float),
            k_sc     = k_sc.astype(float),
            k_sc_r   = k_sc_r.astype(float),
            k_gl     = k_gl.astype(float),
            Gmatrix  = G_mat.astype(float),
            Mmatrix  = M_mat.astype(float),
            R_a      = R_a.astype(float),
            R_b      = R_b.astype(float),
            R_c      = R_c.astype(float),
            c_rad    = self.c,
            Rmatrix  = R_mat.astype(float),
        )
