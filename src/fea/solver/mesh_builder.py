'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Mesh Builder Module.

Assembles the global finite-element model from the per-station GeomData
cross-sections and the structural mesh setup. Consumes BeamElement for
per-element local matrices and produces a MeshData container ready
for MSASolver.

The formulation follows MSA.m (Rahami, 2010), extended with a rigid-offset
congruence transformation that shifts all stiffness quantities from the
cross-section centroid C to the SC SC before the release
modification is applied.

Pipeline
----------------
    1. Get the Finite Element Method artifacts setup
    2. Get node positions from ExLoads database and cross-section centroid
    3. Build 12-DOF beam elements
    4. Assmble beam data
    5. Store mesh attributes

---
Note: The fixed end forces {Qf} and settlement displacements {st}
      transformations are responsabilities of :load_mapper: module.
      This one only carries about mesh topology.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================
_HERE = Path(__file__).resolve().parent           # src/fea/solver/
_SRC  = _HERE.parent.parent                       # src/

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Module imports ================

# Utilities
from utils import io_utils as io

# FEA
from fea.elements.beam_element import BeamElement


# ================================================================================
# Data persistence - Assembled global FEM arrays
# ================================================================================

@dataclass
class MeshData:
    '''
    Container for the global FEM arrays consumed by MSASolver.

    Property    Size            Description                         Units
    --------    ------------    --------------------------------    ----------------
    n           (1,)            Number of nodes                     -
    m           (1,)            Number of beam elements             -
    nc          (1,)            Number of load conditions           -
    dof         (1,)            Total number of DOFs (6 * n)        -

    coord       (n, 3)          Global nodal coordinates [X, Y, Z]  mm
    conn        (m, 4)          Connectivity and release (pers.)    -
    adr         (12, m)         Per-element DOF address arrays      -
    re_flat     (6n,)           Restraint vector (flattened)        -

    R           (12, 12, m)     Per-element rotation matrix         -
    G           (12, 12, m)     Per-element offset matrix           -

    K           (6n, 6n)        Global stiffness matrix             N/mm, N, N*mm/rad
    T_sc        (12, 12, m)     Transf. to obtain Q_sc (local)      N/mm, N, N*mm/rad
    T_c         (12, 12, m)     Transf. to obtain Q_c  (local)      N/mm, N, N*mm/rad
    T_sc_gl     (12, 12, m)     Transf. to obtain Q_sc (global)     N/mm, N, N*mm/rad
    T_c_gl      (12, 12, m)     Transf. to obtain Q_c  (global)     N/mm, N, N*mm/rad
    T_load_ac_sc (6, 6, n)      Per-node force-couple AC -> SC      -
    '''
    n   : int = 0
    m   : int = 0
    nc  : int = 0
    dof : int = 0

    coord   : np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    conn    : np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=int))
    adr     : np.ndarray = field(default_factory=lambda: np.zeros((12, 0), dtype=int))
    re_flat : np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))

    R : np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 0)))
    G : np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 0)))

    K       : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    T_sc    : np.ndarray = field(default_factory=lambda: np.zeros((12, 12, 0)))
    T_c     : np.ndarray = field(default_factory=lambda: np.zeros((12, 12, 0)))
    T_sc_gl : np.ndarray = field(default_factory=lambda: np.zeros((12, 12, 0)))
    T_c_gl  : np.ndarray = field(default_factory=lambda: np.zeros((12, 12, 0)))

    T_load_ac_sc : np.ndarray = field(default_factory=lambda: np.zeros((6, 6, 0)))


# ================================================================================
# PUBLIC API - Global FEM assembler
# ================================================================================

class MeshBuilder:
    '''
    Assembles the global FEM stiffness matrix from a pre-processed mesh setup
    and per-section GeomData, producing the MeshData container
    consumed by LinearStaticSolver.
    '''

    def __init__(
        self,
        data : tuple[object, object],
        enable_logging: bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)
        
        # Unpack inputs
        fem_setup = data[0]
        sections  = data[1]

        self.n       = fem_setup.n
        self.m       = fem_setup.m
        self.dof     = fem_setup.dof
        self.re      = fem_setup.re
        self.re_flat = fem_setup.re_flat
        self.conn    = fem_setup.conn
        self.mcn     = fem_setup.mcn
        self.nc      = int(fem_setup.loads['nc'])

        self.sec_data = sections.sec_data

        self._assemble()
    
    
    # ----------------------------------------
    # Private - Beam element construction
    # ----------------------------------------

    def _get_global_coordinates(self) -> None:
        '''Uses Section centroid as global coordinates of each node'''
        coord = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            coord[i, :] = self.sec_data[i].C
        self.coord = coord


    def _get_current_beam_element_data(
        self,
        conn_i : np.ndarray,
    ) -> None:
        '''
        Calculate local stiffness matrix.
        Assumes a prismatic cross-section from endA (base) to endB (tip)

        TODO - USE TAPERED BEAM ELEMENTS (commit to us!)
        '''
        geomA = self.sec_data[conn_i[0]]
        geomB = self.sec_data[conn_i[1]]
        C = self.coord[conn_i[1]] - self.coord[conn_i[0]]
        rls_code = 2 * conn_i[2] + conn_i[3]
        
        return BeamElement(
            geomA = geomA,
            geomB = geomB,
            coord_vector = C,
            release_type = rls_code,
            enable_logging=False,
        ).data


    # ----------------------------------------
    # Private - Assemble pipeline
    # ----------------------------------------

    def _assemble(self) -> None:
        '''Build per-element matrices and saves into MeshData.'''
        n, m, dof, mcn = self.n, self.m, self.dof, self.mcn

        self.logger.info(
            "Assembling global stiffness matrix "
            f"[n={n} nodes, m={m} elements, dof={dof}]"
        )

        # Step 2. Get global coordinates (once, before loop)
        self._get_global_coordinates()

        # Pre-allocate global arrays
        R       = np.zeros((12, 12, m))
        G       = np.zeros((12, 12, m))
        K       = np.zeros((dof, dof))
        T_sc    = np.zeros((12, 12, m))
        T_sc_gl = np.zeros((12, 12, m))
        T_c     = np.zeros((12, 12, m))
        T_c_gl  = np.zeros((12, 12, m))
        Ei      = np.zeros((12, m), dtype=int)
        for i in range(m):
            # Step 3. Beam element data
            beam_data = self._get_current_beam_element_data(
                conn_i = self.conn[i,:],
            )

            # Step 4. Assemble
            ei = mcn[i]
            idx = np.ix_(ei, ei)
            
            # ---- unpacking ----
            k_gl    = beam_data.k_gl
            k_sc    = beam_data.k_sc_r
            R_i     = beam_data.Rmatrix
            G_i     = beam_data.Gmatrix
            G_inv_i = np.linalg.solve(G_i, np.eye(12))

            K[idx] += k_gl
            Ei[:, i] = ei
            R[:, :, i] = R_i
            G[:, :, i] = G_i

            # Contravariant transformation for SC: obtain internal
            # forces along shear center (SC)
            #       Q_sc    = T_sc @ d_gl
            #       Q_sc_gl = R^{-1} @ T_sc @ d_gl = R @ T_sc @ d_gl
            T_sc[:, :, i] = k_sc @ R_i.T
            T_sc_gl[:, :, i] = R_i @ T_sc[:, :, i]

            # Contravariant transformation for C: obtain internal
            # forces along centroid (C)
            #       Q_c    = T @ d_gl
            #       Q_c_gl = R^{-T} @ T @ d_gl = R @ T @ d_gl
            T_c[:, :, i] = G_inv_i.T @ k_sc @ R_i.T
            T_c_gl[:, :, i] = R_i @ T_c[:, :, i]

        self.logger.debug(
            f"K assembled — sparsity "
            f"{100.0 * np.count_nonzero(K) / K.size:.1f}% non-zero"
        )

        # Step 4b. Per-node AC -> SC force-couple translation
        # F_SC = F_AC ; M_SC = M_AC - [r_AC_to_SC]_x @ F_AC
        T_load_ac_sc = np.zeros((6, 6, n))
        for j in range(n):
            S = self.sec_data[j].skew_matrix_ac   # 3x3 = [r_AC_to_SC]_x
            block = np.eye(6)
            block[3:6, 0:3] = -S
            T_load_ac_sc[:, :, j] = block

        # Step 5. Store
        self.data = MeshData(
            n       = n,
            m       = m,
            nc      = self.nc,
            dof     = int(dof),
            coord   = self.coord.astype(float),
            conn    = self.conn.astype(int),
            adr     = Ei.astype(int),
            re_flat = self.re_flat.astype(int),
            R       = R.astype(float),
            G       = G.astype(float),
            K       = K.astype(float),
            T_sc    = T_sc.astype(float),
            T_c     = T_c.astype(float),
            T_sc_gl = T_sc_gl.astype(float),
            T_c_gl  = T_c_gl.astype(float),
            T_load_ac_sc = T_load_ac_sc.astype(float),
        )