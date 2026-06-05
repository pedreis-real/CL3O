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
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================


# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# FEA
from cl3o.fea.elements.beam_element import BeamElement


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
    skew_nodes   : np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 0)))


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
        use_offset     : bool = True,
        enable_logging : bool = True,
        verbose        : bool = False,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging, verbose)

        # Unpack inputs
        fem_setup = data[0]
        sections  = data[1]
        self.use_offset = use_offset

        self.beam_cache = fem_setup.beam_cache

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


    # ----------------------------------------
    # Private - Assemble pipeline
    # ----------------------------------------

    def _assemble(self) -> None:
        '''Build per-element matrices and saves into MeshData.'''
        n, m, dof, mcn = self.n, self.m, self.dof, self.mcn

        self.logger.debug(
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
            conn_i = self.conn[i, :]
            geomA  = self.sec_data[int(conn_i[0])]
            geomB  = self.sec_data[int(conn_i[1])]
            rls    = int(2 * conn_i[2] + conn_i[3])

            # Step 3. Beam element — use cache when geomA/geomB are reused.
            # Key on the GeomData content key (set by SectionBuilder) so the
            # beam cache stays valid even when the geometry cache evicts and
            # later rebuilds an identical section. Fall back to id() for any
            # direct-construction path that did not populate cache_key.
            keyA = geomA.cache_key if geomA.cache_key is not None else id(geomA)
            keyB = geomB.cache_key if geomB.cache_key is not None else id(geomB)
            beam_key = (keyA, keyB, rls, self.use_offset)
            cached   = self.beam_cache.get(beam_key)

            ei  = mcn[i]
            idx = np.ix_(ei, ei)

            if cached is not None:
                k_gl_i, T_sc_i, T_sc_gl_i, T_c_i, T_c_gl_i, R_i, G_i = cached
            else:
                # Step 3b. Build beam element matrices
                C         = self.coord[conn_i[1]] - self.coord[conn_i[0]]
                beam_data = BeamElement(geomA, geomB, C, rls, self.use_offset, enable_logging=False).data

                k_gl_i = beam_data.k_gl
                k_sc   = beam_data.k_sc_r
                R_i    = beam_data.Rmatrix
                G_i    = beam_data.Gmatrix

                # Analytical inverse of G (G = I + nilpotent correction → G_inv = I − correction).
                # G has nonzero off-diagonal only at [0:3,3:6] and [6:9,9:12];
                # those blocks square to zero, so (I−C)(I+C) = I exactly.
                G_inv_i          = np.eye(12)
                G_inv_i[0:3, 3:6 ] = -G_i[0:3, 3:6 ]
                G_inv_i[6:9, 9:12] = -G_i[6:9, 9:12]

                T_sc_i    = k_sc @ R_i.T
                T_sc_gl_i = R_i @ T_sc_i
                T_c_i     = G_inv_i.T @ k_sc @ R_i.T
                T_c_gl_i  = R_i @ T_c_i

                self.beam_cache[beam_key] = (
                    k_gl_i, T_sc_i, T_sc_gl_i, T_c_i, T_c_gl_i, R_i, G_i,
                )

            # Step 4. Assemble
            K[idx] += k_gl_i
            Ei[:, i] = ei
            R[:, :, i] = R_i
            G[:, :, i] = G_i

            # Contravariant transformation for SC: obtain internal
            # forces along shear center (SC)
            #       T_sc    = k @ R^T
            #       Q_sc    = T_sc @ d_gl
            #       Q_sc_gl = R^{-T} @ T_sc @ d_gl = R @ T_sc @ d_gl
            T_sc[:, :, i]     = T_sc_i
            T_sc_gl[:, :, i]  = T_sc_gl_i

            # Contravariant transformation for C: obtain internal
            # forces along centroid (C)
            #       T_c    = G^{-T} @ k @ R^T
            #       Q_c    = T_c @ d_gl
            #       Q_c_gl = R^{-T} @ T_c @ d_gl = R @ T_c @ d_gl
            T_c[:, :, i]     = T_c_i
            T_c_gl[:, :, i]  = T_c_gl_i

        self.logger.debug(
            f"K assembled — sparsity "
            f"{100.0 * np.count_nonzero(K) / K.size:.1f}% non-zero"
        )

        # Step 4b. Per-node matrices: AC->SC force couple and SC->C displacement offset
        T_load_ac_sc = np.zeros((6, 6, n))
        skew_nodes   = np.zeros((3, 3, n))
        for j in range(n):
            S = self.sec_data[j].skew_matrix_ac
            block = np.eye(6)
            block[3:6, 0:3] = -S
            T_load_ac_sc[:, :, j] = block
            skew_nodes[:, :, j]   = self.sec_data[j].skew_matrix

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
            skew_nodes   = skew_nodes.astype(float),
        )
