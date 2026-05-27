'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
MSA Solver Module.

Solves the linear static equation {F} = [K]{d}, reconstructs reactions, and
returns per-element internal forces both at the local principal-inertia axes
(xyz, at the shear centre) and at the centroidal axis (uvw).

Pipeline
----------------
    1. Unpack inputs from MeshData and FemPreprocessData
        (mesh.T_load_ac_sc carries the per-node AC -> SC force-couple block)
    2. Partition DOFs into free and constrained sets
        V_flag  = 1 - (restraints | settlements)
    2.5 Translate F_i from AC to SC, per-node
        F_SC = F_AC ; M_SC = M_AC - [r_AC_to_SC]_x @ F_AC
    3. Solve the condensed system for DOFs
        [K_ff] {d_f} = {F_f}
    4. Reconstruct the full displacement vector {d}
        {d}[f]   = {d_f}
    5. Support reactions
        {R}      = [K]{d} - {F}
        {R}[f]   = 0
    6. Per-element internal forces at the shear centre in LOCAL xyz
        {Q_sc}_i = [Ni_i] {d[e_i]} + {Qfi_i}
    and in GLOBAL XYZ
        {Q_gl}_i = [Ni_gl_i] {d[e_i]} + {Qfi_gl_i}
    7. Reshap vectors into per-element matrices

DOF ordering
----------------
    Internal forces:
        0..2    forces  [Fx, Fy, Fz]   at beginning node    [N]
        3..5    moments [Mx, My, Mz]   at beginning node    [N*mm]
        6..8    forces  [Fx, Fy, Fz]   at end node          [N]
        9..11   moments [Mx, My, Mz]   at end node          [N*mm]
    
    Displacement:
        0..2    translation [u, v, w]   at beginning node    [mm]
        3..5    rotation    [p, q, r]   at beginning node    [rad]
        6..8    translation [u, v, w]   at end node          [mm]
        9..11   rotation    [p, q, r]   at end node          [rad]

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================


# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# FEA
from cl3o.fea.solver.mesh_builder import MeshData



# ================================================================================
# Data persistence - MSA solver results
# ================================================================================

@dataclass
class FeaResults:
    '''
    Container for the output of a single linear static solve.

    Property    Size            Description                         Units
    --------    ------------    --------------------------------    --------
    n           (1,)            Number of nodes                     -
    m           (1,)            Number of elements                  -
    nc          (1,)            Number of load conditions           -
    nf          (1,)            Number of unconstrained DOFs        -
    f           (nf,)           Indices of unconstrained DOFs       -

    d           (6n, nc)        Flattened nodal displacements       mm, rad
    dmatrix     (6, n, nc)      Reshaped displacements per node     mm, rad

    R           (6n, nc)        Flattened support reactions         N, N*mm
    Rmatrix     (6, n, nc)      Reshaped reactions per node         N, N*mm

    Q_sc        (12, m, nc)     Internal forces at the SC (xyz)     N, N*mm
    Q_sc_gl     (12, m, nc)     Internal forces at the SC (XYZ)     N, N*mm
    Q_c         (12, m, nc)     Internal forces at the C (xyz)      N, N*mm
    Q_c_gl      (12, m, nc)     Internal forces at the C (XYZ)      N, N*mm
    '''
    n  : int = 0
    m  : int = 0
    nc : int = 0
    nf : int = 0
    f  : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    d       : np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    dmatrix : np.ndarray = field(default_factory=lambda: np.zeros((6, 0, 1)))

    R       : np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    Rmatrix : np.ndarray = field(default_factory=lambda: np.zeros((6, 0, 1)))

    Q_sc    : np.ndarray = field(default_factory=lambda: np.zeros((12, 0, 1)))
    Q_c     : np.ndarray = field(default_factory=lambda: np.zeros((12, 0, 1)))
    Q_sc_gl : np.ndarray = field(default_factory=lambda: np.zeros((12, 0, 1)))
    Q_c_gl  : np.ndarray = field(default_factory=lambda: np.zeros((12, 0, 1)))



# ================================================================================
# PUBLIC API - Linear static FEM solver
# ================================================================================

class LinearStaticSolver:
    '''
    Linear static solver for the CL3O beam numerical model.

    The solver consumes the MeshData produced by MeshBuilder and, at
    construction, solves {F} = [K]{d} for the free DOFs, for every load
    condition.
    
    The module runs all pipeline in __init__, populating FeaResults with
    respective fields
    '''

    def __init__(
        self,
        mesh : MeshData,
        loads : dict[str, Any],
        enable_logging : bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)

        # Step 1. Store inputs
        self.mesh = mesh
        self.F = loads['F_flat']
        self.nc = loads['nc']

        self._solve()

    # ----------------------------------------
    # Private metrhods - Solution pipeline
    # ----------------------------------------

    def _partition_dofs(
        self
    ) -> tuple[np.ndarray, np.ndarray]:
        '''Identify free and constrained DOFs.'''
        is_constrained    = self.mesh.re_flat.astype(bool)
        free_dofs         = np.where(~is_constrained)[0].astype(int)
        constrained_dofs  = np.where( is_constrained)[0].astype(int)
        return free_dofs, constrained_dofs

    # ----------------------------------------
    # Private - Solution pipeline
    # ----------------------------------------

    def _solve(self) -> None:
        '''
        Execute the steps for static solution and pack results.

        Steps
        --------
            2. Partition DOFs into free and constrained sets
                V_flag  = 1 - (restraints | settlements)
            2.5 Translate F_i from AC to SC, per-node
                F_SC = F_AC ; M_SC = M_AC - [r_AC_to_SC]_x @ F_AC
            3. Solve the condensed system for DOFs
                [K_ff] {d_f} = {F_f}
            4. Reconstruct the full displacement vector {d}
                {d}[f]   = {d_f}
            5. Support reactions
                {R}      = [K]{d} - {F}
                {R}[f]   = 0
            6. Per-element internal forces at the shear centre in LOCAL xyz
                {Q_sc}_i = [Ni_i] {d[e_i]}
            and in GLOBAL XYZ
                {Q_gl}_i = [Ni_gl_i] {d[e_i]}
            7. Reshape vector into per-element matrices
        '''
        n   = self.mesh.n
        m   = self.mesh.m
        dof = self.mesh.dof
        nc  = self.nc

        self.logger.info(
            f"Solving linear static system "
            f"[dof={dof}, nc={nc} load conditions]"
        )

        # Step 2. Partition DOFs
        f, not_f = self._partition_dofs()
        nf = int(f.size)

        self.logger.debug(f"DOF partition: {nf} free, {dof - nf} constrained")

        # Step 3. Solve condensed system
        K_ff = self.mesh.K[np.ix_(f, f)]
        
        d_sc_gl = np.zeros((dof, nc))
        R = np.zeros((dof, nc))
        dmatrix = np.zeros((6, n, nc))
        Rmatrix = np.zeros((6, n, nc))
        Q_sc    = np.zeros((12, m, nc))
        Q_sc_gl = np.zeros((12, m, nc))
        Q_c     = np.zeros((12, m, nc))
        Q_c_gl  = np.zeros((12, m, nc))

        T_load_ac_sc = self.mesh.T_load_ac_sc   # (6, 6, n)
        for i in range(nc):
            F_i = self.F[:, i].copy()           # decouple from self.F view

            # Step 2.5. Translate F_i from AC (CA = 0.25*chord) to SC, per-node.
            # F_SC = F_AC ; M_SC = M_AC - [r_AC_to_SC]_x @ F_AC.
            for j in range(n):
                sl = slice(6 * j, 6 * (j + 1))
                F_i[sl] = T_load_ac_sc[:, :, j] @ F_i[sl]

            try:
                d_free = np.linalg.solve(K_ff, F_i[f])
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    f"[CL3O] Singular stiffness matrix K_ff.\n"
                    f"| free DOFs : {nf}\n"
                    f"| K_ff shape: {K_ff.shape}\n"
                    f"Check boundary conditions - this is a signature"
                    f" of rigid bosy motion."
                    f"Underlying error: {exc}"
                )

            # Step 4. Displacement vector (global, in SC)
            d_i    = np.zeros((dof))
            d_i[f] = d_free

            # Step 5. Reaction forces
            R_i     = self.mesh.K @ d_i - F_i
            R_i[f]  = 0.0

            # Step 6. Internal forces
            for j in range(m):
                adress  = self.mesh.adr[:, j]
                T_sc    = self.mesh.T_sc[:, :, j]
                T_sc_gl = self.mesh.T_sc_gl[:, :, j]
                T_c    = self.mesh.T_c[:, :, j]
                T_c_gl = self.mesh.T_c_gl[:, :, j]

                Q_sc[:, j, i] = T_sc @ d_i[adress]
                Q_sc_gl[:, j, i] = T_sc_gl @ d_i[adress]

                Q_c[:, j, i] = T_c @ d_i[adress]
                Q_c_gl[:, j, i] = T_c_gl @ d_i[adress]

            # Step 7. Reshape
            d_sc_gl[:, i] = d_i
            R[:, i] = R_i
            dmatrix[:, :, i] = d_i.reshape(6, n, order='F')
            Rmatrix[:, :, i] = R_i.reshape(6, n, order='F')

        # -------- Pack results --------
        self.results = FeaResults(
            n = self.mesh.n,
            m = self.mesh.m,
            nc = self.mesh.nc,
            nf = nf,
            f = f.astype(int),
            d = d_sc_gl.astype(float),
            dmatrix = dmatrix.astype(float),
            R = R.astype(float),
            Rmatrix = Rmatrix.astype(float),
            Q_sc = Q_sc.astype(float),
            Q_sc_gl = Q_sc_gl.astype(float),
            Q_c = Q_c.astype(float),
            Q_c_gl = Q_c_gl.astype(float),
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
