'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Displacement Margins Module.

Evaluates the displacement-based margins of safety from the global nodal
displacement matrix produced by LinearStaticAnalysis:

    MS_u  = u_limit  / u  - 1         (deflection)
    MS_th = th_limit / th - 1         (twist / rotation)
    MS_min_node = min(MS_u, MS_th)
    MS_min = min(MS_min_node)

A violation-counter (nv) is increased whenever any MS < 0. Also, the
information of which station got a violation flag is stored in
DisplacementData so the user can see where wing fails (debug).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    U_MAX_FACTOR, THETA_MAX, LARGE_DISPL_MS, TOL
)

# Utilities
from cl3o.utils import io_utils as io

# FEA
from cl3o.fea.solver.mesh_builder import MeshData


# ================================================================================
# Data persistence - Displacement margins container
# ================================================================================

@dataclass
class DisplacementData:
    '''
    Container for the displacement-based margins of safety.

    Property        Size            Description                             Units
    ------------    ------------    ----------------------------------    --------
    n               (1,)            Number of nodes                       -
    m               (1,)            Number of elements                    -
    nc              (1,)            Number of load conditions             -
    MS_u            (3, n, nc)      Margin of safety - translation        -
    MS_th           (3, n, nc)      Margin of safety - rotation           -
    MS_min_node     (n,)            Per-node min margin                   -
    MS_min          (1,)            Global min margin                     -
    nv              (1,)            Count of violations (MS < 0)          -
    '''
    n  : int = 0
    m  : int = 0
    nc : int = 0

    MS_u        : np.ndarray = field(
        default_factory=lambda: np.full((3, 0, 1), LARGE_DISPL_MS)
    )
    MS_th       : np.ndarray = field(
        default_factory=lambda: np.full((3, 0, 1), LARGE_DISPL_MS)
    )
    MS_min_node : np.ndarray = field(
        default_factory=lambda: np.full((0,), LARGE_DISPL_MS)
    )
    MS_min : float = LARGE_DISPL_MS
    nv     : int   = 0


# ================================================================================
# PUBLIC API - Displacement margin evaluator
# ================================================================================

class DisplacementMargins:
    '''
    Compute the vertical-displacement and rotation margins of safety
    from the global displacement vector returned by MSASolver.

    Use:
        disp  = DisplacementMargins(dmatrix, b)
        data  = disp.data                           # DisplacementData
    '''

    def __init__(
        self,
        mesh           : MeshData,
        dmatrix        : np.ndarray,
        b              : float,
        enable_logging : bool = True,
        verbose        : bool = False,
    ) -> None:
        '''
        Args:
            mesh            : MeshData with n, m, nc fields.
            dmatrix         : (6, n, nc) global per-node displacement matrix.
            b               : Wing semi-span [mm].
            enable_logging  : Toggle logger.
            verbose         : When True, log at DEBUG level.
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)

        self.n  = int(mesh.n)
        self.m  = int(mesh.m)
        self.nc = int(mesh.nc)

        self.dmatrix  = np.asarray(dmatrix,  dtype=float)   # (6, n, nc)
        self.u_limit  = float(U_MAX_FACTOR * b)
        self.th_limit = float(THETA_MAX)

        self._evaluate()

    # ----------------------------------------------------------------
    # Private - Margin evaluation
    # ----------------------------------------------------------------

    def _evaluate(self) -> None:
        '''
        Compare the evaluated displacement with the design limit,
        obtaining so the margin of safety per-node. Repeat this process
        for every load condition, based on 'dmatrix' tensor.

        Returns (self):
            DisplacementData : container for MS data and number of violations
                               used in fpenalty
        '''
        n  = self.n
        nc = self.nc

        # dmatrix rows: 0-2 -> translations [u, v, w], 3-5 -> rotations [p, q, r]
        MS_u  = np.full((3, n, nc), LARGE_DISPL_MS, dtype=float)
        MS_th = np.full((3, n, nc), LARGE_DISPL_MS, dtype=float)

        for k in range(3):
            u_abs  = np.abs(self.dmatrix[k,   :, :])   # (n, nc)
            th_abs = np.abs(self.dmatrix[k+3, :, :])   # (n, nc)

            mask_u  = u_abs  > TOL
            mask_th = th_abs > TOL

            MS_u[k][mask_u]   = self.u_limit  / u_abs[mask_u]  - 1.0
            MS_th[k][mask_th] = self.th_limit / th_abs[mask_th] - 1.0

        # Per-node minimum: collapse over DOF axis (0) and LC axis (2)
        MS_min_node = np.minimum(
            np.min(MS_u,  axis=(0, 2)),
            np.min(MS_th, axis=(0, 2)),
        )   # (n,)

        MS_min = float(np.min(MS_min_node))
        nv     = int(np.sum(MS_u < 0.0) + np.sum(MS_th < 0.0))

        u_max  = float(np.max(np.abs(self.dmatrix[0:3, :, :])))
        th_max = float(np.max(np.abs(self.dmatrix[3:6, :, :])))

        self.logger.debug(f"Displacement margins evaluated.\n"
            f"| u_max        : {u_max:.4f} mm\n"
            f"| th_max       : {np.degrees(th_max):.2f} deg\n"
            f"| u_limit      : {self.u_limit:.4f} mm\n"
            f"| th_limit     : {np.degrees(self.th_limit):.2f} deg\n"
            f"| MS_min       : {MS_min:.4f}\n"
            f"| nv           : {nv}"
        )
        if MS_min < 0.0:
            self.logger.warning(
                f"[CL3O] Displacement limit exceeded: negative margin "
                f"[MS_min={MS_min:.4f}, violations={nv}]."
            )

        self.data = DisplacementData(
            n           = n,
            m           = self.m,
            nc          = nc,
            MS_u        = MS_u,
            MS_th       = MS_th,
            MS_min_node = MS_min_node,
            MS_min      = MS_min,
            nv          = nv,
        )
