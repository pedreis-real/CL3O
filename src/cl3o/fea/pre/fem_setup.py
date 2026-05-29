'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Pre-processing module.

Setup default mesh paremeters (static), based on external loads and wing
geometry.

It is not considered a prescribed nodal displacement, neither a distributed
load. Only nodal forces, from external loads database, are applied in the
linear static Finite Element Analysis.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

from dataclasses import dataclass, field

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Const
from cl3o import Constants

# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.wing import LerpWingData

# Finite Element Analysis
from cl3o.fea.loads.load_mapper import ExLoadsData, LoadsHelper

# ================================================================================
# Data persistence - Global FEM setup container
# ================================================================================

@dataclass
class FemPreprocessData:
    '''
    Static artifacts of the Finite Element Analysis.

    Property    Size        Description                         Units
    --------    --------    --------------------------------    --------
    n           (1,)        Number of nodes                     -
    m           (1,)        Number of element                   -
    dof         (1,)        Total number of DOFs                -
    re          (n, 6)      Restraint matrix (1=DOF fixed)      -
    re_flat     (6n,)       Restraint vector (flattened)        -
    conn        (m, 4)      Connectivity and release            -
    mcn         (m, 12)     Member code numbers                 -
    loads       dict        Loads dictionary with keys:
    --------
        nc          (1,)        Number of load conditions       -
        F           (n,6,nc)    Nodal force                     N, N*mm
        F_flat      (6n,nc)     Flattened nodal force           N, N*mm
    '''
    n       : int = 0
    m       : int = 0
    dof     : int = 0
    re      : np.ndarray = field(default_factory=lambda: np.zeros((0,6)))
    re_flat : np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    conn    : np.ndarray = field(default_factory=lambda: np.zeros((0,4)))
    mcn     : np.ndarray = field(default_factory=lambda: np.zeros((0,12)))

    loads : dict = None

    # Per-element BeamData+T-matrix cache shared across all DE evaluations.
    # Key: (id(geomA), id(geomB), release_code) — safe because GeomData objects
    # in StaticData.geom_cache are never evicted (plain dict, no LRU).
    beam_cache : dict = field(default_factory=dict)



# ================================================================================
# PUBLIC API - FEA Pre processing module
# ================================================================================

class FemSetup:

    def __init__(
        self,
        exloads_db : ExLoadsData,
        lerp_wing_db : LerpWingData,
        wing_side : str = Constants.WING_SIDE,
        enable_logging : bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)
        self.logger.info("Constructing default mesh paremeters...")

        self.exloads_data = exloads_db
        self.wng_data = lerp_wing_db
        self.wing_side = wing_side

        self._build_default_mesh()
        self._retrieve_loads()

        self.fem_setup = self._pack_fem_setup()
    
    # ----------------------------------------
    # Private method - Build up orchestrator
    # ----------------------------------------

    def _build_default_mesh(self) -> None:
        '''Carries default values of mesh topology into 'self' state.'''
        self.n = self.wng_data.n_sta
        self.m = self.n - 1
        self.dof = 6 * self.n

        self.re = np.zeros((self.n, 6), dtype=int)
        self.re[0,:] = int(1)
        self.conn = np.column_stack([
            np.arange(0, self.m, dtype=int),
            np.arange(1, self.n, dtype=int),
            np.ones((self.m,2), dtype=int),
        ])
        self.mcn = np.hstack([
            6 * self.conn[:, 0][:, None] + np.arange(6),
            6 * self.conn[:, 1][:, None] + np.arange(6)
        ]).astype(int)

    def _retrieve_loads(self):
        '''
        Compiles the external loads database and settlement
        displacement into 'self' state.
        '''
        self.nc = self.exloads_data.num_cond
        self.f_nodal = np.zeros((self.n, 6, self.nc), dtype=float)

        # Re-slice the half-span loads from the full-span arrays for the active
        # wing side, so the nodal forces always match the analyzed mesh
        # regardless of which side the persisted '_hf' arrays were built for.
        _, _, _, lift_hf, drag_hf, moment_hf = LoadsHelper.half_span_slice(
            self.exloads_data.X, self.exloads_data.Y, self.exloads_data.Z,
            self.exloads_data.lift, self.exloads_data.drag,
            self.exloads_data.moment,
            wing_side=self.wing_side,
        )

        def _as_n_nc(raw):
            a = np.asarray(raw, dtype=float)
            if a.ndim == 1:
                a = a.reshape(self.n, self.nc)
            elif a.shape == (self.nc, self.n):
                a = a.T
            return a

        self.f_nodal[:, 0, :] = _as_n_nc(drag_hf)
        self.f_nodal[:, 2, :] = _as_n_nc(lift_hf)
        self.f_nodal[:, 4, :] = _as_n_nc(moment_hf)
    
    def _pack_fem_setup(self) -> FemPreprocessData:
        '''Pack the FEM setup tensors.'''
        return FemPreprocessData(
            n  = int(self.n),
            m  = int(self.m),
            dof = int(self.dof),
            re = self.re.astype(int),
            re_flat = self.re.ravel().astype(int),
            conn = self.conn.astype(int),
            mcn = self.mcn.astype(int),
            loads = {
                'nc' : int(self.nc),
                'F'  : self.f_nodal.astype(float),
                'F_flat' : self.f_nodal.reshape(6*self.n, self.nc).astype(float),
            },
        )
