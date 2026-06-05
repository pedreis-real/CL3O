'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Objective Function Module.

Assembles the software main pipeline evaluation routine for obtaining a
scalar fitness for each design vector X:

    TotalScore(X) = mass_coef * mass(X) + p(X)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    WEIGHTING_FACTOR
)

# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.section_builder import SectionBuilder

# FEA
from cl3o.fea.solver.mesh_builder import MeshBuilder
from cl3o.fea.solver.static_analysis import LinearStaticSolver
from cl3o.fea.post.stress_recovery import StressRecovery
from cl3o.fea.post.tsw_failure import TsaiWuFailure
from cl3o.fea.post.displacement_ms import DisplacementMargins

# Optimization
from cl3o.optimization.fpenalty import Penalty, PenaltyData
from cl3o.optimization.fscore import StructuralMass, ScoreData
from cl3o.optimization.de_opt import OptVars


# ================================================================================
# Data persistence
# ================================================================================

# --------- Runtime --------
@dataclass
class RuntimeData:
    '''
    Container for all objects built and consumed during a single CL3O run.

    Unlike StaticData, these objects are fully redefined on every DE
    individual evaluation. BuildEvaluator writes through this container
    step-by-step inside the pipeline closure.

    Property    Description                                                 Units
    --------    --------------------------------------------------------    --------
    optvars     Per-population oprtimization variables (OptVars)            -
    sections    Per-station geometrical properties (SectionData)            -
    mesh        Global mesh, stiffness and connectivity (MeshData)          -
    fea_rts     FEA results from LinearStaticSolver (FeaResults)            -
    stress      Per-panel / per-boom stresses (StressData)                  MPa
    tsw         Tsai-Wu strength ratios and margins (FailureData)           -
    displ       Per-node displacement margins (DisplacementData)            -
    penalty     Penalty function breakdown (PenaltyData)                    -
    score       Per-element structural mass breakdown (ScoreData)           kg
    fitness     Scalar DE fitness and feasibility flag (FitnessData)        -
    '''
    optvars  : object = field(default=None)
    sections : object = field(default=None)
    mesh     : object = field(default=None)
    fea_rts  : object = field(default=None)
    stress   : object = field(default=None)
    tsw      : object = field(default=None)
    displ    : object = field(default=None)
    penalty  : object = field(default=None)
    score    : object = field(default=None)
    fitness  : object = field(default=None)


# --------- Scalar fitness --------
@dataclass
class FitnessData:
    '''
    Container for the DE scalar fitness and its breakdown.

    Per thesis Eq. 3.68 (Figura 50, page 90): z(X) = w_m * m(X) + P(X).

    Property        Size        Description                         Units
    ------------    --------    --------------------------------    --------
    score           float       Structural mass m(X)                kg
    penalty         float       Penalty term P(X)                   -
    total           float       Scalar fitness z(X)                 -
    is_feasible     bool        True when violation count is 0      -
    '''
    score       : float = 0.0
    penalty     : float = 0.0
    total       : float = 0.0
    is_feasible : bool  = False



# ========================================================================
# Internal helper - Design-vector validation
# ========================================================================

class FobjectiveHelper:
    '''
    Compilation of static checks used by BuildEvaluator to guarantee a
    well-formed design vector before the pipeline runs.
    '''

    @staticmethod
    def validate_design_vector(
        X             : np.ndarray,
        expected_size : int,
    ) -> None:
        '''
        Validate a flat DE design vector before decoding.

        Args:
            X            : Candidate design vector emitted by the DE loop.
            expected_size: Required length, computed from n_cpts.

        Raises:
            ValueError: if X is not a 1-D ndarray, has the wrong size,
                or contains non-finite entries.
        '''
        if not isinstance(X, np.ndarray) or X.ndim != 1:
            raise ValueError(
                f"[CL3O] Design vector must be a 1-D numpy ndarray.\n"
                f"| type : {type(X).__name__}\n"
                f"| ndim : {getattr(X, 'ndim', 'n/a')}"
            )
        if X.size != expected_size:
            raise ValueError(
                f"[CL3O] Design vector size mismatch.\n"
                f"| expected : {expected_size}\n"
                f"| got      : {X.size}\n"
                f"Check OptVars schema and DE bounds construction."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"[CL3O] Design vector contains non-finite entries "
                f"(NaN / +-inf). DE bounds or mutation step out of range."
            )



# ================================================================================
# EVALUATOR FACTORY API - Assembles CL3O main evaluation pipeline
# ================================================================================

class BuildEvaluator:
    '''
    Assemble the full CL3O core pipeline into a single callable stored at
    `self.eval_`. SetupOpt receives this callable and the DE loop invokes
    it once per candidate, evaluating it `NP * (1 + k_max)` times overall
    (initial population + NP trials per generation x k_max generations).

    Default use:
    >>> be      = BuildEvaluator(static_data=st)
    >>> fitness = be.eval_(X)
    >>> rt      = be.rt   # populated after each eval_ call

    ----------------------------------------------------------------
    CL3O pipeline assembled by the closure (10 steps):

    1.  Validate the flat design vector.
    2.  Decode X into the OptVars container.
    3.  Build the cross-section at every station.
    4.  Build the global mesh (beam local stiffness, global assembly).
    5.  Solve the linear static analysis ({F} = [K]{d}).
    6.  Recover stresses (normal for booms; shear for panels and webs).
    7.  Run Tsai-Wu failure assessment.
    8.  Compute displacement margins of safety.
    9.  Evaluate the penalty term P(X).
    10. Evaluate the structural mass m(X) and the scalar fitness z(X).
    '''

    def __init__(
        self,
        static_data,
        use_local_in_sr  : bool = True,
        use_offset       : bool = True,
        pipeline_logging : bool = False,
        enable_logging   : bool = True,
    ):
        '''
        Args:
            static_data     : StaticData container with wing_db, fem_setup,
                laminate_db and all upstream JSON-loaded artefacts.
            use_local_in_sr : Use local-frame forces in StressRecovery.
            use_offset      : Apply the shear-centre offset G matrix in
                BeamElement.
            pipeline_logging: When True, enables every pipeline sub-class
                logger at DEBUG level so a single evaluation narrates the
                full 10-step pipeline. Default False to keep the DE inner
                loop quiet (each step receives the shared null logger).
            enable_logging  : Toggle logger for BuildEvaluator itself.

        Notes:
            BuildEvaluator owns its own RuntimeData sink, exposed as
            `self.rt`. The DE inner loop writes through it on every
            eval_(X); downstream consumers (PostProcessing, plotters)
            should read from the same instance after the DE run.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        self.logger.info("Building CL3O Evaluator function.")

        self.st = static_data
        self.rt      : RuntimeData = RuntimeData()
        self.best_rt : RuntimeData = RuntimeData()
        self._best_f : float       = float('inf')
        self.n_cpts = int(self.st.wing_db.n_cpts)
        self.pipeline_logging = bool(pipeline_logging)

        self.use_local_in_sr = use_local_in_sr
        self.use_offset      = use_offset
        self.eval_ = self._assemble()

    # ------------------------------------------------
    # Private method - design vector decodification
    # ------------------------------------------------

    @staticmethod
    def encode_optvars(opt_vars: "OptVars") -> np.ndarray:
        '''
        Inverse of `_decode_design_vector`: pack an OptVars container
        back into the flat DE design vector. Uses the original control
        variables (xw1, xw2, bf*_root, tpr, ls/lw/lf*) — the derived
        bf1..bf4 arrays are regenerated by decode.
        '''
        return np.concatenate([
            np.asarray(opt_vars.xw1, dtype=float).ravel(),
            np.asarray(opt_vars.xw2, dtype=float).ravel(),
            np.array([
                opt_vars.bf1_root, opt_vars.bf2_root,
                opt_vars.bf3_root, opt_vars.bf4_root,
            ], dtype=float),
            np.asarray(opt_vars.tpr, dtype=float).ravel(),
            np.asarray(opt_vars.ls1, dtype=float).ravel(),
            np.asarray(opt_vars.ls2, dtype=float).ravel(),
            np.asarray(opt_vars.lw1, dtype=float).ravel(),
            np.asarray(opt_vars.lw2, dtype=float).ravel(),
            np.asarray(opt_vars.lf1, dtype=float).ravel(),
            np.asarray(opt_vars.lf2, dtype=float).ravel(),
            np.asarray(opt_vars.lf3, dtype=float).ravel(),
            np.asarray(opt_vars.lf4, dtype=float).ravel(),
        ])

    def _decode_design_vector(
        self,
        X : np.ndarray,
    ) -> OptVars:
        '''
        Split a flat X vector into the canonical OptVars container.

        Layout (cf. OptVars docstring at de_opt.py):
            xw1, xw2                          : 2 * n_cpts
            bf1_root..bf4_root                : 4 scalars
            tpr                               : n_cpts - 1
            ls1, ls2, lw1, lw2, lf1..lf4      : 8 * n_cpts
            ---------------------------------- total = 11 * n_cpts + 3
        '''
        n = self.n_cpts
        block_lengths = [
            n, n,            # xw1, xw2
            4,               # bf*_root scalars
            n - 1,           # tpr
            n, n, n, n,      # ls1, ls2, lw1, lw2
            n, n, n, n,      # lf1, lf2, lf3, lf4
        ]
        offsets = np.cumsum([0] + block_lengths)
        blocks  = [X[offsets[k]:offsets[k + 1]] for k in range(len(block_lengths))]
        (xw1, xw2, bf_roots, tpr,
         ls1, ls2, lw1, lw2,
         lf1, lf2, lf3, lf4) = blocks

        # Enforce xw1 < xw2 invariant per control point. DE bounds may overlap,
        # so swap chordwise positions (and their bound web layups) when violated.
        swap = xw2 < xw1
        if np.any(swap):
            xw1, xw2 = np.where(swap, xw2, xw1), np.where(swap, xw1, xw2)
            lw1, lw2 = np.where(swap, lw2, lw1), np.where(swap, lw1, lw2)

        bf1_root, bf2_root, bf3_root, bf4_root = (float(v) for v in bf_roots)
        bf1 = np.concatenate(([bf1_root], bf1_root * tpr))
        bf2 = np.concatenate(([bf2_root], bf2_root * tpr))
        bf3 = np.concatenate(([bf3_root], bf3_root * tpr))
        bf4 = np.concatenate(([bf4_root], bf4_root * tpr))

        return OptVars(
            xw1 = xw1, xw2 = xw2,
            bf1_root = bf1_root, bf2_root = bf2_root,
            bf3_root = bf3_root, bf4_root = bf4_root,
            tpr = tpr,
            bf1 = bf1, bf2 = bf2, bf3 = bf3, bf4 = bf4,
            ls1 = np.floor(ls1), ls2 = np.floor(ls2),
            lw1 = np.floor(lw1), lw2 = np.floor(lw2),
            lf1 = np.floor(lf1), lf2 = np.floor(lf2),
            lf3 = np.floor(lf3), lf4 = np.floor(lf4),
        )
    
    # ----------------------------------------
    # Private method - assemble CL3O pipeline
    # ----------------------------------------

    def _assemble(
        self,
    ) -> Callable[[np.ndarray], float]:
        '''
        Construct the DE evaluator closure.

        The closure maps a flat design vector X to its scalar fitness
        z(X) by chaining the 10 CL3O pipeline steps. Errors raised by
        any sub-step propagate untouched: a poorly conditioned [K] or
        a singular system signals that the DE bounds need review,
        not that the evaluator should silently mask the failure.

        Returns:
            Callable eval_(X) -> float.
        '''
        expected_X_size = 11 * self.n_cpts + 3

        def eval_(X: np.ndarray) -> float:
            FobjectiveHelper.validate_design_vector(X, expected_X_size)  # 1
            self._step_decode(X)                                         # 2
            self._step_cross_section()                                   # 3
            self._step_mesh()                                            # 4
            self._step_static_solve()                                    # 5
            self._step_stress_recovery()                                 # 6
            self._step_failure()                                         # 7
            self._step_displacement()                                    # 8
            self._step_penalty()                                         # 9
            self._step_mass_and_fitness()                                # 10
            return self._record_best()

        return eval_

    # ------------------------------------------------
    # Private methods - individual pipeline steps
    # ------------------------------------------------

    def _step_decode(self, X: np.ndarray) -> None:
        '''Step 2: decode the flat design vector X into OptVars.'''
        self.rt.optvars = self._decode_design_vector(X=X)

    def _step_cross_section(self) -> None:
        '''Step 3: build the cross-section geometry at every station.'''
        log = self.pipeline_logging
        sec = SectionBuilder(
            opt_vars       = self.rt.optvars,
            static_data    = self.st,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.sections = sec.data

    def _step_mesh(self) -> None:
        '''Step 4: assemble the global mesh and stiffness matrix.'''
        log = self.pipeline_logging
        mesh = MeshBuilder(
            data           = (self.st.fem_setup, self.rt.sections),
            use_offset     = self.use_offset,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.mesh = mesh.data

    def _step_static_solve(self) -> None:
        '''Step 5: solve the linear static analysis ({F} = [K]{d}).'''
        log = self.pipeline_logging
        static_analysis = LinearStaticSolver(
            mesh           = self.rt.mesh,
            loads          = self.st.fem_setup.loads,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.fea_rts = static_analysis.results

    def _step_stress_recovery(self) -> None:
        '''Step 6: recover boom normal and panel/web shear stresses.'''
        log = self.pipeline_logging
        stress = StressRecovery(
            sections        = self.rt.sections,
            element_idx     = self.rt.mesh.conn[:, :2],
            fea_results     = self.rt.fea_rts,
            use_local_in_sr = self.use_local_in_sr,
            enable_logging  = log,
            verbose         = log,
        )
        self.rt.stress = stress.data

    def _step_failure(self) -> None:
        '''Step 7: run the Tsai-Wu failure assessment.'''
        log = self.pipeline_logging
        tsw = TsaiWuFailure(
            data           = (self.st, self.rt),
            enable_logging = log,
            verbose        = log,
        )
        self.rt.tsw = tsw.data

    def _step_displacement(self) -> None:
        '''Step 8: compute displacement margins of safety.'''
        log = self.pipeline_logging
        disp = DisplacementMargins(
            mesh           = self.rt.mesh,
            dmatrix        = self.rt.fea_rts.dmatrix,
            b              = self.st.wing_db.b,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.displ = disp.data

    def _step_penalty(self) -> None:
        '''Step 9: evaluate the penalty term P(X).'''
        log = self.pipeline_logging
        penalty = Penalty(
            data           = (self.rt.tsw, self.rt.displ),
            enable_logging = log,
            verbose        = log,
        )
        self.rt.penalty = penalty.data

    def _step_mass_and_fitness(self) -> None:
        '''Step 10: structural mass m(X) and scalar fitness z(X).'''
        log = self.pipeline_logging
        score = StructuralMass(
            sections       = self.rt.sections,
            element_idx    = self.rt.mesh.conn[:, :2],
            laminate_db    = self.st.laminate_db,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.score = score.data

        fitness = TotalScore(
            mass_data      = score.data,
            penalty_data   = self.rt.penalty,
            enable_logging = log,
            verbose        = log,
        )
        self.rt.fitness = fitness.data

    def _record_best(self) -> float:
        '''Return the scalar fitness z(X), snapshotting the best candidate.'''
        total = float(self.rt.fitness.total)
        if total < self._best_f:
            self._best_f = total
            self.best_rt = copy.copy(self.rt)
        return total


# ================================================================================
# PUBLIC API - Total score evaluator
# ================================================================================

class TotalScore:
    '''
    Evaluate the DE objective z(X) from the mass and penalty terms of
    a single candidate (thesis Eq. 3.68 / Figura 50, page 90):

        z(X) = w_m * m(X) + P(X)

    where w_m is the mass weighting factor (consts.DFLT_MASS_COEF),
    m(X) is the structural mass in kg, and P(X) is the scalar penalty.

    Use:
        ts = TotalScore(mass_data, penalty_data)
        f  = ts.data.total
    '''

    def __init__(
        self,
        mass_data      : ScoreData,
        penalty_data   : PenaltyData,
        wm      : float = WEIGHTING_FACTOR,
        enable_logging : bool  = True,
        verbose        : bool  = False,
    ) -> None:
        '''
        Args:
            mass_data     : ScoreData from StructuralMass.
            penalty_data  : PenaltyData from Penalty.
            mass_coef     : Weighting factor for the mass term.
            enable_logging: Toggle logger.
            verbose       : When True, log at DEBUG level.
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)

        mX = float(mass_data.total)
        pX = float(penalty_data.total)
        wm = float(wm)
        fX = wm * mX + pX

        self.logger.debug(
            f"TotalScore evaluated.\n"
            f"| mass      : {mX:.4f} kg\n"
            f"| penalty   : {pX:.4f} kg\n"
            f"| mass_coef : {wm}\n"
            f"| total     : {fX:.4f}"
        )

        # -------- Pack results --------
        self.data = FitnessData(
            score       = mX,
            penalty     = pX,
            total       = fX,
            is_feasible = bool(penalty_data.is_feasible),
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
