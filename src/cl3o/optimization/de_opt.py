'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Differential Evolution Module.

Heuristic optimization routine for minimizing structural mass, based on a
objective function defined by the sum of a score (mass, in kg) and a penalty
(logistic curve based on M.S < 0). Adopting the Differential Evolution
conceptions with the following aspects:

-   DE-2 mutation vector algorithm (DE/current-to-best/1)
    v_i = x_r1 + F * (x_r2 - x_r3) + lambda * (x_best - x_i)
    where r1, r2, r3 are three distinct population indices different from i.
-   Exponential crossover with probability CR (Eq. 2.102-2.104)
    u_j = {v_j,     for j = <n>_D, <n+1>_D, ..., <n+L-1>_D
          {x_ij,    otherwise; where P(L=nu) = CR^nu (Bernoulli run)
-   Greedy selection, keeping the better of (x_i, u_i) per generation.

Entry points:
    BuildEvaluator - builds an evaluator callable that turns
        a design vector X into a (1,)   fitness.
    SetupOpt - builds the initial population, stores bounds + DE
        hyper-parameters and caches the evaluator callable.
    RunOpt - executes the outer DE loop for, at most, k_max generations
        and records the per-generation best and population statistics.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import logging
import pickle
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    DE_HYPERPAR, OPT_LIMS, TOL, STALL_REL_TOL,
)

# Utilities
from cl3o.utils import io_utils as io


# ================================================================================
# Data persistence - DE configuration and history
# ================================================================================

@dataclass
class OptData:
    '''
    DE setup snapshot: bounds, hyper-parameters, initial population.

    Property    Size        Description
    --------    --------    ----------------------------------------
    lo          (D,)        Lower bound per design variable     
    hi          (D,)        Upper bound per design variable     
    NP          (1,)        Population size                     
    CR          (1,)        Crossover probability [0, 1]        
    F           (1,)        Differential weight                 
    lmbda       (1,)        Best-attraction weight              
    k_max       (1,)        Maximum number of generations       
    seed        (1,)        RNG seed for reproducibility        
    seed        (1,)        RNG seed for reproducibility        
    D           (1,)        Number of design variables          
    X0          (NP, D)     Initial population (seeded LHS-like)
    '''
    lo             : np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi             : np.ndarray = field(default_factory=lambda: np.zeros(0))
    NP             : int        = DE_HYPERPAR['NP']
    CR             : float      = DE_HYPERPAR['CR']
    F              : float      = DE_HYPERPAR['F']
    lmbda          : float      = DE_HYPERPAR['lambda']
    k_max          : int        = DE_HYPERPAR['k_max']
    seed           : int        = DE_HYPERPAR['seed']
    tol            : float      = DE_HYPERPAR['std_tol']
    stall_patience : int        = DE_HYPERPAR['stall_patience']
    D              : int        = 0
    X0             : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class HistoryData:
    '''
    Per-generation history of the DE run.

    Property        Size            Description
    ------------    ------------    -----------------------------------
    ng              (1,)            Number of generations executed
    D               (1,)            Design-space dimension        
    best_X          (ng + 1, D)     Best design at each generation     
    best_f          (ng + 1,)       Fitness of best design at each gen.
    mean_f          (ng + 1,)       Population mean fitness            
    std_f           (ng + 1,)       Population fitness std             
    feasible_X      (D,)            Best feasible design vector        
    feasible_f      (1,)            Fitness of best feasible design    
    '''
    ng         : int        = 0
    D          : int        = 0
    best_X     : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    best_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    std_f      : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_X : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_f : float      = float('inf')


@dataclass
class OptVars:
    '''
    Container exposing per-cpt design-variables.

    The DE loop produces a flat design vector X; this container
    de-serialises it into the six continuous and eight discrete
    per-cpt arrays that SectionBuilder expects.

    Property    Size        Description                                 Units
    --------    --------    ----------------------------------------    --------
    xw1         (ncpt,)     Rear wing spar position                     - %c
    xw2         (ncpt,)     Aft  wing spar position                     - %c
    bf1_root    (1,)        Upper rear flange width at root             - %c
    bf2_root    (1,)        Lower rear flange width at root             - %c
    bf3_root    (1,)        Upper aft  flange width at root             - %c
    bf4_root    (1,)        Lower aft  flange width at root             - %c
    tpr         (ncpt-1,)   Flange taper relative to root station       - 0-1
    bf1         (ncpt-1,)   Upper rear flange width per non-root CP     - %c
    bf2         (ncpt-1,)   Lower rear flange width per non-root CP     - %c
    bf3         (ncpt-1,)   Upper aft  flange width per non-root CP     - %c
    bf4         (ncpt-1,)   Lower aft  flange width per non-root CP     - %c
    ls1         (ncpt,)     Skin layup, from LE up to xw1               - index
    ls2         (ncpt,)     Skin layup, from xw1 up to TE               - index
    lw1         (ncpt,)     Rear wing web layup                         - index
    lw2         (ncpt,)     Aft  wing web layup                         - index
    lf1         (ncpt,)     Upper rear flange layup                     - index
    lf2         (ncpt,)     Lower rear flange layup                     - index
    lf3         (ncpt,)     Upper aft  flange layup                     - index
    lf4         (ncpt,)     Lower aft  flange layup                     - index
    '''
    xw1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    xw2 : np.ndarray = field(default_factory=lambda: np.zeros(0))

    bf1_root : float      = 0.0
    bf2_root : float      = 0.0
    bf3_root : float      = 0.0
    bf4_root : float      = 0.0
    tpr      : np.ndarray = field(default_factory=lambda: np.zeros(0))

    bf1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf3 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf4 : np.ndarray = field(default_factory=lambda: np.zeros(0))

    ls1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    ls2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lw1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lw2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf3 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf4 : np.ndarray = field(default_factory=lambda: np.zeros(0))



# ================================================================================
# Internal helper - DE math
# ================================================================================

class OptHelper:
    '''Compilation of static validators used by SetupOpt before a DE run.'''

    @staticmethod
    def validate_bounds(lo: np.ndarray, hi: np.ndarray) -> None:
        if lo.size != hi.size or lo.size == 0:
            raise ValueError(
                f"[CL3O] bounds_lo and bounds_hi must share non-zero length.\n"
                f"| lo.size : {lo.size}\n"
                f"| hi.size : {hi.size}"
            )
        if np.any(hi <= lo):
            raise ValueError(
                "[CL3O] bounds_hi must be strictly greater than bounds_lo "
                "for every design variable."
            )

    @staticmethod
    def validate_hyperpar(NP: int, CR: float) -> None:
        if not (0.0 <= CR <= 1.0):
            raise ValueError(f"[CL3O] CR must be in [0, 1], got {CR}.")
        if NP < 4:
            raise ValueError(
                f"[CL3O] NP must be >= 4 (DE needs r1, r2, r3 distinct "
                f"from i), got {NP}."
            )


# ========================================================================
# Internal helper - Per-generation RuntimeData archiver
# ========================================================================

class GenerationArchiver:
    '''
    Persist one RuntimeData snapshot per DE generation, plus a manifest
    summarising the whole run, intended for offline PySide6 UI consumption.

    Disk layout under `out_dir`:
        manifest.json
        generations/
            gen_0000.pkl
            gen_0001.pkl
            ...

    Each pickle file contains the live RuntimeData returned by the
    evaluator's snapshot callable, serialised at protocol HIGHEST_PROTOCOL.
    The manifest is a JSON document with a top-level schema_version key so
    the UI can branch on archive layout changes without re-running DE.
    '''

    SCHEMA_VERSION = "1.1"

    def __init__(
        self,
        out_dir        : str | Path,
        runtime_data   : object,
        enable_logging : bool = True,
    ) -> None:
        '''
        Args:
            out_dir       : Destination directory; a `generations/`
                sub-directory is created automatically.
            runtime_data  : Live RuntimeData object whose fields are
                already populated by the evaluator before each call to
                archive(). The archiver pickles the object's current
                state in-place — no re-evaluation is done here.
            enable_logging: Toggle logger.
        '''
        self.logger    = io.setup_logger(self, enable_logging)
        self.out_dir   = Path(out_dir)
        self.gens_dir  = self.out_dir / "generations"
        self.gens_dir.mkdir(parents=True, exist_ok=True)

        self.runtime_data = runtime_data
        self.records      : list[dict] = []
        # Dedup state: hash map from rounded design-vector tuple to its index
        # in distinct_records. O(1) lookup replaces the O(k) linear scan.
        self._archived_X_map  : dict[tuple, int] = {}
        self.distinct_records : list[dict]       = []
        self.created_at   = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    # ------------------------------------------------
    # Public methods - per-generation snapshot
    # ------------------------------------------------

    def archive(
        self,
        k : int,
        X : np.ndarray | None = None,
    ) -> None:
        '''
        Pickle the current state of `self.runtime_data` to gen_<k>.pkl
        and append a scalar summary to self.records.

        If `X` is supplied and matches a previously archived design
        vector within DEDUP_TOL (euclidean norm), the pickle write is
        skipped; the new record points to the original file and carries
        `is_duplicate=True` plus `first_seen_gen`. This lets the UI keep
        a linear per-generation timeline without bloating disk usage.

        The caller (RunOpt._archive_generation) is responsible for
        re-evaluating the best individual before calling this, so that
        runtime_data already reflects the best design of generation k.

        Args:
            k: Generation index, zero-based.
            X: Optional flat design vector of the best individual being
               archived. Required for dedup; omit to force a write.
        '''
        try:
            dup_idx = self._find_duplicate(X)
            if dup_idx is not None:
                orig = self.distinct_records[dup_idx]
                rec  = self._summarize(k, orig["file"], self.runtime_data)
                rec["is_duplicate"]   = True
                rec["first_seen_gen"] = int(orig["k"])
                self.records.append(rec)
                return

            fpath = self.gens_dir / f"gen_{k:04d}.pkl"
            with open(fpath, "wb") as f:
                pickle.dump(self.runtime_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            rec = self._summarize(k, fpath.name, self.runtime_data)
            rec["is_duplicate"]   = False
            rec["first_seen_gen"] = int(k)
            self.records.append(rec)
            self.distinct_records.append(rec)
            if X is not None:
                key = tuple(np.round(np.asarray(X, dtype=float).ravel(), 8))
                self._archived_X_map[key] = len(self.distinct_records) - 1
        except Exception as exc:
            self.logger.warning(
                f"[CL3O] Snapshot archive failed at gen {k}.\n"
                f"| reason : {exc!r}"
            )

    def _find_duplicate(
        self,
        X : np.ndarray | None,
    ) -> int | None:
        '''Return index into distinct_records of a previously archived X, or None.'''
        if X is None or not self._archived_X_map:
            return None
        key = tuple(np.round(np.asarray(X, dtype=float).ravel(), 8))
        return self._archived_X_map.get(key, None)

    # ------------------------------------------------
    # Public methods - final manifest
    # ------------------------------------------------

    def write_manifest(
        self,
        history   : "HistoryData",
        opt_data  : "OptData",
        run_label : str = "",
    ) -> Path:
        '''
        Persist `manifest.json` aggregating run metadata, full history
        arrays, and the per-generation snapshot index.

        Returns:
            Full path to the written manifest file.
        '''
        manifest = {
            "schema_version" : self.SCHEMA_VERSION,
            "run_label"      : str(run_label),
            "created_at"     : self.created_at,
            "n_gens"         : int(history.ng),
            "D"              : int(history.D),
            "NP"             : int(opt_data.NP),
            "seed"           : int(opt_data.seed),
            "best_f_hist"    : history.best_f.tolist(),
            "mean_f_hist"    : history.mean_f.tolist(),
            "std_f_hist"     : history.std_f .tolist(),
            "feasible_f"     : float(history.feasible_f),
            "snapshots"      : self.records,
            "distinct_individuals" : self.distinct_records,
        }
        path = self.out_dir / "manifest.json"
        io.write_json(manifest, path)
        self.logger.info(
            f"DE archive written.\n"
            f"| dir       : {self.out_dir}\n"
            f"| snapshots : {len(self.records)}"
        )
        return path

    # ------------------------------------------------
    # Private methods - manifest helpers
    # ------------------------------------------------

    @staticmethod
    def _summarize(
        k        : int,
        filename : str,
        snap     : object,
    ) -> dict:
        '''
        Pluck scalar fitness fields from the snapshot for the manifest
        index. Uses getattr so this remains decoupled from the exact
        RuntimeData / FitnessData layout in fobjective.py.
        '''
        fit = getattr(snap, "fitness", None)
        return {
            "k"           : int(k),
            "file"        : filename,
            "best_f"      : float(getattr(fit, "total",       float('nan'))),
            "mass"        : float(getattr(fit, "score",       float('nan'))),
            "penalty"     : float(getattr(fit, "penalty",     float('nan'))),
            "is_feasible" : bool (getattr(fit, "is_feasible", False)),
        }


# ================================================================================
# PUBLIC API - DE setup
# ================================================================================

class SetupOpt:
    '''
    Prepare a DE run: bounds, hyper-parameters, initial population, and
    the evaluator callable that maps a design vector to a (1,)   fitness.

    Use:
        setup = SetupOpt(evaluator, bounds_lo=lo, bounds_hi=hi)
        opt   = setup.data                           # OptData
    '''

    def __init__(
        self,
        evaluator      : Callable[[np.ndarray], float],
        de_hyperpar    : dict = DE_HYPERPAR,
        bounds_lo      : np.ndarray | None = None,
        bounds_hi      : np.ndarray | None = None,
        enable_logging : bool = True,
        verbose        : bool = False,
    ) -> None:
        '''
        Args:
            evaluator     : Callable X -> fitness (float). The caller
                is responsible for wiring the full FEM + penalty +
                score pipeline. Lower value is better.
            de_hyperpar   : DE hyper-parameters dict. Defaults to
                DFLT_DE_HYPERPAR from Constants. Expected keys:
                NP, CR, F, lam, k_max, seed.
            bounds_lo     : (D,) lower bound per design variable.
            bounds_hi     : (D,) upper bound per design variable.
            enable_logging: Toggle logger.
            verbose       : When True, log at DEBUG level.
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)

        NP    = int  (de_hyperpar.get('NP',       DE_HYPERPAR['NP'      ]))
        CR    = float(de_hyperpar.get('CR',       DE_HYPERPAR['CR'      ]))
        F     = float(de_hyperpar.get('F',        DE_HYPERPAR['F'       ]))
        lmbda = float(de_hyperpar.get('lambda',   DE_HYPERPAR['lambda'  ]))
        k_max = int  (de_hyperpar.get('k_max',    DE_HYPERPAR['k_max'   ]))
        seed  = int  (de_hyperpar.get('seed',     DE_HYPERPAR['seed'    ]))
        tol   = float(de_hyperpar.get('std_tol',  DE_HYPERPAR['std_tol' ]))
        stall = int  (de_hyperpar.get(
            'stall_patience', DE_HYPERPAR['stall_patience']
        ))

        # -------- 1. Validate inputs --------
        if bounds_lo is None or bounds_hi is None:
            raise ValueError(
                "[CL3O] bounds_lo and bounds_hi must be supplied to SetupOpt.\n"
                "Call SetupOpt._build_de_bounds(n_cpts, n_mats) to obtain them."
            )
        lo = np.asarray(bounds_lo, dtype=float).ravel()
        hi = np.asarray(bounds_hi, dtype=float).ravel()
        OptHelper.validate_bounds(lo, hi)
        OptHelper.validate_hyperpar(NP, CR)

        D = int(lo.size)
        self.evaluator = evaluator
        self.rng       = np.random.default_rng(int(seed))

        # -------- 2. Seed initial population --------
        X0 = self._initial_pop(NP, lo, hi, self.rng)

        self.logger.info(
            f"DE setup ready.\n"
            f"| D      : {D}\t"
            f"| NP     : {NP}\t"
            f"| CR     : {CR}\t"
            f"| F      : {F}\t"
            f"| lambda : {lmbda}\t"
            f"| k_max  : {k_max}\t"
            f"| seed   : {seed}\t"
            f"| tol    : {tol}"
        )

        # -------- 3. Pack --------
        self.data = OptData(
            lo = lo,
            hi = hi,
            NP        = int(NP),
            CR        = float(CR),
            F         = float(F),
            lmbda     = float(lmbda),
            k_max          = int(k_max),
            seed           = int(seed),
            tol            = float(tol),
            stall_patience = int(stall),
            D              = D,
            X0             = X0,
        )

    # ----------------------------------------------------------------
    # Static - DE bounds from problem geometry
    # ----------------------------------------------------------------

    @staticmethod
    def _build_de_bounds(
        n_cpts : int,
        n_mats : int,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Construct flat DE bounds matching the OptVars layout decoded in
        fobjective._decode_design_vector (total = 11 * n_cpts + 3):

            xw1         : n_cpts  in [OPT_LIMS['xw1']]
            xw2         : n_cpts  in [OPT_LIMS['xw2']]
            bf*_root    : 4 scalars in [OPT_LIMS['bfk']] mm
            tpr         : n_cpts-1 in [OPT_LIMS['fl_tpr']]
            ls1, ls2    : 2 * n_cpts in [OPT_LIMS['layup_skin']]   (0-based)
            lw1, lw2    : 2 * n_cpts in [OPT_LIMS['layup_web']]    (0-based)
            lf1..lf4    : 4 * n_cpts in [OPT_LIMS['layup_flange']] (0-based)

        Layup indices are 0-based: index k selects LAYUP_ORDER[k] (MAT{k+1}).
        Ceilings are clipped to n_mats-1 so DE never samples a missing entry.
        '''
        xw1_lo, xw1_hi = float(OPT_LIMS['xw1'   ][0]), float(OPT_LIMS['xw1'   ][1])
        xw2_lo, xw2_hi = float(OPT_LIMS['xw2'   ][0]), float(OPT_LIMS['xw2'   ][1])
        tpr_lo, tpr_hi = float(OPT_LIMS['fl_tpr'][0]), float(OPT_LIMS['fl_tpr'][1])
        bf_lo,  bf_hi  = float(OPT_LIMS['bfk'   ][0]), float(OPT_LIMS['bfk'   ][1])
        # Layup indices are 0-based; clip hi so DE never samples a missing MAT.
        max_idx = int(n_mats) - 1
        sk_lo = int(OPT_LIMS['layup_skin'  ][0])
        sk_hi = min(int(OPT_LIMS['layup_skin'  ][1]), max_idx)
        wk_lo = int(OPT_LIMS['layup_web'   ][0])
        wk_hi = min(int(OPT_LIMS['layup_web'   ][1]), max_idx)
        fl_lo = int(OPT_LIMS['layup_flange'][0])
        fl_hi = min(int(OPT_LIMS['layup_flange'][1]), max_idx)

        lo_blocks = [
            np.full(n_cpts,     xw1_lo),
            np.full(n_cpts,     xw2_lo),
            np.array([bf_lo, bf_lo, bf_lo, bf_lo]),
            np.full(n_cpts - 1, tpr_lo),
            np.full(n_cpts, sk_lo),   # ls1
            np.full(n_cpts, sk_lo),   # ls2
            np.full(n_cpts, wk_lo),   # lw1
            np.full(n_cpts, wk_lo),   # lw2
            np.full(n_cpts, fl_lo),   # lf1
            np.full(n_cpts, fl_lo),   # lf2
            np.full(n_cpts, fl_lo),   # lf3
            np.full(n_cpts, fl_lo),   # lf4
        ]
        hi_blocks = [
            np.full(n_cpts,     xw1_hi),
            np.full(n_cpts,     xw2_hi),
            np.array([bf_hi, bf_hi, bf_hi, bf_hi]),
            np.full(n_cpts - 1, tpr_hi),
            np.full(n_cpts, sk_hi),   # ls1
            np.full(n_cpts, sk_hi),   # ls2
            np.full(n_cpts, wk_hi),   # lw1
            np.full(n_cpts, wk_hi),   # lw2
            np.full(n_cpts, fl_hi),   # lf1
            np.full(n_cpts, fl_hi),   # lf2
            np.full(n_cpts, fl_hi),   # lf3
            np.full(n_cpts, fl_hi),   # lf4
        ]

        lo = np.concatenate(lo_blocks)
        hi = np.concatenate(hi_blocks)
        hi = np.where(hi > lo, hi, lo + TOL)
        return lo, hi

    # ----------------------------------------------------------------
    # Private - Population seeding
    # ----------------------------------------------------------------

    @staticmethod
    def _initial_pop(
        NP        : int,
        bounds_lo : np.ndarray,
        bounds_hi : np.ndarray,
        rng       : np.random.Generator,
    ) -> np.ndarray:
        '''
        Initial population of shape (NP, D) via stratified Latin Hypercube
        Sampling (LHS). Each column holds NP samples drawn one-per-strip
        and shuffled independently, giving uniform per-variable coverage
        without the clumping of plain uniform sampling.
        '''
        lo  = np.asarray(bounds_lo, dtype=float).ravel()
        hi  = np.asarray(bounds_hi, dtype=float).ravel()
        D   = lo.size
        
        cut = np.linspace(0.0, 1.0, NP + 1)
        u   = rng.uniform(size=(NP, D))
        a   = cut[:-1][:, None]
        b   = cut[ 1:][:, None]
        
        pts = a + u * (b - a)
        
        for j in range(D):
            rng.shuffle(pts[:, j])
        
        return lo + pts * (hi - lo)


# ================================================================================
# PUBLIC API - DE run loop
# ================================================================================

class RunOpt:
    '''
    Execute the DE-3 outer loop and record a per-generation history.

    Use:
        run  = RunOpt(setup)
        hist = run.history                           # HistoryData
    '''

    def __init__(
        self,
        setup          : SetupOpt,
        feasible_check : Callable[[np.ndarray], bool] | None = None,
        on_generation  : Callable[[int, HistoryData], None] | None = None,
        runtime_data   : object | None = None,
        out_dir        : str | Path | None = None,
        run_label      : str = "",
        enable_logging : bool = True,
        verbose        : bool = False,
    ) -> None:
        '''
        Args:
            setup         : SetupOpt instance containing bounds,
                hyper-parameters, population, and evaluator.
            feasible_check: Optional callable X -> bool. When provided,
                RunOpt also tracks the best *feasible* design seen
                (useful when the evaluator returns a penalized fitness
                but we also want the best feasible design for export).
            on_generation : Optional callable (k, snapshot) invoked after
                each generation (including k=0). snapshot is a
                HistoryData trimmed to the range [0, k] - suitable for
                live plotting.
            runtime_data  : Optional live RuntimeData object populated by
                BuildEvaluator.eval_. When supplied together with
                `out_dir`, RunOpt re-evaluates the best individual of
                each generation (to ensure runtime_data reflects it),
                then pickles it to disk via GenerationArchiver.
            out_dir       : Optional destination directory for the
                per-generation pickle archive and the run manifest.
                Required for runtime_data to take effect.
            run_label     : Free-form label written into manifest.json
                (typically `<aircraft>_<opt>`).
            enable_logging: Toggle logger.
            verbose       : When True, log at DEBUG level (per-generation
                diagnostics, which re-evaluate the best individual).
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)
        self.setup          = setup
        self.feasible_check = feasible_check
        self.on_generation  = on_generation
        self.tol            = float(setup.data.tol)
        self.stall_patience = int(setup.data.stall_patience)
        self.run_label      = str(run_label)
        self.history        = HistoryData()
        self.runtime_data   = runtime_data

        self.archiver = (
            GenerationArchiver(
                out_dir        = out_dir,
                runtime_data   = runtime_data,
                enable_logging = enable_logging,
            )
            if (runtime_data is not None and out_dir is not None)
            else None
        )

        self._optimization_runtime_environment()

    # ----------------------------------------------------------------
    # Private - Outer loop
    # ----------------------------------------------------------------

    def _optimization_runtime_environment(self) -> None:
        '''Run k_max DE generations, updating population in place.'''
        opt   = self.setup.data
        rng   = self.setup.rng
        eval_ = self.setup.evaluator

        lo, hi = opt.lo, opt.hi
        D      = int(opt.D)
        NP     = int(opt.NP)
        k_max  = int(opt.k_max)

        self.logger.info(
            f"Starting DE-2 loop | NP={NP} | D={D} | k_max={k_max}"
        )

        # -------- 1. Initial population and fitness --------
        X = opt.X0.copy()
        f = np.array([float(eval_(X[i])) for i in range(NP)])

        best_X_hist = np.zeros((k_max + 1, D))
        best_f_hist = np.zeros(k_max + 1)
        mean_f_hist = np.zeros(k_max + 1)
        std_f_hist  = np.zeros(k_max + 1)

        feas_best_X = np.zeros(D)
        feas_best_f = float('inf')

        self._record_generation(
            0, X, f, best_X_hist, best_f_hist, mean_f_hist, std_f_hist,
        )
        if self.feasible_check is not None:
            feas_best_X, feas_best_f = self._update_best_feasible(
                X, f, feas_best_X, feas_best_f,
            )

        self._emit_snapshot(
            0, D,
            best_X_hist, best_f_hist, mean_f_hist, std_f_hist,
            feas_best_X, feas_best_f,
        )

        self._archive_generation(0, X, f)

        # -------- 2. DE generations --------
        k_last = 0
        for k in range(1, k_max + 1):
            best_i   = int(np.argmin(f))
            last_eval = NP - 1

            for i in range(NP):
                # Randomization
                r1, r2, r3 = self._pick_three_distinct(rng, NP, i)

                # Mutation
                v = self._mutate_de2(
                    X, i, best_i, r1, r2, r3, opt.F, opt.lmbda,
                )
                v = self._clip_to_bounds(v, lo, hi)

                # Crossover
                u = self._crossover_pop(X[i], v, opt.CR, rng)
                u = self._clip_to_bounds(u, lo, hi)

                # Selection
                f_u = float(eval_(u))
                if f_u <= f[i]:
                    X[i]     = u
                    f[i]     = f_u
                    last_eval = i

            self._record_generation(
                k, X, f, best_X_hist, best_f_hist, mean_f_hist, std_f_hist,
            )
            if self.feasible_check is not None:
                feas_best_X, feas_best_f = self._update_best_feasible(
                    X, f, feas_best_X, feas_best_f,
                )

            self._emit_snapshot(
                k, D,
                best_X_hist, best_f_hist, mean_f_hist, std_f_hist,
                feas_best_X, feas_best_f,
            )

            self._archive_generation(k, X, f, last_i=last_eval)

            self.logger.info(
                f"gen {k:>4}/{k_max} | best={best_f_hist[k]:.4f} "
                f"| mean={mean_f_hist[k]:.4f} | std={std_f_hist[k]:.4f}"
                + (f" | feasible={feas_best_f:.4f}"
                   if feas_best_f < float('inf') else "")
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                if self.archiver is None and self.runtime_data is not None:
                    curr_best_i = int(np.argmin(f))
                    if curr_best_i != last_eval:
                        eval_(X[curr_best_i])
                self.logger.debug(
                    self._debug_gen_msg(
                        k, best_f_hist, mean_f_hist, std_f_hist,
                    )
                )

            k_last = k
            if self._converged(k, std_f_hist, mean_f_hist):
                self.logger.info(
                    f"DE early-stop at gen {k} "
                    f"(tol={self.tol}, std collapsed)."
                )
                break
            if self._stalled(k, best_f_hist):
                self.logger.warning(
                    f"[CL3O] DE early-stop at gen {k} - best fitness "
                    f"plateaued (stall_patience={self.stall_patience} reached)."
                )
                break

        # -------- 3. Pack history --------
        self.logger.info(
            f"DE run complete.\n"
            f"| generations   : {k_last}\n"
            f"| best fitness  : {best_f_hist[k_last]:.4f}\n"
            f"| best feasible : {feas_best_f:.4f}"
        )
        n_eff = k_last + 1
        self.history = HistoryData(
            ng      = k_last,
            D      = D,
            best_X     = best_X_hist[:n_eff],
            best_f     = best_f_hist[:n_eff],
            mean_f     = mean_f_hist[:n_eff],
            std_f      = std_f_hist [:n_eff],
            feasible_X = feas_best_X,
            feasible_f = float(feas_best_f),
        )

        if self.archiver is not None:
            self.archiver.write_manifest(
                history   = self.history,
                opt_data  = opt,
                run_label = self.run_label,
            )

    # ----------------------------------------------------------------
    # Private - Early stopping and live callback
    # ----------------------------------------------------------------

    def _converged(
        self,
        k           : int,
        std_f_hist  : np.ndarray,
        mean_f_hist : np.ndarray,
    ) -> bool:
        '''
        Declare convergence when population std at gen k is below
        `tol * |mean_f|`. The relative check keeps the threshold
        proportional to the fitness scale (kg), so the criterion fires
        when the population has effectively collapsed regardless of the
        absolute magnitude of the objective.
        '''
        mean = abs(float(mean_f_hist[k]))
        return float(std_f_hist[k]) < self.tol * max(1.0, mean)

    def _stalled(
        self,
        k           : int,
        best_f_hist : np.ndarray,
    ) -> bool:
        '''
        Declare a stall when the running best fitness has not improved
        (by more than `STALL_REL_TOL` relative tolerance) for the last
        `stall_patience` generations. A stalled DE can otherwise drift
        for many wasted generations before the std-collapse fires.
        '''
        if self.stall_patience <= 0 or k < self.stall_patience:
            return False
        refefence  = float(best_f_hist[k - self.stall_patience])
        current = float(best_f_hist[k])
        # No improvement = curr is not meaningfully smaller than ref.
        delta = refefence - current
        return delta <= STALL_REL_TOL * max(1.0, abs(refefence))

    def _emit_snapshot(
        self,
        k           : int,
        D       : int,
        best_X_hist : np.ndarray,
        best_f_hist : np.ndarray,
        mean_f_hist : np.ndarray,
        std_f_hist  : np.ndarray,
        feas_best_X : np.ndarray,
        feas_best_f : float,
    ) -> None:
        '''Build a trimmed HistoryData and fire the on_generation hook.'''
        if self.on_generation is None:
            return
        n_eff = k + 1
        snap = HistoryData(
            ng      = k,
            D      = D,
            best_X     = best_X_hist[:n_eff].copy(),
            best_f     = best_f_hist[:n_eff].copy(),
            mean_f     = mean_f_hist[:n_eff].copy(),
            std_f      = std_f_hist [:n_eff].copy(),
            feasible_X = feas_best_X.copy(),
            feasible_f = float(feas_best_f),
        )
        try:
            self.on_generation(k, snap)
        except Exception as exc:
            self.logger.debug(f"on_generation callback raised: {exc!r}")

    # ----------------------------------------------------------------
    # Private - History bookkeeping
    # ----------------------------------------------------------------

    def _record_generation(
        self,
        k           : int,
        X           : np.ndarray,
        f           : np.ndarray,
        best_X_hist : np.ndarray,
        best_f_hist : np.ndarray,
        mean_f_hist : np.ndarray,
        std_f_hist  : np.ndarray,
    ) -> None:
        '''Stash best/mean/std fitness and best design for generation k.'''
        best_i           = int(np.argmin(f))
        best_X_hist[k]   = X[best_i]
        best_f_hist[k]   = float(f[best_i])
        mean_f_hist[k]   = float(np.mean(f))
        std_f_hist [k]   = float(np.std (f))

    def _archive_generation(
        self,
        k      : int,
        X      : np.ndarray,
        f      : np.ndarray,
        last_i : int = -1,
    ) -> None:
        '''
        Ensure runtime_data reflects the best individual of generation k,
        then hand off to the archiver. No-op when no archiver was configured.

        Args:
            k     : Generation index.
            X     : Current population (NP, D).
            f     : Current fitness vector (NP,).
            last_i: Index of the last individual evaluated in the selection
                loop. When it matches best_i, the shared RuntimeData already
                holds the correct state and the extra eval is skipped.
        '''
        if self.archiver is None:
            return
        best_i = int(np.argmin(f))
        if best_i != last_i:
            self.setup.evaluator(X[best_i])
        self.archiver.archive(k, X=X[best_i])

    def _update_best_feasible(
        self,
        X            : np.ndarray,
        f            : np.ndarray,
        feas_best_X  : np.ndarray,
        feas_best_f  : float,
    ) -> tuple[np.ndarray, float]:
        '''Track the lowest-fitness feasible design found so far.'''
        order = np.argsort(f)
        for i in order:
            if self.feasible_check(X[i]):
                if float(f[i]) < feas_best_f:
                    return X[i].copy(), float(f[i])
                break
        return feas_best_X, feas_best_f

    def _debug_gen_msg(
        self,
        k           : int,
        best_f_hist : np.ndarray,
        mean_f_hist : np.ndarray,
        std_f_hist  : np.ndarray,
    ) -> str:
        '''
        Build the per-generation DEBUG log string.

        Includes best/mean/std fitness plus, when runtime_data is
        available, the mass, penalty, Tsai-Wu MS, displacement MS,
        max normal stress, max shear stress, and max tip displacement
        for the generation best individual.

        Args:
            k           : Generation index.
            best_f_hist : Per-generation best fitness array.
            mean_f_hist : Per-generation mean fitness array.
            std_f_hist  : Per-generation std fitness array.

        Returns:
            Formatted multi-line string ready for logger.debug().
        '''
        bar = "-" * 60
        header = (
            f"\n{bar}\n"
            f"| gen {k:4d} || best={best_f_hist[k]:.4f} "
            f"|| mean={mean_f_hist[k]:.4f} "
            f"|| std={std_f_hist[k]:.4f}"
            f"\n{bar}"
        )
        if self.runtime_data is None:
            return header
        rt = self.runtime_data
        try:
            mass    = float(rt.score.total)
            pen     = float(rt.fitness.penalty)
            ms_tsw  = float(rt.tsw.MS_min)
            nv_tsw  = int(rt.tsw.nv)
            ms_disp = float(rt.displ.MS_min)
            nv_disp = int(rt.displ.nv)
            sig_max = float(max(
                np.max(np.abs(rt.stress.sigmaA)),
                np.max(np.abs(rt.stress.sigmaB)),
            ))
            tau_max = float(max(
                np.max(np.abs(rt.stress.tauA)),
                np.max(np.abs(rt.stress.tauB)),
            ))
            u_max = float(
                np.max(np.abs(rt.fea_rts.dmatrix[0:3, :, :]))
            )
            th_max = float(
                np.max(np.abs(rt.fea_rts.dmatrix[4, :, :]))
            )
        except Exception:
            return header
        return (
            f"{header}\n"
            f"| mass     : {mass:.4f} kg\n"
            f"| penalty  : {pen:.4f} kg\n"
            f"| tsw MS   : {ms_tsw:.4f}  (nv={nv_tsw})\n"
            f"| disp MS  : {ms_disp:.4f}  (nv={nv_disp})\n"
            f"| sigma_max: {sig_max:.2f} MPa\n"
            f"| tau_max  : {tau_max:.2f} MPa\n"
            f"| u_max    : {u_max:.4f} mm\n"
            f"| th_max   : {np.degrees(th_max):.2f} deg"
        )

    # ----------------------------------------------------------------
    # Private - DE math primitives
    # ----------------------------------------------------------------

    @staticmethod
    def _pick_three_distinct(
        rng : np.random.Generator,
        NP  : int,
        i   : int,
    ) -> tuple[int, int, int]:
        '''Return three indices in [0, NP) all distinct from i.'''
        pool = [k for k in range(NP) if k != i]
        r1, r2, r3 = rng.choice(pool, size=3, replace=False)
        return int(r1), int(r2), int(r3)

    @staticmethod
    def _mutate_de2(
        X     : np.ndarray,
        i     : int,
        best  : int,
        r1    : int,
        r2    : int,
        r3    : int,
        F     : float,
        lam   : float,
    ) -> np.ndarray:
        '''DE-2 mutation: v = x_r1 + F*(x_r2 - x_r3) + lam*(x_best - x_i).'''
        return (
            X[r1]
            + F   * (X[r2]   - X[r3])
            + lam * (X[best] - X[i])
        )

    @staticmethod
    def _crossover_pop(
        x    : np.ndarray,
        v    : np.ndarray,
        CR   : float,
        rng  : np.random.Generator,
    ) -> np.ndarray:
        '''Exponential crossover per Eq. 2.102-2.104.

        Copies a contiguous, modularly-wrapped block from v into x.
        Starting index n is drawn uniformly; the run extends one
        position at a time while independent Bernoulli(CR) trials
        succeed (P(L=nu) = CR^nu). At least the starting component
        is always taken from v (L >= 1).
        '''
        D = x.size
        n = int(rng.integers(0, D))
        u = x.copy()
        L = 0
        while L < D:
            u[(n + L) % D] = v[(n + L) % D]
            L += 1
            if rng.uniform() >= CR:
                break
        return u

    @staticmethod
    def _clip_to_bounds(
        x         : np.ndarray,
        bounds_lo : np.ndarray,
        bounds_hi : np.ndarray,
    ) -> np.ndarray:
        '''Clamp a candidate vector to [bounds_lo, bounds_hi].'''
        return np.minimum(np.maximum(x, bounds_lo), bounds_hi)


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
