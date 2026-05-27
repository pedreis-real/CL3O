'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Results Writer Module.

Serializes the outcome of a CL3O optimization run to a single JSON
archive, including DE hyper-parameters, per-generation history, and the
best design vector (both raw-best and best-feasible).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ================ Pathing ================
_HERE = Path(__file__).resolve().parent            # src/results/
_SRC  = _HERE.parent                               # src/
_ROOT = _SRC.parent                                # CL3O/

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Default Database Paths ================
_DFLT_OUT_DIR = _ROOT / "outputs"

# ================ Module imports ================
from utils import io_utils as io
from optimization.de_opt import OptData, HistoryData

# ================ Global variables ================
_N_DEC = 4


# ================================================================================
# Data persistence - Run archive container
# ================================================================================

@dataclass
class ResultsData:
    '''
    Container bundling every numeric output from a DE run.

    Property       Size               Description                          Units
    -----------    ----------------   -----------------------------    --------
    aircraft       scalar             Aircraft identifier              str
    opt_name       scalar             Optimization run label           str
    n_dim          scalar             Design-space dimension           -
    n_gen          scalar             Number of generations executed   -
    NP             scalar             Population size                  -
    CR             scalar             Crossover probability            -
    F              scalar             Differential weight              -
    lam            scalar             DE-3 best-attraction weight      -
    k_max          scalar             Max generations                  -
    seed           scalar             RNG seed                         -
    bounds_lo      (n_dim,)           Lower bounds per design var      -
    bounds_hi      (n_dim,)           Upper bounds per design var      -
    best_X         (n_gen+1, n_dim)   Best design each generation      -
    best_f         (n_gen+1,)         Best fitness each generation     -
    mean_f         (n_gen+1,)         Population mean fitness          -
    std_f          (n_gen+1,)         Population fitness std           -
    feasible_X     (n_dim,)           Best feasible design found       -
    feasible_f     scalar             Fitness of best feasible design  -
    '''
    aircraft   : str        = ""
    opt_name   : str        = ""
    n_dim      : int        = 0
    n_gen      : int        = 0
    NP         : int        = 0
    CR         : float      = 0.0
    F          : float      = 0.0
    lam        : float      = 0.0
    k_max      : int        = 0
    seed       : int        = 0
    bounds_lo  : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bounds_hi  : np.ndarray = field(default_factory=lambda: np.zeros(0))
    best_X     : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    best_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    std_f      : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_X : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_f : float      = float("inf")


# ================================================================================
# Internal Helpers
# ================================================================================

class ResultsHelper:
    def __init__(self):
        pass

    @staticmethod
    def default_filepath(
        aircraft_name : str,
        opt_name      : str,
        out_dir       : Path = _DFLT_OUT_DIR,
    ) -> Path:
        '''Return the default archive path under CL3O/outputs/.'''
        stem = f"{aircraft_name.lower()}_{opt_name.lower()}_ResultsData.json"
        return out_dir / stem

    @staticmethod
    def pack_results(
        aircraft_name : str,
        opt_name      : str,
        opt           : OptData,
        history       : HistoryData,
    ) -> ResultsData:
        '''Fold OptData + HistoryData into a single ResultsData archive.'''
        return ResultsData(
            aircraft   = str(aircraft_name),
            opt_name   = str(opt_name),
            n_dim      = int(history.D),
            n_gen      = int(history.ng),
            NP         = int(opt.NP),
            CR         = float(opt.CR),
            F          = float(opt.F),
            lam        = float(opt.lmbda),
            k_max      = int(opt.k_max),
            seed       = int(opt.seed),
            bounds_lo  = np.round(opt.lo,      _N_DEC),
            bounds_hi  = np.round(opt.hi,      _N_DEC),
            best_X     = np.round(history.best_X,     _N_DEC),
            best_f     = np.round(history.best_f,     _N_DEC),
            mean_f     = np.round(history.mean_f,     _N_DEC),
            std_f      = np.round(history.std_f,      _N_DEC),
            feasible_X = np.round(history.feasible_X, _N_DEC),
            feasible_f = float(round(history.feasible_f, _N_DEC)),
        )


# ================================================================================
# PUBLIC API - Persist DE run as JSON archive
# ================================================================================

class ResultsWriter:
    '''
    Persist a CL3O optimization run to disk as JSON.

    Use:
        writer = ResultsWriter(aircraft_name, opt_name, opt, history)
        writer.write()
        data = writer.data                    # ResultsData
    '''

    def __init__(
        self,
        aircraft_name  : str,
        opt_name       : str,
        opt_data       : OptData,
        history        : HistoryData,
        out_dir        : Optional[str | Path] = None,
        enable_logging : bool                 = True,
    ) -> None:
        '''
        Args:
            aircraft_name : Aircraft identifier (used in filename).
            opt_name      : Optimization run label (used in filename).
            opt_data      : OptData produced by SetupOpt.
            history       : HistoryData produced by RunOpt.
            out_dir       : Optional override directory. Defaults to
                CL3O/outputs/.
            enable_logging: Toggle logger.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        # -------- 1. Store inputs --------
        self.aircraft_name = str(aircraft_name)
        self.opt_name      = str(opt_name)
        self.out_dir       = (
            Path(out_dir) if out_dir is not None else _DFLT_OUT_DIR
        )

        # -------- 2. Pack archive --------
        self.data = ResultsHelper.pack_results(
            aircraft_name = self.aircraft_name,
            opt_name      = self.opt_name,
            opt           = opt_data,
            history       = history,
        )

        # -------- 3. Resolve default filepath --------
        self.filepath = ResultsHelper.default_filepath(
            aircraft_name = self.aircraft_name,
            opt_name      = self.opt_name,
            out_dir       = self.out_dir,
        )

        self.logger.info(
            f"ResultsWriter ready.\n"
            f"| aircraft : {self.aircraft_name}\n"
            f"| opt_name : {self.opt_name}\n"
            f"| n_gen    : {self.data.n_gen}\n"
            f"| n_dim    : {self.data.n_dim}\n"
            f"| filepath : {self.filepath}"
        )

    # ----------------------------------------------------------------
    # Public - Disk I/O
    # ----------------------------------------------------------------

    def write(
        self,
        filepath : Optional[str | Path] = None,
    ) -> Path:
        '''
        Serialize the bundled ResultsData to JSON.

        Args:
            filepath: Optional override path. Defaults to self.filepath.

        Returns:
            Resolved Path of the written file.
        '''
        target = (
            Path(filepath) if filepath is not None else self.filepath
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        io.write_json(obj=self.data, filepath=target)

        self.logger.info(f"Results archive written to {target}")
        return target


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
