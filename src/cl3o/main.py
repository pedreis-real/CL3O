'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Main module.

This module is responsible for calling all other principal modules of CL3O
software, a computational procedure for optimizing the structural framework
and composite layup of a Lifting Surface (LS). This software is based on the
following assumptions:

1. The LS cross section is formed by three closed cells.
2. The core calculations are based on structural idealization, matrix
   structural analysis and Tsai-Wu failure criteria.
3. It uses Differential Evolution algorithm for optmizing the LS.

See the README and the accompanying undergraduate thesis:

"Desenvolvimento de software para dimensionamento estrutural
preliminar de asas em material composto"

for the full theoretical background.

========================================================
| Created by: Pedro Henrique Reis da Silva Lima        |
| Revised by: Prof. Me. Luiz Fernando Barbosa Carvalho |
| First deployment date: 27/05/2026                    |
========================================================
|      Universidade Federal de Minas Gerais - UFMG     |
|        Bacharelado em Engenharia Aeroespacial        |
|                                                      |
|                 Belo Horizonte - MG                  |
|                         2026                         |
========================================================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@                                MIT License                                   @
@                                                                              @
@ Copyright (c) 2026-present Pedro Reis and Contributors                       @
@                                                                              @
@ Permission is hereby granted, free of charge, to any person obtaining a      @
@ copy of this software and associated documentation files (the "Software"),   @
@ to deal in the Software without restriction, including without limitation    @
@ the rights to use, copy, modify, merge, publish, distribute, sublicense,     @
@ and/or sell copies of the Software, and to permit persons to whom the        @
@ Software is furnished to do so, subject to the following conditions:         @
@                                                                              @
@ The above copyright notice and this permission notice shall be included in   @
@ all copies or substantial portions of the Software.                          @
@                                                                              @
@ THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   @
@ IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     @
@ FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL      @
@ THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER   @
@ LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      @
@ FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER          @
@ DEALINGS IN THE SOFTWARE.                                                    @
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--------------------------------------------------------------------------------
Database building, i.e., "CL3O/data/**.JSON" file creation, are delegated
to the module level usage of related database:
================
    + wing      -->  ../ data / wings     / < id >_WingData.json
    + airfoil   -->  ../ data / airfoils  / < id >_AirfoilData.json
    + material  -->  ../ data / materials / < id >_LaminateData.json
    + exloads   -->  ../ data / loads     / < id >_ExLoadsData.json
    + inloads   -->  ../ data / loads     / < id >_InLoadsData.json
    + oppoints  -->  ../ data / oppoints  / < id >_OppData.json

RunCLEO
========
- Orchestrates the main Runtime procedure of the software.
- Loads all static data from database.
- Setup and Run optimization runtime enviroment

PostProcessing
===============
- Plots (possibly) all relevant outputs of the analysis.

================================================================================
'''

# ================ PyLib imports ================
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Type

from dataclasses import dataclass, field

import numpy as np

# ================ Default Database Paths ================
from cl3o.paths import (
    WINGS_DIR    as _DFLT_WNG_DIR,
    AIRFOILS_DIR as _DFLT_AFL_DIR,
    PLIES_DIR    as _DFLT_PLY_DIR,
    MATERIALS_DIR as _DFLT_MAT_DIR,
    OPPOINTS_DIR as _DFLT_OPP_DIR,
    LOADS_DIR    as _DFLT_LDS_DIR,
    OUTPUTS_DIR  as _DFLT_OUT_DIR,
)

# ================ Module imports ================

# Constants
from cl3o.Constants import DE_HYPERPAR, WING_SIDE, GEOM_CACHE_MAXSIZE, LAYUP_ORDER

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils.lru_cache import LRUCache
from cl3o.utils.oppoints import OppData

# Optimization
from cl3o.optimization.de_opt import (
    SetupOpt, RunOpt, OptData, HistoryData, OptVars
)
from cl3o.optimization.fobjective import BuildEvaluator, RuntimeData

# stdlib (used by run_single)
import copy as _copy
import pickle as _pickle

# Materials
from cl3o.materials.laminate import LaminateData, PlyData

# Geometry
from cl3o.geometry.wing    import WingData, LerpWingData, WingHelper
from cl3o.geometry.airfoil import AirfoilData

# Finite Element Analysis
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData
from cl3o.fea.pre.fem_setup import FemSetup, FemPreprocessData

# Results
from cl3o.results.live_plotter import LivePlotter


# ================================================================================
# INTERNAL HELPERS API
# ================================================================================

# -------- Runner options --------
_DFLT_RUNNER_OPTIONS: dict[str, bool] = {
    "use_local_in_sr" : True,
    "use_offset"      : True,
    "pipeline_logging": False,
    "enable_logging"  : True,
    "live_plot"       : True,
}

class MainHelpers:
    '''
    Collection of static helper methods used across the main module.

    These are small utilities shared by :main: APIs
    '''

    @staticmethod
    def banner(text: str, width: int = 80) -> None:
        '''Print a formatted section banner to std::out.'''
        bar = "=" * width
        print(f"\n{bar}\n  {text}\n{bar}")

    @staticmethod
    def section_separator(
        label: str = "",
        char: str = "-",
        width: int = 80,
    ) -> str:
        '''
        Return a separator string, optionally embedding a label.
        Default width is 80 characters long.
        '''
        if label:
            side = (width - len(label) - 2) // 2
            return f"{char * side} {label} {char * side}"
        return char * width

    @staticmethod
    def spec_key(spec: object) -> str:
        '''Return the dcls class name.'''
        return spec.dcls.__name__

    @staticmethod
    def spec_field_name(spec: object) -> str:
        '''Return the StaticData field name for spec.'''
        name = spec.dcls.__name__
        if name.endswith("Data"):
            name = name.removesuffix("Data")
        return f"{name.lower()}_db"

    @staticmethod
    def spec_filepath(spec: object) -> Path:
        '''Return the full JSON file path for spec.'''
        return spec.dirpath / f"{spec.selfhood}_{spec.dcls.__name__}.json"

    @staticmethod
    def spec_type_nbr(spec: object) -> int:
        '''Return the integer type code for spec.dcls.'''
        match spec.dcls.__name__:
            case "WingData":     return 1
            case "AirfoilData":  return 2
            case "LaminateData": return 3
            case "OppData":      return 4
            case "ExLoadsData":  return 5
            case "InLoadsData":  return 6
            case _:              return -1

    @staticmethod
    def verify_missing_database(
        db_specs: list[object]
    ) -> None:
        '''Raise FileNotFoundError if any spec filepath does not exist.'''
        for idx, spec in enumerate(db_specs):
            if not spec.filepath.exists():
                raise FileNotFoundError(
                    f"[CL3O] Required database file not found.\n"
                    f"| Index : {idx}\n"
                    f"| Path  : {spec.filepath}\n"
                    f"Run directly the related module first"
                    f" to create the missing archive."
                )



# ================================================================================
# DATA PERSISTENCE API
# ================================================================================

# -------- Specifications --------
@dataclass
class DatabaseSpec:
    '''
    Container for one database file specification.
    Fields dcls/dirpath/selfhood are set by the caller; the remaining
    four are populated once by _resolve_db_specs via MainHelpers static methods.

    Property        Description
    ------------    --------------------------------------------------------
    dcls            Dataclass type for the database entry
    dirpath         Directory path where the JSON file resides
    selfhood        Identity key used to name the JSON file
    key             dcls class name (derived)
    field_name      StaticData field name (derived)
    filepath        Full JSON file path (derived)
    type_nbr        Integer type code 1..6, -1 for unknown (derived)
    '''
    dcls       : Type[Any]
    dirpath    : Path
    selfhood   : str
    key        : str       = ""
    field_name : str       = ""
    filepath   : Path      = field(default_factory=Path)
    type_nbr   : int       = -1


# -------- Static data --------
@dataclass
class StaticData:
    '''
    Container for all static data used in CL3O main routine.
    
    These variables remains unchanged during Runtime. Every class
    attribute maps directly to one JSON database file.

    The database is writen only once per runtime. So it remains
    "static" during optimization!

    Property        Description
    ------------    --------------------------------------------------------
    wing_db         Wing geometrical properties per-cpt
    lerp_wing_db    Wing geometrical properties per-station
    airfoil_db      Database of processed airfoil data for all wing profiles
    laminate_db     Database of materials, classified by number
    opp_db          Operational points: V-n, flight altitude, atmosISA
    exloads_db      External Loads per-station x condition
    inloads_db      Internal Loads per-station x condition

    fem_setup       FEA artifacts that do not change during runtime

    opt_setup       Differential Evolution algorithm hyperparameters
    opt_result      Optimization runtime results
    '''
    wing_db      : WingData                = field(default=None)
    lerp_wing_db : LerpWingData            = field(default=None)
    airfoil_db   : dict[str, AirfoilData]  = field(default=None)
    laminate_db  : dict[str, LaminateData] = field(default=None)
    ply_db       : dict[str, PlyData]      = field(default=None)
    opp_db       : OppData                 = field(default=None)
    exloads_db   : ExLoadsData             = field(default=None)
    inloads_db   : InLoadsData             = field(default=None)

    fem_setup    : FemPreprocessData       = field(default=None)

    # Cross-section geometry cache shared across all DE evaluations.
    # Key: (station_idx, xw1_r, xw2_r, bf*_r, ls1..lf4) — all vars that affect GeomData.
    # Hit rate grows as population converges; cost on miss is one dict lookup.
    # Bounded LRU: caps RAM on long sweeps while keeping recently-used (i.e.
    # repeated) candidates resident. See Constants.GEOM_CACHE_MAXSIZE.
    geom_cache   : LRUCache                = field(
        default_factory=lambda: LRUCache(GEOM_CACHE_MAXSIZE)
    )

    opt_setup    : OptData                 = field(default=None)
    opt_result   : HistoryData             = field(default=None)



# ================================================================================
# RUNTIME API - Runs CL3O main routine procedure
# ================================================================================

class RunCLEO:
    '''
    Top-level CL3O runtime driver.

    Centralises the full setup so that, after __init__, both
    `self.static.fem_setup` and `self.static.opt_setup` are already
    populated. The DE outer loop is then driven by `run_optimization`,
    while the end-to-end pipeline (live plotter + DE + post-processing)
    is wrapped by the convenience `run` method.

    Setup sequence performed by __init__:
        1. Load all static JSON databases (populates fem_setup).
        2. Assemble the DE evaluator closure via BuildEvaluator.
        3. Resolve DE bounds (caller-supplied or default OptVars layout).
        4. Build SetupOpt and stash it on static.opt_setup.
    '''

    def __init__(
        self,
        aircraft_name  : str,
        opt_name       : str,
        db_specs       : list[object],
        bounds_lo      : np.ndarray | None = None,
        bounds_hi      : np.ndarray | None = None,
        de_hyperpar    : dict = DE_HYPERPAR,
        runner_options : dict[str, bool] | None = None,
    ) -> None:
        '''
        Args:
            aircraft_name : Label for banner / output directory.
            opt_name      : Label for banner / output directory.
            db_specs      : Resolved DatabaseSpec list (post _resolve_db_specs).
            bounds_lo     : Optional (D,) DE lower bounds. When omitted,
                SetupOpt._build_de_bounds() derives them from OPT_LIMS.
            bounds_hi     : Optional (D,) DE upper bounds.
            de_hyperpar   : DE hyper-parameters dict (NP, CR, F, lam,
                k_max, seed). Defaults to DFLT_DE_HYPERPAR from Constants.
            runner_options: dict[str, bool] with all pipeline boolean switches.
                Defaults to _DFLT_RUNNER_OPTIONS (all features at their defaults).
        '''
        runner_options = {**_DFLT_RUNNER_OPTIONS, **(runner_options or {})}
        self.runner_options = runner_options
        self.logger = io.setup_logger(self, runner_options["enable_logging"])
        # io file-I/O traces are DEBUG; surface them only under pipeline_logging.
        io.set_io_verbose(runner_options["pipeline_logging"])

        self.aircraft_name  = aircraft_name
        self.opt_name       = opt_name
        self.db_specs       = db_specs
        self.de_hyperpar    = de_hyperpar

        MainHelpers.banner(f"CL3O  -  {self.aircraft_name}  |  {self.opt_name}")
        self.logger.info(
            f"Initialising RunCLEO [aircraft={self.aircraft_name}, "
            f"opt={self.opt_name}]."
        )
        self.logger.debug(
            f"Runner options: "
            + ", ".join(f"{k}={v}" for k, v in runner_options.items())
        )

        # -------- 1. Load static data (populates fem_setup) --------
        self.static : StaticData  = self._import_database()
        self.logger.info("Step 1/4: static databases loaded; fem_setup ready.")

        # -------- 2. Assemble the DE evaluator closure --------
        builder = BuildEvaluator(
            static_data      = self.static,
            use_local_in_sr  = runner_options["use_local_in_sr"],
            use_offset       = runner_options["use_offset"],
            pipeline_logging = runner_options["pipeline_logging"],
            enable_logging   = runner_options["enable_logging"],
        )
        self.evaluator = builder.eval_
        self.runtime   = builder.rt
        self._builder  = builder
        self.logger.info("Step 2/4: DE evaluator assembled.")

        # -------- 3. Resolve DE bounds --------
        if bounds_lo is None or bounds_hi is None:
            n_cpts = int(self.static.wing_db.n_cpts)
            n_mats = len(self.static.laminate_db)
            bounds_lo, bounds_hi = SetupOpt._build_de_bounds(n_cpts, n_mats)
        self.logger.info(
            f"Step 3/4: DE bounds resolved (D={int(np.asarray(bounds_lo).size)})."
        )

        # -------- 4. Build SetupOpt (populates opt_setup) --------
        self.static.opt_setup = SetupOpt(
            evaluator      = self.evaluator,
            enable_logging = runner_options["enable_logging"],
            verbose        = runner_options["pipeline_logging"],
            de_hyperpar    = self.de_hyperpar,
            bounds_lo      = bounds_lo,
            bounds_hi      = bounds_hi,
        )
        self.logger.debug("Step 4/4: static.opt_setup populated.")
        self.logger.info(
            "RunCLEO ready."
        )

    # ----------------------------------------
    # Public method - end-to-end pipeline
    # ----------------------------------------

    def run(
        self,
        out_dir        : str | Path | None = None,
        feasible_check : Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        '''
        End-to-end driver: DE optimization plus optional post-processing.

        Args:
            out_dir       : Destination directory for post-processing.
                Defaults to outputs/<aircraft>_<opt>.
            feasible_check: Optional X -> bool callable for feasibility.
        '''
        out_dir = Path(out_dir) if out_dir is not None else (
            _DFLT_OUT_DIR
            / f"{self.aircraft_name.lower()}_{self.opt_name.lower()}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {out_dir}")

        on_gen = None
        if self.runner_options["live_plot"]:
            try:
                plotter = LivePlotter(self.static)
                self.logger.info("Live viewer enabled.")

                def on_gen(k: int, hist: HistoryData) -> None:
                    plotter.update(k, hist, self._builder.best_rt)
            except RuntimeError as exc:
                self.logger.warning(
                    f"[CL3O] Live viewer unavailable - continuing headless.\n"
                    f"| reason : {exc}"
                )
                on_gen = None
        else:
            self.logger.info("Live viewer disabled (live_plot=False).")

        self.run_optimization(
            feasible_check = feasible_check,
            on_generation  = on_gen,
            out_dir        = out_dir,
        )

    # ----------------------------------------
    # Public method - DE optimization driver
    # ----------------------------------------

    def run_optimization(
        self,
        on_generation  : Callable[[int, HistoryData], None] | None = None,
        feasible_check : Callable[[np.ndarray], bool] | None = None,
        out_dir        : str | Path | None = None,
    ) -> HistoryData:
        '''
        Execute the DE outer loop using the cached self.static.opt_setup
        built in __init__. Stores the RunOpt instance on
        self.static.opt_result and returns the per-generation history.

        When `out_dir` is supplied, the DE archiver pickles one
        RuntimeData snapshot per generation under `out_dir/generations/`
        and writes `out_dir/manifest.json` for offline UI consumption.
        '''
        setup = self.static.opt_setup
        if setup is None:
            raise RuntimeError(
                "[CL3O] static.opt_setup is empty - RunCLEO.__init__ did "
                "not run SetupOpt. Re-instantiate RunCLEO."
            )

        self.logger.info(
            f"Starting DE run [NP={setup.data.NP}, k_max={setup.data.k_max}, "
            f"D={setup.data.D}]."
        )
        run = RunOpt(
            setup          = setup,
            feasible_check = feasible_check,
            on_generation  = on_generation,
            runtime_data   = self.runtime,
            out_dir        = out_dir,
            run_label      = f"{self.aircraft_name}_{self.opt_name}",
            enable_logging = self.runner_options["enable_logging"],
            verbose        = self.runner_options["pipeline_logging"],
        )
        self.static.opt_result = run
        return run.history

    # ----------------------------------------
    # Public method - evaluate a single design
    # ----------------------------------------

    def run_single(
        self,
        X        : "list | np.ndarray | OptVars",
        out_path : str | Path | None = None,
    ) -> RuntimeData:
        '''
        Evaluate one design vector through the full CL3O pipeline,
        bypassing the DE outer loop. Returns a deep copy of the
        RuntimeData snapshot produced by the evaluator (so subsequent
        calls do not mutate it).

        Args:
            X       : Flat design vector (list / ndarray) or an OptVars
                container produced by a previous run.
            out_path: Optional file path. When given, the snapshot is
                pickled there in the same format as a per-generation
                archive entry, so the UI can load it standalone.
        '''
        if isinstance(X, OptVars):
            X_flat = BuildEvaluator.encode_optvars(X)
        else:
            X_flat = np.asarray(X, dtype=float).ravel()

        self.logger.info(f"Evaluating single design [D={X_flat.size}] ...")
        fitness = float(self.evaluator(X_flat))
        snap = _copy.deepcopy(self.runtime)
        self.logger.info(f"Single design evaluated: fitness z(X) = {fitness:.4f}")

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                _pickle.dump(snap, f, protocol=_pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Snapshot pickled to: {out_path}")

        return snap

    # ----------------------------------------
    # Private method - loads static data
    # ----------------------------------------

    def _import_database(self) -> StaticData:
        '''Load every JSON spec, plus ply files referenced by laminates,
        plus the global FEM pre-process artefacts, into a StaticData.'''
        db_loaded : dict = {}
        mat_list  : list = []
        afl_dict  : dict = {}

        self.logger.info(f"Loading {len(self.db_specs)} database specs ...")
        for spec in self.db_specs:
            obj = io.read_json(
                filepath = spec.filepath,
                dcls     = spec.dcls,
            )
            self.logger.debug(f"Loaded {spec.field_name} <- {spec.filepath}")
            if spec.field_name == "laminate_db":
                mat_list.append(obj)
            elif spec.field_name == "airfoil_db":
                afl_dict[spec.selfhood.lower()] = obj
            else:
                db_loaded[spec.field_name] = obj

        if mat_list:
            db_loaded["laminate_db"] = {
                f"MAT{k+1}": lam    # '+1 to start in MAT1, not MAT0'
                for k, lam in enumerate(mat_list)
            }
        if afl_dict:
            db_loaded["airfoil_db"] = afl_dict

        if mat_list:
            self.logger.info(f"Loaded {len(mat_list)} laminate(s).")
        else:
            self.logger.warning(
                "[CL3O] No laminates loaded - the catalogue is empty. "
                "Run composite_library to build MAT_*_LaminateData.json."
            )
        if not afl_dict:
            self.logger.warning("[CL3O] No airfoil data loaded.")

        if db_loaded.get("laminate_db"):
            ply_db: dict[str, PlyData] = {}
            for lam in db_loaded["laminate_db"].values():
                for ply_name in lam.plies:
                    if ply_name not in ply_db:
                        ply_db[ply_name] = io.read_json(
                            filepath = _DFLT_PLY_DIR / f"PlyData_{ply_name}.json",
                            dcls     = PlyData,
                        )
            db_loaded["ply_db"] = ply_db
            self.logger.info(f"Loaded {len(ply_db)} unique ply file(s).")

        # Fold the full-span stations onto the analyzed wing at runtime so the
        # mesh follows Constants.WING_SIDE without regenerating the loads DB.
        self.logger.debug(f"Folding wing stations onto side '{WING_SIDE}'.")
        db_loaded["lerp_wing_db"] = WingHelper.lerp_from_data(
            wng_data  = db_loaded["wing_db"],
            Y_sta     = np.asarray(db_loaded["exloads_db"].Y, dtype=float),
            wing_side = WING_SIDE,
        )

        db_loaded["fem_setup"] = FemSetup(
            exloads_db = db_loaded["exloads_db"],
            lerp_wing_db = db_loaded["lerp_wing_db"],
            wing_side = WING_SIDE,
            enable_logging = self.runner_options["enable_logging"],
        ).fem_setup

        self.logger.info("Static database load complete.")
        return StaticData(**db_loaded)


# ================================================================================
# TOP-LEVEL RUNTIME ROUTINES - Called sequentially by __main__
# ================================================================================

def _resolve_db_specs(
    db_specs: list[object],
) -> list[object]:
    '''
    Populate each DatabaseSpec's derived fields (key, field_name, filepath,
    type_nbr) by calling the MainHelpers static methods exactly once per spec.

    Args:
        db_specs: Raw spec list built by the caller.

    Returns:
        New list of DatabaseSpec with all four derived fields set.
    '''
    return [
        DatabaseSpec(
            dcls       = spec.dcls,
            dirpath    = spec.dirpath,
            selfhood   = spec.selfhood,
            key        = MainHelpers.spec_key(spec),
            field_name = MainHelpers.spec_field_name(spec),
            filepath   = MainHelpers.spec_filepath(spec),
            type_nbr   = MainHelpers.spec_type_nbr(spec),
        )
        for spec in db_specs
    ]


# ======================================================= #
# ------------------------------------------------------- #
# ======================================================= #
#         /$$$$$$  /$$        /$$$$$$   /$$$$$$           #
#        /$$__  $$| $$       /$$__  $$ /$$__  $$          #
#       | $$  \__/| $$      |__/  \ $$| $$  \ $$          #
#       | $$      | $$         /$$$$$/| $$  | $$          #
#       | $$      | $$        |___  $$| $$  | $$          #
#       | $$    $$| $$       /$$  \ $$| $$  | $$          #
#       |  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/          #
#        \______/ |________/ \______/  \______/           #
# ======================================================= #
# ------------------------------------------------------- #
# ======================================================= #

if __name__ == "__main__":
    aircraft_name = "DA62"
    opt_name = "Opt-final-newmat-seed-67 (83)"

    # ---------------- Set database specifications ----------------
    # Laminates are discovered by glob over MAT_*_LaminateData.json; the
    # MAT_ prefix (with underscore) selects the curated catalogue written
    # by composite_library.write_laminates and skips legacy test laminates
    # named MAT{int} (no underscore).
    skip_mat : list[str] = []

    # Load laminates in the canonical LAYUP_ORDER sequence (0-based indices).
    # Index k in the DE vector maps to LAYUP_ORDER[k] (stored as MAT{k+1}).
    materials_to_load = [
        name for name in LAYUP_ORDER
        if (_DFLT_MAT_DIR / f"{name}_LaminateData.json").exists()
        and name not in skip_mat
    ]

    airfoils_to_load = [
        "WortmannFX63137",
    ]

    database_loading_specs: list[object] = []
    database_loading_specs.append(
        # type_nbr = 1
        DatabaseSpec(WingData, _DFLT_WNG_DIR, f"{aircraft_name.lower()}_simplified"),
    )
    for afl in airfoils_to_load:
        database_loading_specs.append(
            # type_nbr = 2
            DatabaseSpec(AirfoilData, _DFLT_AFL_DIR, f"{afl.lower()}"),
        )
    for mat_name in materials_to_load:
        database_loading_specs.append(
            # type_nbr = 3
            DatabaseSpec(LaminateData, _DFLT_MAT_DIR, mat_name),
        )
    database_loading_specs.append(
        # type_nbr = 4
        DatabaseSpec(OppData, _DFLT_OPP_DIR, f"{aircraft_name.lower()}"),
    )
    database_loading_specs.append(
        # type_nbr = 5
        DatabaseSpec(ExLoadsData, _DFLT_LDS_DIR, f"{aircraft_name.lower()}"),
    )
    database_loading_specs.append(
        # type_nbr = 6
        DatabaseSpec(InLoadsData, _DFLT_LDS_DIR, f"{aircraft_name.lower()}"),
    )

    db_specs = _resolve_db_specs(database_loading_specs)
    MainHelpers.verify_missing_database(db_specs)

    # ---------------- Runs CL3O main routine ----------------

    runner = RunCLEO(
        aircraft_name  = aircraft_name,
        db_specs       = db_specs,
        opt_name       = opt_name,
        runner_options = {
            "use_local_in_sr" : True,
            "use_offset"      : True,
            "live_plot"       : False,
            "pipeline_logging": False,
            "enable_logging"  : True,
        },
        de_hyperpar    = {**DE_HYPERPAR,
                          'seed'   : 83},
    )
    runner.run()
