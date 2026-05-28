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

See the README and the accompanying undergraduate thesis (TCC, UFMG 2026)
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
from typing import Any, Callable, Optional, Type

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
try:
    import vtk                       # only the optional live 3-D viewer needs it
except ImportError:                  # headless env without OpenGL (e.g. CI)
    vtk = None

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
from cl3o.Constants import DE_HYPERPAR

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils.oppoints import OppData

# Optimization
from cl3o.optimization.de_opt    import SetupOpt, RunOpt, OptData, HistoryData, OptVars
from cl3o.optimization.fobjective import BuildEvaluator, RuntimeData

# Materials
from cl3o.materials.laminate import LaminateData, PlyData

# Geometry
from cl3o.geometry.wing    import WingData, LerpWingData, WingHelper
from cl3o.geometry.airfoil import AirfoilData

# Finite Element Analysis
from cl3o.fea.loads.load_mapper import ExLoadsData, InLoadsData
from cl3o.fea.pre.fem_setup import FemSetup, FemPreprocessData


# ================================================================================
# INTERNAL HELPERS API
# ================================================================================

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
        aircraft_name    : str,
        opt_name         : str,
        db_specs         : list[object],
        bounds_lo        : Optional[np.ndarray] = None,
        bounds_hi        : Optional[np.ndarray] = None,
        de_hyperpar      : dict = DE_HYPERPAR,
        pipeline_logging : bool  = False,
        enable_logging   : bool  = True,
    ) -> None:
        '''
        Args:
            aircraft_name   : Label for banner / output directory.
            opt_name        : Label for banner / output directory.
            db_specs        : Resolved DatabaseSpec list (post _resolve_db_specs).
            bounds_lo       : Optional (D,) DE lower bounds. When omitted,
                SetupOpt._build_de_bounds() derives them from OPT_LIMS.
            bounds_hi       : Optional (D,) DE upper bounds.
            de_hyperpar     : DE hyper-parameters dict (NP, CR, F, lam,
                k_max, seed). Defaults to DFLT_DE_HYPERPAR from Constants.
            tol             : DE early-stop tolerance.
            stall_patience  : Generations of no improvement before stop.
            pipeline_logging: Propagate info-level logs into the per-
                candidate pipeline sub-classes (noisy; default False).
            enable_logging  : Toggle logger for RunCLEO itself.
        '''
        self.logger = io.setup_logger(self, enable_logging)
        self.logger.info("Initialising RunCLEO")

        self.aircraft_name  = aircraft_name
        self.opt_name       = opt_name
        self.db_specs       = db_specs
        self.de_hyperpar    = de_hyperpar

        MainHelpers.banner(f"CL3O  -  {self.aircraft_name}  |  {self.opt_name}")

        # -------- 1. Load static data (populates fem_setup) --------
        self.static : StaticData  = self._import_database()
        self.logger.info("Step 1/4: static databases loaded; fem_setup ready.")

        # -------- 2. Assemble the DE evaluator closure --------
        builder = BuildEvaluator(
            static_data      = self.static,
            pipeline_logging = pipeline_logging,
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
            enable_logging = enable_logging,
            de_hyperpar    = self.de_hyperpar,
            bounds_lo      = bounds_lo,
            bounds_hi      = bounds_hi,
        )
        self.logger.debug("Step 4/4: static.opt_setup populated.")
        self.logger.info(
            "RunCLEO ready: fem_setup and opt_setup populated; "
            "call run() to drive the full pipeline."
        )

    # ----------------------------------------
    # Public method - end-to-end pipeline
    # ----------------------------------------

    def run(
        self,
        out_dir        : Optional[str | Path] = None,
        feasible_check : Optional[Callable[[np.ndarray], bool]] = None,
        live_plot      : bool = True,
    ) -> None:
        '''
        End-to-end driver: DE optimization plus optional post-processing.

        Args:
            out_dir       : Destination directory for post-processing.
                Defaults to outputs/<aircraft>_<opt>.
            feasible_check: Optional X -> bool callable for feasibility.
            live_plot     : When True, opens a live Matplotlib viewer
                updated every generation with the convergence curve and
                the 3D wing geometry of the current best individual.
        '''
        out_dir = Path(out_dir) if out_dir is not None else (
            _DFLT_OUT_DIR
            / f"{self.aircraft_name.lower()}_{self.opt_name.lower()}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        on_gen = None
        if live_plot:
            plotter = LivePlotter(self.static)

            def on_gen(k: int, hist: HistoryData) -> None:
                plotter.update(k, hist, self._builder.best_rt)

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
        on_generation  : Optional[Callable[[int, HistoryData], None]] = None,
        feasible_check : Optional[Callable[[np.ndarray], bool]] = None,
        out_dir        : Optional[str | Path] = None,
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
            enable_logging = True,
        )
        self.static.opt_result = run
        return run.history

    # ----------------------------------------
    # Private method - loads static data
    # ----------------------------------------

    def _import_database(self) -> StaticData:
        '''Load every JSON spec, plus ply files referenced by laminates,
        plus the global FEM pre-process artefacts, into a StaticData.'''
        db_loaded : dict = {}
        mat_list  : list = []
        afl_dict  : dict = {}

        for spec in self.db_specs:
            obj = io.read_json(
                filepath = spec.filepath,
                dcls     = spec.dcls,
            )
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

        db_loaded["lerp_wing_db"] = WingHelper.lerp_from_data(
            wng_data = db_loaded["wing_db"],
            Y_sta    = np.asarray(db_loaded["exloads_db"].Y_hf, dtype=float),
        )

        db_loaded["fem_setup"] = FemSetup(
            exloads_db = db_loaded["exloads_db"],
            lerp_wing_db = db_loaded["lerp_wing_db"]
        ).fem_setup

        return StaticData(**db_loaded)




# ================================================================================
# LIVE PLOT API - Matplotlib convergence and geometry viewer
# ================================================================================

class LivePlotter:
    '''
    Live viewer for the DE optimization loop.

    Opens two windows refreshed once per generation:
      - Matplotlib window : DE convergence curve with std shaded band.
      - VTK window        : 3-D wing geometry of the current best individual
                            (outer skin, booms, spar surfaces, centroid and
                            shear-centre lines). Camera is fixed at the
                            top-aft-left corner (Xmin, Ymin, Zmax).
    '''

    def __init__(
        self,
        static_data    : StaticData,
        enable_logging : bool = True,
    ) -> None:
        '''
        Args:
            static_data   : StaticData (wing geometry reference).
            enable_logging: Toggle logger.
        '''
        if vtk is None:
            raise RuntimeError(
                "The live 3-D viewer requires VTK, which failed to import "
                "(no OpenGL/libGL in this environment). Run with "
                "live_plot=False or install OpenGL libraries (e.g. libgl1)."
            )
        self.logger = io.setup_logger(self, enable_logging)
        self.st     = static_data

        # -------- Convergence panel (matplotlib) --------
        plt.ion()
        self.fig     = plt.figure(figsize=(5, 7))
        self.ax_conv = self.fig.add_subplot(1, 1, 1)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

        # Pin convergence window to the top-right corner of the screen
        _mgr = plt.get_current_fig_manager()
        try:
            import tkinter as _tk
            _root = _tk.Tk(); _root.withdraw()
            _sw   = _root.winfo_screenwidth()
            _root.destroy()
            _fw   = int(self.fig.get_size_inches()[0] * self.fig.dpi)
            _x    = max(0, _sw - _fw - 10)
            _mgr.window.update_idletasks()
            _mgr.window.wm_geometry(f"+{_x}+0")
            _mgr.window.update_idletasks()
        except Exception:
            try:
                _mgr.window.move(1290, 0)
            except Exception:
                pass

        # -------- Wing geometry panel (VTK) --------
        self._setup_vtk_window()

        self.logger.info("LivePlotter ready.")

    # ----------------------------------------
    # Public method - per-generation update
    # ----------------------------------------

    def update(
        self,
        k    : int,
        hist : HistoryData,
        rt   : RuntimeData,
    ) -> None:
        '''
        Redraw both panels for generation k.

        Args:
            k   : Current generation index.
            hist: HistoryData snapshot trimmed to [0, k].
            rt  : RuntimeData of the current best individual.
        '''
        self._update_convergence(k, hist)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        self._update_wing_vtk(rt)

    # ------------------------------------------------
    # Private method - VTK window initialisation
    # ------------------------------------------------

    def _setup_vtk_window(self) -> None:
        '''Initialise VTK renderer, render window, and interactor.'''
        self._ren = vtk.vtkRenderer()
        self._ren.SetBackground(0.12, 0.12, 0.12)

        self._rw = vtk.vtkRenderWindow()
        self._rw.AddRenderer(self._ren)
        self._rw.SetSize(1280, 720)
        self._rw.SetWindowName("CL3O - Wing Geometry")

        self._iren = vtk.vtkRenderWindowInteractor()
        self._iren.SetRenderWindow(self._rw)
        self._iren.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )
        self._iren.Initialize()
        self._rw.Render()

    # ----------------------------------------
    # Private method - convergence subplot
    # ----------------------------------------

    def _update_convergence(
        self,
        k    : int,
        hist : HistoryData,
    ) -> None:
        '''Redraw convergence with best-f curve and population std band.'''
        ax = self.ax_conv
        ax.cla()

        gens = np.arange(hist.ng + 1)

        ax.plot(
            gens, hist.best_f,
            color="#085302", lw=1.5, label='best f',
        )
        ax.fill_between(
            gens,
            hist.mean_f - hist.std_f,
            hist.mean_f + hist.std_f,
            alpha=0.25,
            color='#4DB8FF',
            label=r'mean $\pm$ std',
        )
        if hist.feasible_f < float('inf'):
            ax.axhline(
                hist.feasible_f,
                color='green',
                linestyle='--',
                lw=1.2,
                label=f'Feasible: {hist.feasible_f:.3f}',
            )

        ax.set_xlabel(r'Generation', fontsize=10)
        ax.set_ylabel(r'Fitness z(X)', fontsize=10)
        ax.tick_params(axis='both', labelsize=6)

        ax.set_title(f'DE Convergence  [gen {k}]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ----------------------------------------
    # Private method - VTK wing geometry
    # ----------------------------------------

    def _update_wing_vtk(self, rt: RuntimeData) -> None:
        '''
        Rebuild all VTK actors for the current best-individual pipeline.

        Coordinate layout: X = chord direction, Y = span, Z = vertical.
        Boom global coords: X = C[0] + boom_u[j], Z = C[2] + boom_w[j].
        Boom order (0-indexed): B1=0 .. B7=6.
        '''
        ren = self._ren
        ren.RemoveAllViewProps()

        secs = rt.sections
        Y    = np.array([sd.C[1] for sd in secs.sec_data])
        bX   = np.array(
            [[sd.C[0] + sd.boom_u[j] for j in range(7)]
             for sd in secs.sec_data]
        )
        bZ   = np.array(
            [[sd.C[2] + sd.boom_w[j] for j in range(7)]
             for sd in secs.sec_data]
        )

        # -------- (a) Outer wing skin surface --------
        profiles = []
        for sd in secs.sec_data:
            outer = np.concatenate([
                sd.T1[0]['pts'],
                sd.T1[1]['pts'][1:],
                sd.T1[2]['pts'][1:],
                sd.T1[3]['pts'][1:],
                sd.T1[4]['pts'][1:],
            ], axis=0)
            profiles.append(outer)
        ren.AddActor(LivePlotter._skin_actor(profiles, Y))
        ren.AddActor(LivePlotter._section_outlines_actor(profiles, Y))

        _SPAR_EDGE = (0.28, 0.28, 0.28)

        # -------- (b) Aft spar: B3 (idx 2) to B5 (idx 4) --------
        ren.AddActor(LivePlotter._spar_actor(
            Y,
            bX[:, 2], bZ[:, 2], bX[:, 4], bZ[:, 4],
            color=(0.27, 0.51, 0.71), alpha=0.72,
            edge_color=_SPAR_EDGE,
        ))

        # -------- (c) Rear spar: B1 (idx 0) to B7 (idx 6) --------
        ren.AddActor(LivePlotter._spar_actor(
            Y,
            bX[:, 0], bZ[:, 0], bX[:, 6], bZ[:, 6],
            color=(1.00, 0.39, 0.28), alpha=0.72,
            edge_color=_SPAR_EDGE,
        ))

        # -------- Axes labels and title overlay --------
        all_X = bX.ravel()
        all_Z = bZ.ravel()
        bounds = (
            float(all_X.min()), float(all_X.max()),
            float(Y.min()),     float(Y.max()),
            float(all_Z.min()), float(all_Z.max()),
        )
        LivePlotter._add_axes(ren, bounds)
        ren.AddActor2D(
            LivePlotter._title_actor("Wing Geometry (best individual)")
        )

        # -------- Camera: top-aft-left (Xmin, Ymin, Zmax) --------
        self._set_camera(ren, all_X, Y, all_Z)

        self._rw.Render()
        try:
            self._iren.ProcessEvents()
        except AttributeError:
            pass

    # ----------------------------------------
    # Private method - camera positioning
    # ----------------------------------------

    def _set_camera(
        self,
        ren   : object,
        all_X : np.ndarray,
        all_Y     : np.ndarray,
        all_Z : np.ndarray,
    ) -> None:
        '''
        Position camera at the top-aft-left corner (Xmin, Ymin, Zmax).

        Args:
            ren  : vtkRenderer instance.
            all_X: All X-coordinate values for bounds computation.
            Y    : Spanwise Y-coordinate array.
            all_Z: All Z-coordinate values for bounds computation.
        '''
        cx = float((all_X.min() + all_X.max()) * 0.60)
        cy = float((all_Y.min() + all_Y.max()) * 0.60)
        cz = float((all_Z.min() + all_Z.max()) * 0.60)
        span = float(max(
            all_X.max() - all_X.min(),
            all_Y.max() - all_Y.min(),
            all_Z.max() - all_Z.min(),
        ))
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(
            float(all_X.min()) - 0.50 * span,
            float(all_Y.min()) - 0.30 * span,
            float(all_Z.max()) + 0.35 * span,
        )
        cam.SetViewUp(0.0, 0.0, 1.0)
        ren.ResetCameraClippingRange()

    # ------------------------------------------------
    # Private static methods - VTK actor builders
    # ------------------------------------------------

    @staticmethod
    def _skin_actor(
        profiles : list,
        Y        : np.ndarray,
    ) -> object:
        '''
        Build a transparent gray surface from adjacent profile quads.

        Args:
            profiles: List of (N, 2) outer-skin point arrays per station.
            Y       : Spanwise Y coordinates.

        Returns:
            vtkActor for the outer skin surface.
        '''
        pts   = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        pid   = 0
        for j in range(len(Y) - 1):
            p0, p1 = profiles[j], profiles[j + 1]
            y0, y1 = float(Y[j]), float(Y[j + 1])
            m = min(len(p0), len(p1)) - 1
            for i in range(m):
                x0, z0 = float(p0[i,   0]), float(p0[i,   1])
                x1, z1 = float(p0[i+1, 0]), float(p0[i+1, 1])
                x2, z2 = float(p1[i+1, 0]), float(p1[i+1, 1])
                x3, z3 = float(p1[i,   0]), float(p1[i,   1])
                pts.InsertNextPoint(x0, y0, z0)
                pts.InsertNextPoint(x1, y0, z1)
                pts.InsertNextPoint(x2, y1, z2)
                pts.InsertNextPoint(x3, y1, z3)
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, pid)
                quad.GetPointIds().SetId(1, pid + 1)
                quad.GetPointIds().SetId(2, pid + 2)
                quad.GetPointIds().SetId(3, pid + 3)
                cells.InsertNextCell(quad)
                pid += 4

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.88, 0.88, 0.88)
        actor.GetProperty().SetOpacity(0.28)
        actor.GetProperty().SetEdgeVisibility(0)
        return actor

    @staticmethod
    def _section_outlines_actor(
        profiles : list,
        Y        : np.ndarray,
    ) -> object:
        '''
        Build a single actor with closed cross-section outline polylines
        at every spanwise station.

        Args:
            profiles: List of (N, 2) outer-skin point arrays per station.
            Y       : Spanwise Y coordinates (one per station).

        Returns:
            vtkActor containing one closed polyline per station.
        '''
        pts    = vtk.vtkPoints()
        cells  = vtk.vtkCellArray()
        offset = 0
        for prof, y in zip(profiles, Y):
            n = len(prof)
            yf = float(y)
            for x, z in prof:
                pts.InsertNextPoint(float(x), yf, float(z))
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n + 1)
            for i in range(n):
                pl.GetPointIds().SetId(i, offset + i)
            pl.GetPointIds().SetId(n, offset)
            cells.InsertNextCell(pl)
            offset += n

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.92, 0.92, 0.92)
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetOpacity(0.70)
        return actor

    @staticmethod
    def _spar_actor(
        Y          : np.ndarray,
        X1         : np.ndarray,
        Z1         : np.ndarray,
        X2         : np.ndarray,
        Z2         : np.ndarray,
        color      : tuple,
        alpha      : float,
        edge_color : Optional[tuple] = None,
    ) -> object:
        '''
        Build a quad-strip actor between two boom lines.

        Args:
            Y         : Spanwise Y coordinates.
            X1/Z1     : Chord/vertical coords of the first boom line.
            X2/Z2     : Chord/vertical coords of the second boom line.
            color     : (r, g, b) float triple in [0, 1].
            alpha     : Surface opacity.
            edge_color: When supplied, draw quad edges with this color.

        Returns:
            vtkActor for the spar web surface.
        '''
        pts   = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        pid   = 0
        for j in range(len(Y) - 1):
            p0x, p0z = float(X1[j]),   float(Z1[j])
            p1x, p1z = float(X1[j+1]), float(Z1[j+1])
            q0x, q0z = float(X2[j]),   float(Z2[j])
            q1x, q1z = float(X2[j+1]), float(Z2[j+1])
            y0,  y1  = float(Y[j]),    float(Y[j+1])
            pts.InsertNextPoint(p0x, y0, p0z)
            pts.InsertNextPoint(p1x, y1, p1z)
            pts.InsertNextPoint(q1x, y1, q1z)
            pts.InsertNextPoint(q0x, y0, q0z)
            quad = vtk.vtkQuad()
            for i in range(4):
                quad.GetPointIds().SetId(i, pid + i)
            cells.InsertNextCell(quad)
            pid += 4

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(alpha)
        if edge_color is not None:
            actor.GetProperty().SetEdgeVisibility(1)
            actor.GetProperty().SetEdgeColor(*edge_color)
        else:
            actor.GetProperty().SetEdgeVisibility(0)
        return actor

    @staticmethod
    def _line_actor(
        X     : np.ndarray,
        Y     : np.ndarray,
        Z     : np.ndarray,
        color : tuple,
        lw    : float = 1.5,
    ) -> object:
        '''
        Build a spanwise polyline actor.

        Args:
            X/Y/Z: Coordinate arrays along the line.
            color: (r, g, b) float triple in [0, 1].
            lw   : Line width in points.

        Returns:
            vtkActor for the polyline.
        '''
        pts = vtk.vtkPoints()
        for x, y, z in zip(X, Y, Z):
            pts.InsertNextPoint(float(x), float(y), float(z))

        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(len(X))
        for i in range(len(X)):
            pl.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(pl)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(lw)
        return actor

    @staticmethod
    def _add_axes(ren: object, bounds: tuple) -> None:
        '''
        Attach a vtkCubeAxesActor labelled X/Y/Z [mm] to the renderer.

        Args:
            ren   : vtkRenderer.
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) bounding box.
        '''
        axes = vtk.vtkCubeAxesActor()
        axes.SetBounds(*bounds)
        axes.SetCamera(ren.GetActiveCamera())
        axes.SetXTitle("X [mm]")
        axes.SetYTitle("Y [mm]")
        axes.SetZTitle("Z [mm]")
        for i in range(3):
            axes.GetTitleTextProperty(i).SetColor(1.0, 1.0, 1.0)
            axes.GetLabelTextProperty(i).SetColor(0.8, 0.8, 0.8)
        axes.GetXAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.GetYAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.GetZAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.SetFlyModeToOuterEdges()
        ren.AddActor(axes)

    @staticmethod
    def _title_actor(text: str) -> object:
        '''Return a 2D screen-space text actor at the bottom-left corner.'''
        ta = vtk.vtkTextActor()
        ta.SetInput(text)
        ta.GetTextProperty().SetFontSize(13)
        ta.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        ta.SetPosition(10, 10)
        return ta


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


# ========================================================= #
# --------------------------------------------------------- #
# ========================================================= #
#         /$$$$$$  /$$        /$$$$$$   /$$$$$$             #
#        /$$__  $$| $$       /$$__  $$ /$$__  $$            #
#       | $$  \__/| $$      |__/  \ $$| $$  \ $$            #
#       | $$      | $$         /$$$$$/| $$  | $$            #
#       | $$      | $$        |___  $$| $$  | $$            #
#       | $$    $$| $$       /$$  \ $$| $$  | $$            #
#       |  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/            #
#        \______/ |________/ \______/  \______/             #
# ========================================================= #
# --------------------------------------------------------- #
# ========================================================= #

if __name__ == "__main__":
    aircraft_name = "DA62"
    opt_name      = "OptTeste3"

    # ---------------- Set database specifications ----------------
    # Laminates are discovered by glob over MAT_*_LaminateData.json; the
    # MAT_ prefix (with underscore) selects the curated catalogue written
    # by composite_library.write_laminates and skips legacy test laminates
    # named MAT{int} (no underscore).
    skip_mat : list[str] = []

    materials_to_load = sorted(
        f.stem.removesuffix("_LaminateData")
        for f in _DFLT_MAT_DIR.glob("MAT_*_LaminateData.json")
        if f.stem.removesuffix("_LaminateData") not in skip_mat
    )

    airfoils_to_load = [
        "WortmannFX63137",
    ]

    database_loading_specs: list[object] = []
    database_loading_specs.append(
        # type_nbr = 1
        DatabaseSpec(WingData, _DFLT_WNG_DIR, f"{aircraft_name.lower()}"),
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
        opt_name       = opt_name,
        db_specs       = db_specs,
        pipeline_logging = False,
        de_hyperpar    = {**DE_HYPERPAR, 'NP': 16, 'k_max': 32},
    )
    runner.run(live_plot = True)
