'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Load Mapper module.

Transforms .txt (or a .xlsx) output file from XFLR5 into a set of loads for each
operational point. The external loads are saved into a JSON file named
./{aircraft_name}_ExLoadsData.json .

THIS MODULE ONLY SUPPORT SYMMETRIC LOAD CONDITIONS IN CURRENT ARCHITECTURE.
FEEL FREE TO ADD ROUTINES TO CALCULATE INTERNAL LOADS OF ASYMMETRIC LOAD
CONDITIONS AND COMMIT TO /github.com/cwss.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import warnings
from pathlib import Path
from typing import Optional, Any

from dataclasses import dataclass

import numpy as np

# ================ Default paths ================
from cl3o.paths import (
    LOADS_DIR    as _DFLT_LDS_DIR,
    OPPOINTS_DIR as _DFLT_OPP_DIR,
    WINGS_DIR    as _DFLT_WNG_DIR,
)

# ================ Module imports ================

# Const
from cl3o import Constants

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils.oppoints import OppData

# Geometry
from cl3o.geometry.wing import WingData

# ================ Global variables ================

# Number of decimal places
_N_DEC = 4

# Tolerance to assure external load condition is symmetric
_SYMMETRY_TOL = 0.05

_CONDITIONS = [
    "cruise",
]

_XFLR5_FILES = [
    "MainWing_a=0.00_v=87.97ms.txt",
]


# ================================================================================
# Data containers
# ================================================================================

@dataclass
class ExLoadsData:
    '''
    Container for storing external forces and moments acting on wing surface,
    as well as the position of each load, along CA = 0.25 * chord.

    This module expects multi-condition loads, each one containing:

    Property        Size        Description                             Units
    ------------    --------    ------------------------------------    --------
    n               (1,)        Number of nodes (half span)             -
    num_cond        (1,)        Number of load conditions               -

    conditions      (nc,)       List of conditions analysed             str

    X, Y, Z         (n,)        CA coordinates, full span                mm
    lift            (nc,n)      Distributed lift force, full span        N
    drag            (nc,n)      Distributed drag force, full span        N
    moment          (nc,n)      Distributed aero moment, full span       N*mm

    X, Y, Z *_hf    (n,)        CA coordinates, analyzed wing root->tip  mm
    lift_hf         (nc,n)      Distributed lift force, analyzed wing    N
    drag_hf         (nc,n)      Distributed drag force, analyzed wing    N
    moment_hf       (nc,n)      Distributed aero moment, analyzed wing   N*mm

    Obs.:   - 'nc' stands for 'num_cond'.
            - Forces and moment are nodal values (one per node).
            - Conditions list has one-to-one relation with each column
              of forces and moments.
            - ALL Raw Data must have same size.
            - '_hf' arrays slice the analyzed half-span (Constants.WING_SIDE)
              ordered root to tip, matching LerpWingData.Y_sta. For the right
              wing Y increases ~0 -> +b/2; for the left wing Y decreases
              ~0 -> -b/2.
    '''
    n        : int
    num_cond : int

    conditions : list[str]

    X  : np.ndarray
    Y  : np.ndarray
    Z  : np.ndarray

    lift   : np.ndarray
    drag   : np.ndarray
    moment : np.ndarray

    X_hf  : np.ndarray
    Y_hf  : np.ndarray
    Z_hf  : np.ndarray

    lift_hf   : np.ndarray
    drag_hf   : np.ndarray
    moment_hf : np.ndarray


@dataclass
class InLoadsData:
    '''
    Container for storing internal forces values for all load condition.

    The internal forces and moments are obtained by numerical integration
    along CA line. Stored as lists of size (n,nc)

    Property    Size        Description                                     Units
    --------    --------    ----------------------------------------    --------
    n           (1,)        Number of nodes                             -
    num_cond    (1,)        Number of load conditions                   -

    conditions  (nc,)       List of conditions analysed                 str
    
    X, Y, Z     (n,)        CA coordinates in global frame of ref.      mm
    Ny          (n,nc)      Axial force along ca_cma, Y-axis            N
    Vx, Vz      (n,nc)      Shear force along ca_cma, X and Z           N
    Mfx, Mfz    (n,nc)      Bending moment along ca_cma, X and Z        N*mm
    Mty         (n,nc)      Torque along ca_cma, Y-axis                 N*mm

    Obs.:   - 'nc' stands for 'num_cond'.
    
    Sign convention (global Cartesian frame, root = fixed end):
        Vz  > 0  :  shear in +Z (upward, lift dir.), both wings
        Vx  > 0  :  shear in +X (drag dir.), both wings
        Ny  = 0  :  no spanwise load for symmetric flight
        Mfx > 0  :  flapwise bending, right wing (Y > 0)
        Mfx < 0  :  flapwise bending, left wing  (Y < 0)
        Mfz < 0  :  edgewise bending, right wing (Y > 0)
        Mfz > 0  :  edgewise bending, left wing  (Y < 0)
        Mty     :   nose-up torsion (same sign, both wings)
    '''
    n : int
    num_cond : int

    conditions : list[str]

    X  : np.ndarray
    Y  : np.ndarray
    Z  : np.ndarray

    Ny : np.ndarray
    Vx : np.ndarray
    Vz : np.ndarray

    Mfx : np.ndarray
    Mfz : np.ndarray
    Mty : np.ndarray



# ================================================================================
# Internal Helpers
# ================================================================================

class LoadsHelper:

    def __init__(self):
        pass

    @staticmethod
    def check_input_db(
        opp_filepath : Path,
        wng_filepath : Path,
    ) -> None:
        '''
        Raise FileNotFoundError if either database file is missing.

        Args:
            opp_filepath: Resolved OppData JSON path.
            wng_filepath: Resolved WingData JSON path.
        '''
        if not opp_filepath.exists():
            raise FileNotFoundError(
                f"[CL3O] OppData file not found.\n"
                f"| Path : {opp_filepath}\n"
                f"Run BuildDatabase first to create the missing archive."
            )
        if not wng_filepath.exists():
            raise FileNotFoundError(
                f"[CL3O] WingData file not found.\n"
                f"| Path : {wng_filepath}\n"
                f"Run BuildDatabase first to create the missing archive."
            )

    @staticmethod
    def verify_symmetry(
        L_left: np.ndarray,
        L_right: np.ndarray,
    ) -> None:
        '''Verifies if the load condition is symmetric within tolerance.'''
        L_tot_l = np.sum(L_left)
        L_tot_r = np.sum(L_right)

        symmetry = (np.abs(1 - L_tot_l / L_tot_r) <= _SYMMETRY_TOL)

        if not symmetry:
            warnings.warn(
                f"[CL3O] Load condition is not symmetric within "
                f"tolerance of {_SYMMETRY_TOL * 100:.2f}%.",
                stacklevel=2,
            )


# ================================================================================
# Public API - External loads raw data processing
# ================================================================================

class LoadMapper:
    
    def __init__(
        self,
        aircraft_name: str,
        db_filepath: str | Path,
        conditions: Optional[list[str]] = None,
        xflr5_files: Optional[list[str]] = None,
        wing_side: str = Constants.WING_SIDE,
        enable_logging: bool = True,    # always last entry
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)
        self.wing_side = wing_side

        self.db_filepath = Path(db_filepath)
        self.inl_filepath = self.db_filepath.with_name(
            self.db_filepath.name.replace("ExLoadsData", "InLoadsData")
        )
        self._retrieve_input_data(aircraft_name=aircraft_name)
        self._raw_data_cache: dict[str, dict[str, np.ndarray]] = {}

        # Accumulators
        self.conditions_list : list[str] = []
        all_lift   : list[np.ndarray] = []
        all_drag   : list[np.ndarray] = []
        all_moment : list[np.ndarray] = []
        all_Ny     : list[np.ndarray] = []
        all_Vx     : list[np.ndarray] = []
        all_Vz     : list[np.ndarray] = []
        all_Mfx    : list[np.ndarray] = []
        all_Mfz    : list[np.ndarray] = []
        all_Mty    : list[np.ndarray] = []
        
        first_condition = True
        for condition, file in zip(
            conditions  or _CONDITIONS,
            xflr5_files or _XFLR5_FILES
        ):
            self.logger.info(f"Processing condition '{condition}' from file '{file}'")
            
            X, Y, Z, lift, drag, moment, Ny, Vx, Vz, Mfx, Mfz, Mty = (
                self._process_raw_data(condition, file)
            )

            if first_condition:
                self.X, self.Y, self.Z = X, Y, Z
                self.n = len(Y) // 2
                first_condition = False

            self.conditions_list.append(condition)
            all_lift.append(lift)
            all_drag.append(drag)
            all_moment.append(moment)
            all_Ny.append(Ny)
            all_Vx.append(Vx)
            all_Vz.append(Vz)
            all_Mfx.append(Mfx)
            all_Mfz.append(Mfz)
            all_Mty.append(Mty)

        self.num_cond = len(self.conditions_list)

        # Stack per-condition lines into (nc, n) matrices
        self.lift   = np.column_stack(all_lift).T
        self.drag   = np.column_stack(all_drag).T
        self.moment = np.column_stack(all_moment).T
        self.Ny     = np.column_stack(all_Ny).T
        self.Vx     = np.column_stack(all_Vx).T
        self.Vz     = np.column_stack(all_Vz).T
        self.Mfx    = np.column_stack(all_Mfx).T
        self.Mfz    = np.column_stack(all_Mfz).T
        self.Mty    = np.column_stack(all_Mty).T

        # Half-span slices, ordered root -> tip for the analyzed wing.
        # Y_full is ascending (left-tip -> right-tip), self.n = len // 2:
        #   right wing -> second half [n:], already root(0) -> tip(+b/2)
        #   left wing  -> first half  [:n], flipped to root(0) -> tip(-b/2)
        if self.wing_side == "right":
            half = slice(self.n, None)
            order = slice(None, None, 1)      # keep ascending Y (root -> tip)
        else:
            half = slice(0, self.n)
            order = slice(None, None, -1)     # flip to root -> tip (descending Y)

        self.X_hf  = self.X[half][order]
        self.Y_hf  = self.Y[half][order]
        self.Z_hf  = self.Z[half][order]
        self.lift_hf   = self.lift[:, half][:, order]
        self.drag_hf   = self.drag[:, half][:, order]
        self.moment_hf = self.moment[:, half][:, order]

        self.logger.info("Packing ExLoadsData and writing JSON database")
        self.exl_data = self._pack_exloads()
        self.inl_data = self._pack_inloads()

        io.write_json(
            obj=self.exl_data,
            filepath=self.db_filepath,
        )

        self.logger.info(f"External loads database successfully written to: {self.db_filepath}")

        io.write_json(
            obj=self.inl_data,
            filepath=self.inl_filepath,
        )

        self.logger.info(f"Internal loads database successfully written to: {self.inl_filepath}")
    

    # ----------------------------------------
    # Private methods - I/O
    # ----------------------------------------

    def _retrieve_input_data(
        self,
        aircraft_name: str,
    ) -> None:
        '''
        Create instances of operational points and wing data.
        
        The wing geometric values are in [mm].
        '''
        self.opp_filename = f"{aircraft_name.lower()}_OppData.json"
        self.wng_filename = f"{aircraft_name.lower()}_WingData.json"
        self.opp_filepath = _DFLT_OPP_DIR / self.opp_filename
        self.wng_filepath = _DFLT_WNG_DIR / self.wng_filename
        LoadsHelper.check_input_db(
            opp_filepath = self.opp_filepath,
            wng_filepath = self.wng_filepath,
        )

        self.opp_data = io.read_json(
            filepath=self.opp_filepath,
            dcls=OppData,
        )
        self.wng_data = io.read_json(
            filepath=self.wng_filepath,
            dcls=WingData,
        )

        self.b   = float(self.wng_data.b)
        self.mac = float(self.wng_data.mac)
        self.pos = np.array(self.wng_data.pos, dtype=float)
        self.x_le = np.array(self.wng_data.x_le, dtype=float)
        self.z_le = np.array(self.wng_data.z_le, dtype=float)
        self.chords = np.array(self.wng_data.chords, dtype=float)

    
    # ----------------------------------------
    # Private methods - Inner Calculations
    # ----------------------------------------

    @staticmethod
    def _strip_areas(
        y: np.ndarray,
        c: np.ndarray,
        b: float,
    ) -> np.ndarray:
        '''Computes the planform area of each spanwise strip.'''
        mid = 0.5 * (y[:-1] + y[1:])
        bound = np.insert(np.append(mid, b / 2.0), 0, -b / 2.0)
        delta_y = np.diff(bound)

        return delta_y * c
    

    @staticmethod
    def _cut_external_loads(
        mask: np.ndarray,
        X_full: np.ndarray,
        Y_full: np.ndarray,
        Z_full: np.ndarray,
        lift_full: np.ndarray,
        drag_full: np.ndarray,
        moment_full: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        '''Separete loads based on a mask delimiter.'''
        sort_idx = np.argsort(Y_full[mask])

        X = X_full[mask][sort_idx]
        Y = Y_full[mask][sort_idx]
        Z = Z_full[mask][sort_idx]
        lift   = lift_full[mask][sort_idx]
        drag   = drag_full[mask][sort_idx]
        moment = moment_full[mask][sort_idx]

        return X, Y, Z, lift, drag, moment
    
    
    def _compute_ca(
        self,
        y_span: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolates CA (quarter-chord line) 3-D coordinates at each station.

        The wing database stores geometry for the right half-span (0 to b/2).
        Stations on the left half are handled by folding with np.abs().
        '''
        y_abs = np.abs(y_span)

        x_le_i  = np.interp(y_abs, self.pos, self.x_le)
        z_le_i  = np.interp(y_abs, self.pos, self.z_le)
        chord_i = np.interp(y_abs, self.pos, self.chords)

        X = (x_le_i + 0.25 * chord_i)
        Y = y_span
        Z = z_le_i

        return X, Y, Z


    def _compute_force_distribution(
        self,
        loads: dict,
        q: float
    ) -> tuple[Any, ...]:
        '''
        Converts XFLR5 aerodynamic coefficients into distributed nodal forces.

        Notes
        --------
        - Strip area Si [mm^2] * q [MPa] * Cl [-] --> force  in [N].
        - Moment reference length is the MAC [mm].
        '''
        y  = loads["y-span"] * 1000     # [mm]
        c  = loads["Chord"] * 1000      # [mm]
        cl = loads["Cl"]
        cd = loads["PCd"] + loads["ICd"]
        cm = loads["CmAirf@chord/4"]

        mac = self.mac
        b   = self.b

        Si = self._strip_areas(y, c, b)

        lift   = q * cl * Si
        drag   = q * cd * Si
        moment = q * cm * Si * mac

        return y, lift, drag, moment


    @staticmethod
    def _half_internal_forces(
        X  : np.ndarray,
        Y  : np.ndarray,
        Z  : np.ndarray,
        L  : np.ndarray,
        D  : np.ndarray,
        Ma : np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        '''
        Computes six internal force and moment components for one wing half
        using outboard-to-root suffix sums (O(n)).

        Input arrays must be sorted root-to-tip (index 0 = root station,
        last index = tip station): ascending Y for the right wing, descending
        Y for the left wing.

        At section i the resultants equal the sum over all outboard stations j:

            Vz [i]  =   L_j
            Vx [i]  =   D_j
            Mfx[i]  =   L_j*(Y_j-Y_i)  =  sfx_LY - Y_i*Vz
            Mfz[i]  = - D_j*(Y_j-Y_i)  = -(sfx_DY - Y_i*Vx)
            Mty[i]  =   M_j - L_j*(X_j-X_i) + D_j*(Z_j-Z_i)

        Returns:
            Tuple (Ny, Vx, Vz, Mfx, Mfz, Mty), each of shape (n,).
        '''
        Vz = np.cumsum(L[::-1])[::-1]
        Vx = np.cumsum(D[::-1])[::-1]
        Ny = np.zeros_like(Vz)

        sfx_LY = np.cumsum((L * Y)[::-1])[::-1]
        sfx_DY = np.cumsum((D * Y)[::-1])[::-1]
        sfx_LX = np.cumsum((L * X)[::-1])[::-1]
        sfx_DZ = np.cumsum((D * Z)[::-1])[::-1]
        sfx_M  = np.cumsum(Ma[::-1])[::-1]

        Mfx = sfx_LY - Y * Vz
        Mfz = -(sfx_DY - Y * Vx)
        Mty = sfx_M - (sfx_LX - X * Vz) + (sfx_DZ - Z * Vx)

        return Ny, Vx, Vz, Mfx, Mfz, Mty


    def _compute_internal_forces(
        self,
        X_full      : np.ndarray,
        Y_full      : np.ndarray,
        Z_full      : np.ndarray,
        lift_full   : np.ndarray,
        drag_full   : np.ndarray,
        moment_full : np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        '''
        Integrates aerodynamic loads outboard-to-root on each wing half,
        yielding the six internal force and moment components at every
        spanwise station.

        Both wings are treated independently as cantilever beams (root =
        fixed end, tip = free end). The reaction at the root equals the sum
        of all distributed loads; integrating inward from the tip via suffix
        sums gives the correct global-frame resultants at each section i:

            Vz [i]  =   L_j
            Vx [i]  =   D_j
            Ny [i]  =   0

            Mfx[i]  =   L_j*(Y_j-Y_i)  =  sfx_LY  - Y_i*Vz[i]
            Mfz[i]  = - D_j*(Y_j-Y_i)  = -[sfx_DY - Y_i*Vx[i]]
            Mty[i]  =   M_j - L_j*(X_j-X_i) + D_j*(Z_j-Z_i)
                    =   sfx_M
                        - [sfx_LX - X_i*Vz[i]]
                        + [sfx_DZ - Z_i*Vx[i]]

        Each bracketed term is a scalar suffix sum (O(n)).

        Global-frame sign convention of the returned arrays:
            Vz, Vx  > 0  both wings (symmetric under positive lift/drag)
            Mfx     > 0  right wing (Y > 0);  < 0  left wing (Y < 0)
            Mfz     < 0  right wing (Y > 0);  > 0  left wing (Y < 0)
            Mty          same sign on both wings

        The returned arrays cover the full span in ascending Y order
        (left-tip to right-tip), aligned with Y_full.
        '''
        # -------- Separate wings --------
        left_mask  = Y_full <= 0.0
        right_mask = Y_full > 0.0

        # Right wing: ascending Y (root ~0 -> tip Y_max)
        X_R, Y_R, Z_R, L_R, D_R, Ma_R = self._cut_external_loads(
            right_mask, X_full, Y_full, Z_full, lift_full, drag_full, moment_full
        )

        # Left wing: _cut returns ascending Y (tip Y_min -> root ~0).
        # Reverse to root-to-tip (descending Y) so suffix sums
        # integrate from the free end inward, same as the right wing.
        X_La, Y_La, Z_La, L_La, D_La, Ma_La = self._cut_external_loads(
            left_mask, X_full, Y_full, Z_full, lift_full, drag_full, moment_full
        )
        X_L  = X_La[::-1]
        Y_L  = Y_La[::-1]
        Z_L  = Z_La[::-1]
        L_L  = L_La[::-1]
        D_L  = D_La[::-1]
        Ma_L = Ma_La[::-1]

        LoadsHelper.verify_symmetry(L_L, L_R)

        # -------- Internal forces, per wing --------
        Ny_R, Vx_R, Vz_R, Mfx_R, Mfz_R, Mty_R = self._half_internal_forces(
            X_R, Y_R, Z_R, L_R, D_R, Ma_R
        )
        Ny_L, Vx_L, Vz_L, Mfx_L, Mfz_L, Mty_L = self._half_internal_forces(
            X_L, Y_L, Z_L, L_L, D_L, Ma_L
        )

        # -------- Full span, ascending Y (left-tip to right-tip) --------
        # Left-wing arrays are root-to-tip (descending Y); flip back to
        # ascending (tip-to-root) to align with Y_full ordering.
        return (
            np.concatenate((Ny_L[::-1],  Ny_R)),
            np.concatenate((Vx_L[::-1],  Vx_R)),
            np.concatenate((Vz_L[::-1],  Vz_R)),
            np.concatenate((Mfx_L[::-1], Mfx_R)),
            np.concatenate((Mfz_L[::-1], Mfz_R)),
            np.concatenate((Mty_L[::-1], Mty_R)),
        )


    # ----------------------------------------
    # Private methods - ExLoads definitions
    # ----------------------------------------

    def _process_raw_data(
        self,
        condition: list[str],
        file: list[str],
    ) -> tuple[np.ndarray, ...]:
        '''
        Full processing pipeline for a single flight condition.

        Steps
        -----
        1. Read XFLR5 output file (.txt or .xlsx).
        2. Retrieve dynamic pressure from the operational point database.
        3. Compute distributed forces over the full span.
        5. Interpolate CA coordinates.
        6. Integrate to obtain internal forces and moments.
        
        Units matchs the folowing:
            Length .... mm
            Force ..... N
            Moment .... N*mm
            Pressure .. MPa
            Density ... ton/mm^3 (1 ton = 1000 kg)
            Velocity .. mm/s
        '''
        # -------- 1. Read XFLR5 file --------
        xflr5_loads_filepath = _DFLT_LDS_DIR / f"{file}"

        if file in self._raw_data_cache:
            self.logger.debug(f"Using cached raw data for file '{file}'")
            cached = self._raw_data_cache[file]

            loads = cached
        else:
            self.logger.debug(f"'{file}' not found in cache. Reading from raw data.")

            if file.endswith(".txt"):
                loads = io.read_txt(
                    filepath=xflr5_loads_filepath,
                )
            elif file.endswith(".xlsx"):
                loads = io.read_xlsx(
                    filepath=xflr5_loads_filepath,
                )
            else:
                raise ValueError(
                    f"Unsupported file extension for '{file}'."
                    f"Expected '.txt' or '.xlsx'."
                    f"Aborting program execution with erro num. : {hex(id(self))}."
                )
        
        self._raw_data_cache[file] = loads

        # -------- 2. Flight condition -> Dynamic pressure [MPa] --------
        try:
            idx = list(self.opp_data.cond_tags).index(condition)
        except ValueError:
            raise ValueError(
                f"[CL3O] Condition '{condition}' not found in OppData.\n"
                f"| available : {list(self.opp_data.cond_tags)}\n"
                f"Regenerate OppData with BuildDatabase.create_opp_db."
            )
        rho = float(self.opp_data.rho[idx])      * 1.0e-12  # kg/m^3 -> t/mm^3
        v   = float(self.opp_data.velocity[idx]) * 1.0e3    # m/s    -> mm/s
        q   = 0.5 * rho * v**2                              # [MPa]

        # -------- 3. Distributed forces (full span) --------
        y_span, lift, drag, moment = self._compute_force_distribution(loads, q)

        # -------- 4. CA node coordinates [mm] --------
        X, Y, Z = self._compute_ca(y_span)

        # -------- 5. Internal forces [N], [N*mm] --------
        Ny, Vx, Vz, Mfx, Mfz, Mty = self._compute_internal_forces(
            X, Y, Z, lift, drag, moment
        )

        return X, Y, Z, lift, drag, moment, Ny, Vx, Vz, Mfx, Mfz, Mty


    # ----------------------------------------
    # Private method - ExLoadsData packing
    # ----------------------------------------

    def _pack_exloads(self) -> ExLoadsData:
        '''
        Transforms self instances into an ExLoadsData dataclass.

        Internal loads data (Ny, Vx, Vz, Mfx, Mfz, Mty) is skipped.
        '''
        return ExLoadsData(
            n        = int(self.n),
            num_cond = int(self.num_cond),

            conditions = self.conditions_list,

            X  = np.round(self.X.astype(float), _N_DEC),
            Y  = np.round(self.Y.astype(float), _N_DEC),
            Z  = np.round(self.Z.astype(float), _N_DEC),

            lift   = np.round(self.lift.astype(float), _N_DEC),
            drag   = np.round(self.drag.astype(float), _N_DEC),
            moment = np.round(self.moment.astype(float), _N_DEC),

            X_hf  = np.round(self.X_hf.astype(float), _N_DEC),
            Y_hf  = np.round(self.Y_hf.astype(float), _N_DEC),
            Z_hf  = np.round(self.Z_hf.astype(float), _N_DEC),

            lift_hf   = np.round(self.lift_hf.astype(float), _N_DEC),
            drag_hf   = np.round(self.drag_hf.astype(float), _N_DEC),
            moment_hf = np.round(self.moment_hf.astype(float), _N_DEC),
        )

    def _pack_inloads(self) -> InLoadsData:
        '''
        Transforms self instances into an InLoadsData dataclass.

        External loads data (lift, drag, moment) is skipped.
        '''
        return InLoadsData(
            n        = int(self.n),
            num_cond = int(self.num_cond),

            conditions = self.conditions_list,

            X  = np.round(self.X.astype(float), _N_DEC),
            Y  = np.round(self.Y.astype(float), _N_DEC),
            Z  = np.round(self.Z.astype(float), _N_DEC),

            Ny  = np.round(self.Ny.astype(float), _N_DEC),
            Vx  = np.round(self.Vx.astype(float), _N_DEC),
            Vz  = np.round(self.Vz.astype(float), _N_DEC),

            Mfx = np.round(self.Mfx.astype(float), _N_DEC),
            Mfz = np.round(self.Mfz.astype(float), _N_DEC),
            Mty = np.round(self.Mty.astype(float), _N_DEC),
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    aircraft_name = "da62"

    conditions_to_analyse  = [
        "VC_n1pos",
        # "VS1_n1pos",
        # "VA_nMax",
        # "VC_nMax",
        # "VD_nMax",
        # "VS2_n1neg",
    ]
    xflr5_files_to_analyse = [
        "MainWing_a=0.00_v=87.97ms.txt",
        # "MainWing_a=0.00_v=87.97ms.txt",
        # "MainWing_a=0.00_v=87.97ms.txt",
        # "MainWing_a=0.00_v=87.97ms.txt",
        # "MainWing_a=0.00_v=87.97ms.txt",
        # "MainWing_a=0.00_v=87.97ms.txt",
    ]

    LoadMapper(
        aircraft_name=aircraft_name,
        db_filepath=_DFLT_LDS_DIR / f"{aircraft_name}_ExLoadsData.json",
        # db_filepath=_DFLT_LDS_DIR / f"{aircraft_name}_ExLoadsData_TESTE.json",
        conditions=conditions_to_analyse,
        xflr5_files=xflr5_files_to_analyse,
    )
