'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Wing Module.

Create a new wing based on minimal geometrical data.
Saves into a JSON file named ./{aircraft_name}_WingData.json .

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Any

from dataclasses import dataclass, asdict, field

import numpy as np

# ================ Default Database Paths ================
from cl3o.paths import AIRFOILS_DIR as _DFLT_AFL_DIR, WINGS_DIR as _DFLT_WNG_DIR

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils import math_utils as mthu

# Geometry
from cl3o.geometry.airfoil import Airfoil, AirfoilData

# ================ Global variables ================
_N_DEC = 4



# ================================================================================
# Data persistence - Wing data container
# ================================================================================

@dataclass
class WingData:
    '''
    Container for storing external geometric paremeters of the wing.

    Property    Size          Description                                     Unit
    --------    ----------    ----------------------------------------    --------
    n_cpts      (1,)          Number of control points (cpts)             -        

    span        (1,)          Wing span                                   mm       
    cr          (1,)          Root chord value                            mm       
    root_off    (1,)          Spanwise position where root really is      mm       
    afl_lst     (n_cpts,)     Airfoil name at each cpt                    str  

    area        (1,)          Wing planform area                          mm^2     
    AR          (1,)          Wing aspect ratio                           -        
    mgc         (1,)          Mean geometric chord                        mm       
    mac         (1,)          Mean aerodynamic chord                      mm       

    taper       (n_cpts,)     Taper ratio at cpts (c / cr)                % [0; 1] 
    pci         (n_cpts,)     Percentage of semi-span of each cpt         % [0; 1] 
    chords      (n_cpts,)     Chord value at each cpt                     mm       
    pos         (n_cpts,)     Position of cpts along spanwise axis        mm       

    twist       (n_cpts,)     Twist angle at each control point           degree   
    sweep       (n_cpts,)     Sweep angle relative to 1/4 chord           degree   
    sweep_LE    (n_cpts,)     Sweep angle relative to leading edge        degree   
    dihedral    (n_cpts,)     Dihedral (+) / Anhedral (-) angle           degree   

    ds          (n_cpts,)     Chord variation between adjacent points     mm       
    dc          (n_cpts,)     Length of each wing section                 mm       
    sec_span    (n_cpts,)     Total length of analyzed wing section       mm       

    x_le        (n_cpts,)     LE x-coordinate at each station (3D)        mm
    z_le        (n_cpts,)     LE z-coordinate at each station (3D)        mm
    x_te        (n_cpts,)     TE x-coordinate at each station (3D)        mm
    z_te        (n_cpts,)     TE z-coordinate at each station (3D)        mm       
    '''
    # Number of control points
    n_cpts : int

    # Main atributes
    b        : float
    cr       : float
    root_off : float
    afl_lst  : list[str]

    # Main geometrical properties
    area : float
    AR   : float
    mgc  : float
    mac  : float

    # Linear geometric values per station
    taper  : np.ndarray
    pci    : np.ndarray
    chords : np.ndarray
    pos    : np.ndarray
    
    # Angular geometric values per station
    twist    : np.ndarray
    sweep    : np.ndarray
    sweep_LE : np.ndarray
    dihedral : np.ndarray

    # Differential geometric values
    ds       : np.ndarray
    dc       : np.ndarray
    sec_span : np.ndarray

    # Wing outline
    x_le : np.ndarray
    z_le : np.ndarray
    x_te : np.ndarray
    z_te : np.ndarray


# -------- Interpolated WingData --------
@dataclass
class LerpWingData:
    '''
    Stores interpolated wing outline coordinates for the LEFT wing only.

    All spanwise arrays are ordered root -> tip with Y decreasing from
    ~0 down to ~-b/2 (left-wing convention).

    Property    Size            Description                             Unit
    --------    ------------    ------------------------------------    --------
    n_sta       (1,)            Number of left-wing spanwise stations   -
    Y_cp        (n_cpts,)       Spanwise control points location        mm
    Y_sta       (n_sta,)        Left-wing stations (Y<=0, root->tip)    mm
    LE          (n_sta,3)       Leading edge coordinates                mm
    TE          (n_sta,3)       Trailing edge coordinates               mm
    chord       (n_sta,)        Chord length of each station            mm
    twist       (n_sta,)        Twist value of each station             rad
    afl_list    (n_cpts,)       Copy of WingData.afl_list               -
    '''
    n_sta : int = 0

    Y_cp    : np.ndarray = field(default_factory=lambda: np.array([]))
    Y_sta   : np.ndarray = field(default_factory=lambda: np.array([]))
    LE      : np.ndarray = field(default_factory=lambda: np.array([]))
    TE      : np.ndarray = field(default_factory=lambda: np.array([]))
    chord   : np.ndarray = field(default_factory=lambda: np.array([]))
    twist   : np.ndarray = field(default_factory=lambda: np.array([]))

    afl_lst : list[str]  = field(default_factory=list)



# ================================================================================
# Internal Helpers
# ================================================================================

class WingHelper:
    def __init__(self):
        pass
    

    @staticmethod
    def validate_inputs(obj) -> None:
        '''Validates input data.'''
        arrays_to_check = {
            'taper': obj.taper,
            'twist': obj.twist,
            'sweep': obj.sweep,
            'dihedral': obj.dihedral,
            'afl_lst': obj.afl_lst
        }
        
        for name, array in arrays_to_check.items():
            if len(array) != obj.n_cpts:
                raise ValueError(
                    f"Input data must have same size."
                    f"Expected: {obj.n_cpts} (based on pci array),"
                    f"Found: {len(array)} in {name}."
                )
        
        if not np.all(np.diff(obj.pci) > 0):
            raise ValueError("'pci' values must be in ascending ordem.")
    

    @staticmethod
    def lerp_from_data(
        wng_data : WingData,
        Y_sta    : np.ndarray,
    ) -> LerpWingData:
        '''
        Build a LerpWingData by interpolating a WingData outline at the
        supplied spanwise stations. Pure helper - takes WingData rather
        than a live Wing instance so it can be called from anywhere
        that has access to the persisted dataclass.

        The output is always restricted to the LEFT wing (Y <= 0) and
        stacked root -> tip (Y decreasing from ~0 down to ~-b/2);
        full-span or unordered inputs are folded onto that convention
        internally so LE, TE, chord and twist are consistently aligned.

        Args:
            wng_data: WingData loaded from JSON or freshly built.
            Y_sta   : Spanwise station positions [mm]; full-span or
                      left-wing only, any ordering accepted.

        Returns:
            LerpWingData with LE/TE 3-D coordinates, chord and twist at
            each left-wing station, ordered root -> tip.
        '''
        Y_in = np.asarray(Y_sta, dtype=float)
        Y_cp = np.asarray(wng_data.pos, dtype=float)

        # Fold to left-wing, root -> tip (|Y| ascending, Y decreasing).
        Y_abs = np.sort(np.abs(Y_in[Y_in <= 0.0]))
        Y_sta = -Y_abs

        n_sta = int(Y_sta.shape[0])

        def _i(arr: np.ndarray) -> np.ndarray:
            return np.interp(Y_abs, Y_cp, np.asarray(arr, dtype=float))

        x_le = _i(wng_data.x_le)
        z_le = _i(wng_data.z_le)
        x_te = _i(wng_data.x_te)
        z_te = _i(wng_data.z_te)

        LE    = np.column_stack([x_le, Y_sta, z_le])
        TE    = np.column_stack([x_te, Y_sta, z_te])
        chord = np.linalg.norm(TE - LE, axis=1)
        twist = _i(np.radians(wng_data.twist))

        afl_lst = [
            (p[:-4] if str(p).lower().endswith(".dat") else str(p)).lower()
            for p in wng_data.afl_lst
        ]

        return LerpWingData(
            n_sta   = n_sta,
            Y_cp    = Y_cp,
            Y_sta   = Y_sta,
            LE      = LE,
            TE      = TE,
            chord   = chord,
            twist   = twist,
            afl_lst = afl_lst,
        )


    @staticmethod
    def check_missing_airfoil_data(obj) -> None:
        '''Crate database for missing profiles'''
        for profile in obj.afl_lst:
            stem = profile[:-4] if profile.endswith(".dat") else profile
            afl_filepath = _DFLT_AFL_DIR / f"{stem.lower()}_AirfoilData.json"

            if not afl_filepath.exists():
                obj.logger.info(f"Profile {stem} not found in Database."
                                 f"Creating new AirfoilData for {stem}")

                Airfoil(
                    filename=stem,
                    db_filepath=afl_filepath,
                )



# ================================================================================
# Public API - Define external geometry of the wing
# ================================================================================

class Wing:
    
    def __init__(
        self,
        wing_specs: dict[str, Any],
        db_filepath: str | Path,
        enable_logging: bool = True,
    ) -> None:
        '''
        Initiates
        '''
        self.logger = io.setup_logger(self, enable_logging)

        # Self instances
        self.db_filepath = Path(db_filepath)

        self.span     = float(wing_specs["span"])
        self.cr       = float(wing_specs["cr"])
        self.taper    = np.asarray(wing_specs["taper"]   , dtype=float)
        self.pci      = np.asarray(wing_specs["pci"]     , dtype=float)
        self.twist    = np.asarray(wing_specs["twist"]   , dtype=float)
        self.sweep    = np.asarray(wing_specs["sweep"]   , dtype=float)
        self.dihedral = np.asarray(wing_specs["dihedral"], dtype=float)
        self.afl_lst  = wing_specs["airfoil"]

        # Validation
        self.n_cpts = len(self.pci)
        WingHelper.validate_inputs(self)

        if not self.db_filepath.exists():
            # Build wing data
            self.logger.info(f"Building Wing Database file")
            self.wng_data = self._define_wing_data()

            # Save to database
            self.logger.info(f"Writing JSON Wing Database file")

            io.write_json(
                obj=self.wng_data,
                filepath=self.db_filepath,
            )

            self.logger.info(f"Wing Database successfully writen to: {self.db_filepath}")
        else:
            self.wng_data = io.read_json(
                filepath=self.db_filepath,
                dcls=WingData,
            )
    
    
    # ----------------------------------------
    # Private methods - inner calculations
    # ----------------------------------------

    def _calculate_chords(self) -> np.ndarray:
        '''Calculates chord at each control point.'''
        return self.cr * self.taper
    
    def _calculate_sw_pos(self) -> np.ndarray:
        '''Determines y position of each control point.'''
        return 0.5 * self.pci * self.span
    
    def _get_root_offset(self) -> float:
        '''
        Determines the offset of wing root.

        In other words, it considers the first segment of wing
        inside the fuselage, whenever the pci starts with 0.
        Otherwise, it returns the first element of spanwise position.
        '''
        return self.pos[0] if self.pci[0] == 0 else self.pos[1]
    
    def _calculate_delta_span(self) -> np.ndarray:
        '''Calculates length of each section.'''
        return 0.5 * np.diff(self.pci) * self.span
    
    def _calculate_delta_chord(self) -> np.ndarray:
        '''Calculates the deviation of chord values.'''
        return -np.diff(self.chords)
    
    def _calculate_sec_span(self):
        '''Calculates the equivalent span value of analyzed wing section.'''
        return (self.pci[-1] - self.pci[0]) * self.span
    
    def _calculate_area(self) -> float:
        '''Calculates wing planform area.'''
        ds_aft = np.insert(self.ds, 0, 0)
        ds_fwd = np.append(self.ds, 0)

        return float(np.sum(self.chords * (ds_aft + ds_fwd)))
    
    def _calculate_sweep_LE(self) -> np.ndarray:
        '''Calculates leading edge angle with y-axis.'''
        sweep_LE = np.zeros(self.n_cpts)
        
        for i in range(self.n_cpts - 1):
            aux1 = ((self.chords[i] - self.chords[i+1]) * 0.25) / self.ds[i]
            aux2 = np.tan(np.radians(self.sweep[i+1]))
            sweep_LE[i + 1] = np.atan(aux1 + aux2)
        
        return np.degrees(sweep_LE)
    
    def _calculate_aspect_ratio(self) -> float:
        '''Calculates wing aspect ratio.'''
        return float(self.span**2 / self.area)
    
    def _calculate_aspect_ratio_sec(self) -> float:
        '''Calculates wing aspect ratio of analyzed wing section.'''
        return float((self.sec_span)**2 / self.area)
    
    def _calculate_mgc(self) -> float:
        '''Calculates mean geometric chord of full wing.'''
        return float(self.area / self.span)
    
    def _calculate_mgc_half_sec(self) -> float:
        '''Calculates mean geometric chord of analyzed wing section.'''
        return float(self.area / self.sec_span)
    
    def _calculate_mac(self) -> float:
        '''Calculates mean aerodynamic chord of full wing.'''
        return float(
            2.0 * mthu.integrate_piecewise_squared(self.chords, self.pos)
            / self.area
        )
    
    def _calculate_edge_points(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Calculate LE and TE points at each cpt.'''
        x_le = np.zeros(self.n_cpts)
        z_le = np.zeros(self.n_cpts)

        for i in range(1, self.n_cpts):
            ds_prev = self.ds[i - 1]
            Lambda  = np.radians(self.sweep_LE[i - 1])
            Gamma   = np.radians(self.dihedral[i - 1])
            x_le[i] = x_le[i - 1] + ds_prev * np.tan(Lambda)
            z_le[i] = z_le[i - 1] + ds_prev * np.tan(Gamma)

        twist_rad = np.radians(self.twist)
        x_te = x_le + self.chords * np.cos(twist_rad)
        z_te = z_le - self.chords * np.sin(twist_rad)

        return x_le, z_le, x_te, z_te
    

    # ----------------------------------------
    # Private method - Building database
    # ----------------------------------------

    def _pack_wing_data(self) -> WingData:
        '''Transforms self instances into WingData.'''
        packed_wng_data = WingData(
            n_cpts   = int(self.n_cpts),

            b     = round(float(self.span), _N_DEC),
            cr       = round(float(self.cr), _N_DEC),
            root_off = round(float(self.root_off), _N_DEC),
            afl_lst  = self.afl_lst,
            
            area     = round(float(self.area), _N_DEC),
            AR       = round(float(self.AR), _N_DEC),
            mgc      = round(float(self.mgc), _N_DEC),
            mac      = round(float(self.mac), _N_DEC),

            taper    = np.round(self.taper.astype(float), _N_DEC),
            pci      = np.round(self.pci.astype(float), _N_DEC),
            chords   = np.round(self.chords.astype(float), _N_DEC),
            pos      = np.round(self.pos.astype(float), _N_DEC),
            twist    = np.round(self.twist.astype(float), _N_DEC),
            sweep    = np.round(self.sweep.astype(float), _N_DEC),
            sweep_LE = np.round(self.sweep_LE.astype(float), _N_DEC),
            dihedral = np.round(self.dihedral.astype(float), _N_DEC),

            ds       = np.round(self.ds.astype(float), _N_DEC),
            dc       = np.round(self.dc.astype(float), _N_DEC),
            sec_span = np.round(self.sec_span.astype(float), _N_DEC),

            x_le     = np.round(self.x_le.astype(float), _N_DEC),
            z_le     = np.round(self.z_le.astype(float), _N_DEC),
            x_te     = np.round(self.x_te.astype(float), _N_DEC),
            z_te     = np.round(self.z_te.astype(float), _N_DEC),
        )
        return packed_wng_data


    def _define_wing_data(self) -> WingData:
        '''
        Main database builder pipeline.

        Do internal geometric calculations sequentially and returns
        the WingData dataclass for analysed wing.
        '''
        # Check for airfoil data
        WingHelper.check_missing_airfoil_data(self)

        # Calculate secondary parameters
        self.chords   = self._calculate_chords()
        self.pos      = self._calculate_sw_pos()
        self.root_off = self._get_root_offset()
        self.ds       = self._calculate_delta_span()
        self.dc       = self._calculate_delta_chord()
        self.area     = self._calculate_area()
        self.sec_span = self._calculate_sec_span()
        
        # Calculate tertiary parameters
        self.sweep_LE = self._calculate_sweep_LE()

        if self.pci[0] != 0:
            self.AR  = self._calculate_aspect_ratio_sec()
            self.mgc = self._calculate_mgc_half_sec()
        else:
            self.AR  = self._calculate_aspect_ratio()
            self.mgc = self._calculate_mgc()
        
        self.mac = self._calculate_mac()

        # Wing outline
        self.x_le, self.z_le, self.x_te, self.z_te = self._calculate_edge_points()

        return self._pack_wing_data()
    

    # ----------------------------------------
    # Public method - Wing interpolation
    # ----------------------------------------

    def lerp_wing_geometry(
        self,
        Y_sta: np.ndarray,
    ) -> LerpWingData:
        '''
        Interpolate wing outline at each station along the span.

        The returned dataclass is restricted to the LEFT wing (Y <= 0)
        and stacked root -> tip; full-span or unordered Y_sta inputs
        are folded onto that convention internally.

        Args:
            Y_sta: Spanwise station positions [mm]; full-span or left-
                   -wing only, any ordering accepted.

        Returns:
            LerpWingData with LE/TE 3-D coordinates, chord and twist
            at each left-wing station, ordered root -> tip.
        '''
        return WingHelper.lerp_from_data(self.wng_data, Y_sta)



# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    aircraft_name = "da62"

    wing_specs = {
        # LS planform span
        "span": 12969.0,

        # LS chord value at root
        "cr": 1728.0,

        # Taper ratios at each spanwise control point (cpt)
        "taper": np.array([1.0,
                           1.0, 
                           1316.0 / 1728.0,
                           915.0 / 1728.0]),
        
        # Non-dimensional spanwise positions of cpt
        "pci": np.array([0.0,
                         623.0 / 6484.0,
                         2035.0 / 6484.0,
                         1.0]),
        
        # Chordline twist of each cpt relative to root
        "twist": np.array([0.0, 0.0, 0.0, 0.0]),

        # CA sweep value BEFORE each cpt
        "sweep": np.array([0.0, 6.6, -0.2, -0.2]),

        # Dihedral (+) / Anhedral (-) values BEFORE each cpt
        "dihedral": np.array([0.0, 0.0, 5.2, 5.2]),

        # Airfoil file of each cpt
        "airfoil": [
            "WortmannFx63137.dat",
            "WortmannFx63137.dat",
            "WortmannFx63137.dat",
            "WortmannFx63137.dat",
        ],
    }

    Wing(
        wing_specs=wing_specs,
        db_filepath=_DFLT_WNG_DIR / f"{aircraft_name}_WingData.json"
    )
