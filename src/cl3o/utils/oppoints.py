'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Operational Points module.

Organizes atmospheric parameters and V-n envelope into a single JSON file
named ./{aircraft_name}_OppData.json .

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Optional, Any

from dataclasses import dataclass, asdict, field

import numpy as np

# ================ Default Database Paths ================
from cl3o.paths import OPPOINTS_DIR as _DFLT_OPP_DIR

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils.atmosisa import atmosisa
from cl3o.utils.convlength import convlength
from cl3o.utils.convvel import convvel

# ================ Global variables ================
_N_DEC = 8


# ================================================================================
# Data persistence - < MODULE > data containers
# ================================================================================

@dataclass
class OppData:
    '''
    Container for storing the Operational Points.

    Property        Size        Description                                 Units
    ------------    --------    ------------------------------------    --------
    aircraft        (1,)        Aircraft name                           str
    num_cond        (1,)        Number of conditions stored             -
    cond_tags       (nc,)       List of condition tags                  str
    velocity        (nc,)       List of airplane velocity               m/s
    load_factor     (nc,)       List of vertical load factors           -
    altitude        (nc,)       List of flight altitude-density [H]     m
    T               (nc,)       ISA Air temperature @ H                 K
    a               (nc,)       ISA Sound speed @ H                     m/s
    P               (nc,)       ISA Air pressure @ H                    Pa
    rho             (nc,)       ISA Air density @ H                     kg/m^3
    mu              (nc,)       ISA Dynamic viscosity @ H               N*s/m^2
    nu              (nc,)       ISA Kinematic viscosity @ H             m^2/s

    input_units     (2,)        [Optional. Default = SI]
                                Units of [0]: velocity                  str
                                         [1]: flight altitude

    output_units    (2,)        [Optional. Default = SI]

    Obs.:   - 'nc' stands for num_cond.
            - Stores all values in SI.
            - When input data (Vel., Alt.) is not in SI, the
              units must be given. Pay attention !!
    '''

    aircraft  : str
    num_cond  : int
    cond_tags : list[str]

    velocity    : np.ndarray
    load_factor : np.ndarray
    altitude    : np.ndarray

    T   : np.ndarray
    a   : np.ndarray
    P   : np.ndarray
    rho : np.ndarray
    nu  : np.ndarray
    mu  : np.ndarray

    input_units  : list[str] = field(default_factory=lambda: ["m/s", "m"])
    output_units : list[str] = field(default_factory=lambda: ["m/s", "m"])


# ================================================================================
# Public API - Pack given operational points
# ================================================================================

class OperationalPoints:
    
    def __init__(
        self,
        aircraft_name: str,
        conditions: dict[str, Any],
        db_filepath: str | Path,
        input_units: Optional[list[str]] = ["m/s", "m"],
        enable_logging: bool = True,    # always last entry
    ) -> None:
        '''
        Organizes and store flight envelope data and atmosphere conditions.

        The 'conditions' input is a dictionary:

        Key                Description
        ---------------    --------------------------------------------------------
        'cond_tags'        List of condition names. e.g. ["cruise_max_lift", ...]
        'load_factor'      List of vertical load factor of each condition
        'velocity'         List of velocities of each condition (same units)
        'altitude'         List of altitude-density of flight (same units)

        - 'input_units' is an optional list of Units of
                [0]: velocity
                [1]: flight altitude
            Default is ["m/s", "m"]
        '''
        self.logger = io.setup_logger(self, enable_logging)

        self.aircraft    = aircraft_name
        self.db_filepath = Path(db_filepath) 
        self.input_units = input_units

        self.cond_tags   = list(conditions["cond_tags"])
        self.num_cond    = len(self.cond_tags)
        self.load_factor = np.asarray(conditions["load_factor"], dtype=float)
        self.velocity    = convvel(
            conditions["velocity"], input_units[0], "m/s",
        )
        self.altitude    = convlength(
            conditions["altitude"], input_units[1], "m",
        )
        self.T,  self.a,  self.P,  self.rho,  self.nu,  self.mu = atmosisa(
            self.altitude
        )

        self.logger.debug(
            f"T = {self.T} | a = {self.a} | P = {self.P} |" \
            f"rho = {self.rho} | nu = {self.nu} | mu = {self.mu}."
        )

        self.opp_data = self._pack_opp_data()

        io.write_json(
            obj=self.opp_data,
            filepath=self.db_filepath,
        )
        self.logger.info(
            f"All Operational Points were successfully writen to: {db_filepath}"
        )


    # ----------------------------------------
    # Private methods
    # ----------------------------------------

    def _pack_opp_data(self) -> OppData:
        return OppData(
            aircraft  = self.aircraft,
            num_cond  = int(self.num_cond),
            cond_tags = self.cond_tags,

            velocity    = np.round(self.velocity.astype(float), _N_DEC),
            load_factor = np.round(self.load_factor.astype(float), _N_DEC),
            altitude    = np.round(self.altitude.astype(float), _N_DEC),

            T   = np.round(self.T.astype(float), _N_DEC),
            a   = np.round(self.a.astype(float), _N_DEC),
            P   = np.round(self.P.astype(float), _N_DEC),
            rho = np.round(self.rho.astype(float), _N_DEC),
            nu  = np.round(self.nu.astype(float), _N_DEC),
            mu  = np.round(self.mu.astype(float), _N_DEC),

            input_units  = list(self.input_units),
            output_units = ["m/s", "m"],
        )
        

# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    aircraft_name = "da62"

    # Velocities in kts
    VS1 = 78.0
    VS2 = 85.0
    VG  = 104.0
    VA  = 152.0
    VC  = 171.0
    VD  = 205.0
    VH  = 162.0
    VC_min = 96.0

    # Load factors
    n1 = 3.8
    n2 = -1.52

    # Flight altitudes
    H0   = 0.0
    Hmax = 20000.0
    HC   = 16000.0
    Hmid = 10000.0

    # Conditions dictionary
    conditions = {
        'cond_tags'   : ["VS1_n1pos", "VA_nMax", "VC_nMax", "VD_nMax",
                         "VS2_n1neg", "VG_nMin", "VC_nMin", "VD_nzero",
                         "VA_n1pos", "VC_n1pos", "VH_n1pos", "VH_nMax",
                         "VCmin_n1pos"],
        'load_factor' : [ 1, n1, n1, n1,
                         -1, n2, n2,  0,
                          1,  1,  1, n1,
                          1],
        'velocity'    : [VS1, VA, VC, VD,
                         VS2, VG, VC, VD,
                          VA, VC, VH, VH,
                         VC_min],
        'altitude'    : [H0, Hmid, HC, Hmax,
                         H0, Hmid, HC, HC,
                         Hmid, HC, Hmid, HC,
                         HC],
    }
    input_units = ['kts', 'ft']

    OperationalPoints(
        aircraft_name=aircraft_name,
        conditions=conditions,
        db_filepath=_DFLT_OPP_DIR / f"{aircraft_name}_OppData.json",
        input_units=input_units,
    )
