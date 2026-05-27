'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Airfoil Module.

This module contains the Airfoil class, which transforms a raw .dat airfoil
file into upper and lower adimensional coordinates. The resulting data is
stored in a dataclass container, called by downstream modules when needed.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path

from dataclasses import dataclass

import numpy as np

# ================ Pathing ================
_SRC  = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Default Database Paths ================
_DFLT_AFL_DIR = _ROOT / "data" / "airfoils"

# ================ Module imports ================

# Utilities
from utils import io_utils as io
from utils import math_utils as mthu

# ================ Global variables ================
# The '2' multiplier means that Upper and Lower surfaces
# will have the same number of points
_N_POINTS = 2 * 50  

# If 6, when chord in [mm], final values will have
# 2 decimal places, in general.
_N_DEC_PLACES = 6

# Points per surface for the internal fine parametric grid
# used by the NACA analytic generators before cosine resampling.
_N_FINE = 500



# ================================================================================
# Data persistence - Airfoil data container
# ================================================================================

@dataclass
class AirfoilData:
    '''
    Container for storing airfoil adimensional points into upper and lower
    coordinates. Camber line coordinates are also included.

    Property    Size                Description                         Unit
    --------    ----------------    ----------------------------    --------
    x_upper     (n_points/2,)       Upper surface x-coordinate      mm [0; 1]
    y_upper     (n_points/2,)       Upper surface y-coordinate      mm [0; 1]

    x_lower     (n_points/2,)       Lower surface x-coordinate      mm [0; 1]
    y_lower     (n_points/2,)       Lower surface y-coordinate      mm [0; 1]

    x_camber    (n_points/2,)       Camber line x-coordinate        mm [0; 1]
    y_camber    (n_points/2,)       Camber line y-coordinate        mm [0; 1]

    '''
    x_upper  : np.ndarray
    y_upper  : np.ndarray
    x_lower  : np.ndarray
    y_lower  : np.ndarray
    x_camber : np.ndarray
    y_camber : np.ndarray



# ================================================================================
# Public API - Manipulates and saves airfoil coordinates into database
# ================================================================================

class Airfoil:
    '''
    Builds and stores airfoil data from raw .dat file

    The full construction pipeline runs during '__init__', so that downstream
    instances of the object contains all desired information.
    '''
    def __init__(
            self,
            filename: str,
            n_points: int = int(_N_POINTS / 2),
            enable_logging: bool = True
        ) -> None:
        '''
        Initializes the Airfoil data with given entries.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        # Self instances
        self.db_filepath = Path(_DFLT_AFL_DIR / f"{filename.lower()}_AirfoilData.json")
        self.n_points = n_points
        self.filename = filename.lower()

        # Build airfoil data [.dat]
        if not self.filename.endswith(".dat"):
            self.filename += ".dat"
        self.filepath = _DFLT_AFL_DIR / f"{self.filename}"

        self.afl_data = self._define_airfoil_data()

        # Save to Database
        self.logger.info(f"Writing JSON Airfoil Database file")

        io.write_json(
            obj=self.afl_data,
            filepath=self.db_filepath
        )

        self.logger.info(f"Airfoil Database successfully writen to: {self.db_filepath}")


    # ----------------------------------------
    # Private methods - inner calculation
    # ----------------------------------------

    def _split_upper_lower(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Splits airfoil coordinates into upper and lower surfaces.

        Identifies, first, if x-values are sorted from LE to TE or otherwise,
        and then manipulates to go from LE to TE.
        '''
        le = int(np.argmin(x))
        te = int(np.argmax(x))
        if y[le+1] < 0:
            x_u = x[:le + 1][::-1]
            y_u = y[:le + 1][::-1]
            x_l = x[le:]
            y_l = y[le:]
        else:
            x_u = x[:te + 1]
            y_u = y[:te + 1]
            x_l = x[te:][::-1]
            y_l = y[te:][::-1]
        return (x_u, y_u, x_l, y_l)
    
    def _cosine_resample(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ''' Redistributes a surface to self.n_points using cosine spacing. '''

        x_cos = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, self.n_points)))
        y_new = mthu.interp1d_cubic(x_cos, x, y)
        return x_cos, y_new
    
    def _camber_line(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            0.5 * (self.x_upper + self.x_lower),
            0.5 * (self.y_upper + self.y_lower)
        )

    def _naca_thickness(
        self,
        x: np.ndarray,
        t: float,
    ) -> np.ndarray:
        ''' NACA 4-digit symmetric thickness distribution. '''
        return 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    def _apply_camber_normal(
        self,
        x    : np.ndarray,
        y_c  : np.ndarray,
        dy_c : np.ndarray,
        yt   : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Projects thickness normal to the camber line.

        Returns:
            Tuple (x_upper, y_upper, x_lower, y_lower).
        '''
        theta = np.arctan(dy_c)
        return (
            x - yt * np.sin(theta),
            y_c + yt * np.cos(theta),
            x + yt * np.sin(theta),
            y_c - yt * np.cos(theta),
        )


    # ----------------------------------------
    # Private method - Building database
    # ----------------------------------------

    def _define_airfoil_data(self) -> AirfoilData:
        '''
        Main airfoil database builder.

        Expects airfoil data in Selig format, i.e., going from TE to LE in
        upper surface, then continuing from LE to TE in lower surface.
        
        Parced raw data is already normalized.

        Returns:
            ArfoilData object
        '''
        self.logger.info(f"Reading .dat file: {self.filepath}")

        raw = io.read_dat_file(self.filepath)
        x_raw, y_raw = raw["x"], raw["y"]

        self.logger.info("Spliting upper and lower surfaces")
        x_u, y_u, x_l, y_l = self._split_upper_lower(x_raw, y_raw)

        self.logger.info(
            f"Applying cosine redistribution: {self.n_points} pts/surface"
        )
        self.x_upper, self.y_upper = self._cosine_resample(x_u, y_u)
        self.x_lower, self.y_lower = self._cosine_resample(x_l, y_l)
        
        self.logger.info("Constructing camber line")
        self.x_camber, self.y_camber = self._camber_line()

        self.logger.info("Airfoil Database has been built successfully.")

        airfoil_data = {
            "x_upper" : self.x_upper,
            "y_upper" : self.y_upper,
            "x_lower" : self.x_lower,
            "y_lower" : self.y_lower,
            "x_camber": self.x_camber,
            "y_camber": self.y_camber,
        }
        for key, value in airfoil_data.items():
            airfoil_data[key] = np.round(value, _N_DEC_PLACES)

        return AirfoilData(**airfoil_data)
    
    def _build_NACA_airfoil_data(
        self,
        x_u : np.ndarray,
        y_u : np.ndarray,
        x_l : np.ndarray,
        y_l : np.ndarray,
    ) -> AirfoilData:
        '''
        Resamples raw surfaces to cosine grid and packs into AirfoilData.
        '''
        self.x_upper, self.y_upper = self._cosine_resample(x_u, y_u)
        self.x_lower, self.y_lower = self._cosine_resample(x_l, y_l)
        self.x_camber, self.y_camber = self._camber_line()
        return AirfoilData(
            x_upper  = np.round(self.x_upper,  _N_DEC_PLACES),
            y_upper  = np.round(self.y_upper,  _N_DEC_PLACES),
            x_lower  = np.round(self.x_lower,  _N_DEC_PLACES),
            y_lower  = np.round(self.y_lower,  _N_DEC_PLACES),
            x_camber = np.round(self.x_camber, _N_DEC_PLACES),
            y_camber = np.round(self.y_camber, _N_DEC_PLACES),
        )

    # ----------------------------------------
    # Public methods - NACA Generators
    # ----------------------------------------

    def naca_four_digits_generator(
        self,
        designation: str,
    ) -> AirfoilData:
        '''
        Generates a NACA 4-digit airfoil resampled to cosine distribution.

        Parses the 4-character designation to extract camber and thickness
        parameters. Surface coordinates use the standard NACA 4-digit
        parabolic-arc mean line with exact normal projection.

        Args:
            designation: NACA code string, e.g. "2412".

        Returns:
            AirfoilData with self.n_points cosine-resampled per surface.

        Raises:
            ValueError: If designation is not exactly 4 characters.
        '''
        if len(designation) != 4:
            raise ValueError(
                f"[CL3O] NACA designation must be exactly 4 digits.\n"
                f"| Got : {designation!r}"
            )
        M  = int(designation[0])
        P  = int(designation[1])
        TT = int(designation[2:4])

        m = M / 100.0
        p = P / 10.0
        t = TT / 100.0

        x = np.linspace(0.0, 1.0, _N_FINE)

        # Camber line and gradient (parabolic arcs)
        if m == 0.0 or p == 0.0:
            y_c  = np.zeros_like(x)
            dy_c = np.zeros_like(x)
        else:
            fwd  = x <= p
            y_c  = np.where(
                fwd,
                (m / p**2) * (2.0*p*x - x**2),
                (m / (1.0 - p)**2) * (1.0 - 2.0*p + 2.0*p*x - x**2),
            )
            dy_c = np.where(
                fwd,
                (2.0*m / p**2) * (p - x),
                (2.0*m / (1.0 - p)**2) * (p - x),
            )

        yt = self._naca_thickness(x, t)
        x_u, y_u, x_l, y_l = self._apply_camber_normal(
            x, y_c, dy_c, yt
        )
        return self._build_NACA_airfoil_data(x_u, y_u, x_l, y_l)


    def naca_five_digits_generator(
        self,
        designation: str,
    ) -> AirfoilData:
        '''
        Generates a NACA 5-digit airfoil resampled to cosine distribution.

        Parses the 5-character designation to extract camber and thickness
        parameters. Surface coordinates are computed from the standard NACA
        TN 427 formulas and resampled with the same cosine scheme used for
        .dat-sourced airfoils.

        Only non-reflexed mean lines (third digit = 0) are supported.
        k1 values from NACA TN 427 are tabulated for L=2 and scaled
        linearly for other L values.

        Args:
            designation: NACA code string, e.g. "23012".

        Returns:
            AirfoilData with self.n_points cosine-resampled per surface.

        Raises:
            ValueError: If designation is not 5 chars, S != 0, or
                        P is outside the supported range [1-5].
        '''
        if len(designation) != 5:
            raise ValueError(
                f"[CL3O] NACA designation must be exactly 5 digits.\n"
                f"| Got : {designation!r}"
            )
        L  = int(designation[0])
        P  = int(designation[1])
        S  = int(designation[2])
        TT = int(designation[3:5])

        if S != 0:
            raise ValueError(
                f"[CL3O] Non-reflexed mean line (S=0) required.\n"
                f"| Got S = {S}"
            )

        t = TT / 100.0
        m = P * 0.05

        # k1 table from NACA TN 427, normalised to L=2
        k1_map = {1: 361.4, 2: 51.64, 3: 15.957, 4: 6.643, 5: 3.230}
        if P not in k1_map:
            raise ValueError(
                f"[CL3O] P={P} not supported. "
                f"Valid: {list(k1_map.keys())}"
            )
        k1 = k1_map[P] * L / 2.0

        x = np.linspace(0.0, 1.0, _N_FINE)

        # Camber line and gradient (NACA TN 427, non-reflexed)
        fwd  = x < m
        y_c  = np.where(
            fwd,
            (k1 / 6.0) * (x**3 - 3*m*x**2 + m**2*(3.0 - m)*x),
            (k1 * m**3 / 6.0) * (1.0 - x),
        )
        dy_c = np.where(
            fwd,
            (k1 / 6.0) * (3*x**2 - 6*m*x + m**2*(3.0 - m)),
            np.full_like(x, -k1 * m**3 / 6.0),
        )

        yt = self._naca_thickness(x, t)
        x_u, y_u, x_l, y_l = self._apply_camber_normal(
            x, y_c, dy_c, yt
        )
        return self._build_NACA_airfoil_data(x_u, y_u, x_l, y_l)


    def naca_six_digits_generator(
        self,
        designation: str,
    ) -> AirfoilData:
        '''
        Generates a NACA 6-series airfoil resampled to cosine distribution.

        Parses the 6-character designation string (format "6XYZTT"):

            6 X Y Z TT
            | | | | '--  thickness percentage (e.g. 15 -> 15%)
            | | | '----  design Cl x10  (e.g. 4 -> Cl_des=0.4)
            | | '------  half bucket width x10 (not used in geometry)
            | '--------  min-pressure x/c position x10 (not used in geometry)
            '----------  series identifier (must be '6')

        Mean line uses the a=1 uniform-load NACA formulation (NACA TN 2502).
        Thickness uses the NACA 4-digit polynomial as an approximation (exact
        6-series thickness forms are sub-series specific and tabulated).
        Simple vertical stacking is used instead of normal projection because
        the a=1 mean line has a logarithmic LE singularity that would push
        upper-surface x-coordinates below zero near the leading edge.

        Args:
            designation: NACA code string, e.g. "632415".

        Returns:
            AirfoilData with self.n_points cosine-resampled per surface.

        Raises:
            ValueError: If designation is not 6 chars or first char != '6'.
        '''
        if len(designation) != 6 or designation[0] != '6':
            raise ValueError(
                f"[CL3O] NACA 6-series designation must be 6 characters "
                f"starting with '6'.\n"
                f"| Got : {designation!r}"
            )
        Z  = int(designation[3])
        TT = int(designation[4:6])

        t      = TT / 100.0
        cl_des = Z / 10.0

        x = np.linspace(0.0, 1.0, _N_FINE)

        # a=1 uniform-load mean line (NACA TN 2502)
        # Clip endpoints to avoid log singularities; boundary y_c = 0 by
        # continuity (lim x->0  x*ln(x) = 0, same at x->1).
        x_s  = np.clip(x, 1e-10, 1.0 - 1e-10)
        y_c  = -(cl_des / (2.0*np.pi)) * (
            x_s * np.log(x_s) + (1.0 - x_s) * np.log(1.0 - x_s)
        )
        y_c[0] = y_c[-1] = 0.0

        yt = self._naca_thickness(x, t)

        # Vertical stacking avoids x_u < 0 near the cusped LE
        x_u = x.copy()
        y_u = y_c + yt
        x_l = x.copy()
        y_l = y_c - yt

        return self._build_NACA_airfoil_data(x_u, y_u, x_l, y_l)



# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    filename = "wortmannfx63137"

    Airfoil(
        filename=filename,
        n_points=200,
    )
    # pass