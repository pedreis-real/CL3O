'''
================================================================================
CL3O - Composite Wing Structural Sizing.
Laminate Module.

Ply-by-ply calculations and laminate engineering constants for composite
materials. Follows the on-axis / off-axis convention from Tsai & Hahn (1980):
    xys --> on-axis, aligned with principal fiber direction
    126 --> off-axis, aligned with global composite axis (angle = 0)

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path

from dataclasses import dataclass, field

import numpy as np

# ================ Default paths ================
from cl3o.paths import MATERIALS_DIR as _DFLT_MAT_DIR, PLIES_DIR as _DFLT_PLY_DIR

# ================ Module imports ================

# Constants
from cl3o.Constants import TOL

# Utilities
from cl3o.utils import io_utils as io


# ================================================================================
# Data containers
# ================================================================================

@dataclass
class PlyData:
    '''
    Container for mechanical and geometrical properties of a single ply.

    On-axis (xys) properties are aligned with the principal fiber orientation.
    Off-axis (126) properties are aligned with the global composite axis.

    Property    Size        Description                                   Unit
    --------    --------    ------------------------------------------    ------
    name        -           Ply name                                      str
    thick       (1,)        Ply thickness                                 mm
    angle       (1,)        Fiber angle                                   degree
    core        -           True if the ply is a core (sandwich)          bool
    rho         (1,)        Constituent material density                  ton/mm^3
    gsm         (1,)        GSM measure of ply material                   g/m^2

    Ex          (1,)        Young modulus, longitudinal                   MPa
    Ey          (1,)        Young modulus, transversal                    MPa
    Es          (1,)        Shear modulus                                 MPa
    nux         (1,)        Longitudinal Poisson ratio                    -
    nuy         (1,)        Transversal Poisson ratio                     -

    Qxys        (3,3)       On-axis stiffness tensor                      MPa
    Sxys        (3,3)       On-axis compliance tensor                     1/MPa
    Qxys_vec    (4,1)       On-axis stiffness vector                      MPa
    Sxys_vec    (4,1)       On-axis compliance vector                     1/MPa

    Ts_p        (3,3)       Stress transformation matrix, angle +         -
    Ts_n        (3,3)       Stress transformation matrix, angle -         -
    Te_p        (3,3)       Strain transformation matrix, angle +         -
    Te_n        (3,3)       Strain transformation matrix, angle -         -
    Tstiff_p    (6,4)       Stiffness transformation matrix, angle +      -
    Tstiff_n    (4,6)       Stiffness transformation matrix, angle -      -
    Tcompl_p    (6,4)       Compliance transformation matrix, angle +     -
    Tcompl_n    (4,6)       Compliance transformation matrix, angle -     -

    Q126        (3,3)       Off-axis stiffness tensor                     MPa
    S126        (3,3)       Off-axis compliance tensor                    1/MPa
    Q126_vec    (6,1)       Off-axis stiffness vector                     MPa
    S126_vec    (6,1)       Off-axis compliance vector                    1/MPa

    Xt / Xc     (1,)        Tensile / compressive strength, long.         MPa
    Yt / Yc     (1,)        Tensile / compressive strength, trans.        MPa
    S           (1,)        Shear strength                                MPa
    
    Fxx / Fyy   (1,)        Tsai-Wu quadratic factors, long./trans.       MPa^{-2}
    Fss         (1,)        Tsai-Wu quadratic factor, shear               MPa^{-2}
    Fxy         (1,)        Tsai-Wu biaxial interaction factor            MPa^{-2}
    Fx / Fy     (1,)        Tsai-Wu linear factors, long./trans.          MPa^{-1}
    '''
    name  : str
    thick : float
    angle : float
    core  : bool
    rho   : float
    gsm   : float

    # All fields below are None for core plies
    Ex    : float = None
    Ey    : float = None
    Es    : float = None
    nux   : float = None
    nuy   : float = None

    Xt : float = None
    Xc : float = None
    Yt : float = None
    Yc : float = None
    S  : float = None

    Qxys     : np.ndarray = None
    Sxys     : np.ndarray = None
    Qxys_vec : np.ndarray = None
    Sxys_vec : np.ndarray = None

    Ts_p : np.ndarray = None
    Ts_n : np.ndarray = None
    Te_p : np.ndarray = None
    Te_n : np.ndarray = None

    Tstiff_p : np.ndarray = None
    Tcompl_p : np.ndarray = None
    Tstiff_n : np.ndarray = None
    Tcompl_n : np.ndarray = None

    Q126     : np.ndarray = None
    S126     : np.ndarray = None
    Q126_vec : np.ndarray = None
    S126_vec : np.ndarray = None

    Fxx : float = None
    Fyy : float = None
    Fss : float = None
    Fxy : float = None
    Fx  : float = None
    Fy  : float = None


@dataclass
class LaminateData:
    '''
    Container for engineering constants of a generic composite laminate.

    Property    Size        Description                                     Unit
    --------    --------    ----------------------------------------    --------
    name        -           Laminate name                               str
    n_plies     (1,)        Total number of plies in layup              -
    plies       (n_plies,)  Ordered list of ply names                   str

    thick       (1,)        Total laminate thickness                    mm
    rho         (1,)        Equivalent density                          ton/mm^3
    gsm         (1,)        Equivalent GSM of laminate material         g/m^2

    stiff_A     (3, 3)      In-plane stiffness matrix                   N/mm
    stiff_B     (3, 3)      Coupling stiffness matrix                   N
    stiff_D     (3, 3)      Bending stiffness matrix                    N*mm
    stiff_ABD   (6, 6)      Full ABD stiffness matrix                   -

    compl_a     (3, 3)      In-plane compliance matrix                  mm/N
    compl_b     (3, 3)      In-plane/bending coupling compliance        N^{-1}
    compl_c     (3, 3)      Bending/in-plane coupling compliance        N^{-1}
    compl_d     (3, 3)      Bending compliance matrix                   1/(N*mm)
    compl_abcd  (6, 6)      Full compliance matrix                      -

    E1          (1,)        Membrane longitudinal modulus (A-based)     MPa
    E2          (1,)        Membrane transversal modulus (A-based)      MPa
    G12         (1,)        Membrane shear modulus (A-based)            MPa
    nu12        (1,)        Membrane Poisson ratio 1->2                 -
    nu21        (1,)        Membrane Poisson ratio 2->1                 -
    eta16       (1,)        Shear coupling factors 1 to 6               -
    eta26       (1,)        Shear coupling factors 2 to 6               -
    eta61       (1,)        Shear coupling factors 6 to 1               -
    eta62       (1,)        Shear coupling factors 6 to 2               -

    E1_bend     (1,)        Flexural longitudinal modulus (D-based)     MPa
    E2_bend     (1,)        Flexural transversal modulus (D-based)      MPa
    G12_bend    (1,)        Flexural shear modulus (D-based)            MPa
    nu12_bend   (1,)        Flexural Poisson ratio 1->2                 -
    nu21_bend   (1,)        Flexural Poisson ratio 2->1                 -

    eng_compl   (3, 3)      Equivalent compliance tensor                1/MPa
    eng_stiff   (3, 3)      Equivalent stiffness tensor                 MPa

    stacking_seq  -         Standard stacking sequence string           str
    '''
    name         : str
    plies        : list[str]
    stacking_seq : str

    n_plies : int   = 0
    thick   : float = 0
    rho   : float = 0
    gsm   : float = 0

    stiff_A   : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    stiff_B   : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    stiff_D   : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    stiff_ABD : np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))

    compl_a    : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    compl_b    : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    compl_c    : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    compl_d    : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    compl_abcd : np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))

    E1    : float = 0
    E2    : float = 0
    G12   : float = 0
    nu12  : float = 0
    nu21  : float = 0
    eta16 : float = 0
    eta26 : float = 0
    eta61 : float = 0
    eta62 : float = 0

    E1_bend  : float = 0
    E2_bend  : float = 0
    G12_bend : float = 0
    nu12_bend : float = 0
    nu21_bend : float = 0

    eng_compl : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    eng_stiff : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))



# ================================================================================
# Internal helpers
# ================================================================================

class MaterialHelper:

    def __init__(self):
        pass

    @staticmethod
    def validate_lamina_inputs(obj) -> None:
        '''Raises ValueError if any required lamina property is missing.'''
        values_to_check = {
            "rho" : obj.rho,
            "Ex"  : obj.Ex,
            "Ey"  : obj.Ey,
            "Es"  : obj.Es,
            "nux" : obj.nux,
            "nuy" : obj.nuy,
            "Xt"  : obj.Xt,
            "Xc"  : obj.Xc,
            "Yt"  : obj.Yt,
            "Yc"  : obj.Yc,
            "S"   : obj.S,
        }
        for name, value in values_to_check.items():
            if value is None:
                raise ValueError(
                    f"All lamina properties must be given for a non-core ply. "
                    f"Missing: {name}."
                )
    
    @staticmethod
    def laminate_by_index(laminate_db: dict, lam_idx: int) -> LaminateData:
        '''
        Resolve a LaminateData from the database by its 0-based DE index.

        The DE design vector stores each laminate as a 0-based index into
        LAYUP_ORDER; the database keys them as 'MAT{index + 1}'. Centralising
        the convention here keeps StructuralMass and TsaiWuFailure in sync.
        '''
        return laminate_db[f"MAT{int(lam_idx) + 1}"]

    @staticmethod
    def get_ply_filename(name, thickness, angle) -> str:
        '''
        Persistent Ply name for single ply object.
        
        Args:
            name        Ply name    (e.g. 'CFRP-uni', 'GFRP-bi')
            thickness   Ply thickness [mm]
            angle       Ply angle relative to global composite coord. system [deg.]
        '''
        return f"{name.lower()}_t={thickness}_ang={int(angle)}"
    
    @staticmethod
    def get_gsm(obj) -> float:
        '''
        Returns the gram-per-square-metre (GSM) measure of this ply.
        
        Units:
            rho       - ton/mm^3
            thickness - mm
            gsm       - g / m^2
        
        So that
            .. 1g/m^2 = (1e-6 ton) / (1e3 mm)^3 * (1e3 mm)
            .. 1g/m^2 = (1e-15 ton/mm^3) * (1e3 mm)
        '''
        return (obj.rho * 1e15) * (obj.thick * 1e-3)



# ================================================================================
# Public API - Ply-level calculations
# ================================================================================

class Ply:

    def __init__(
        self,
        name  : str,
        thick : float,
        angle : float,
        core  : bool = False,
        rho   : float | None = None,
        Ex    : float | None = None,
        Ey    : float | None = None,
        Es    : float | None = None,
        nux   : float | None = None,
        Xt    : float | None = None,
        Xc    : float | None = None,
        Yt    : float | None = None,
        Yc    : float | None = None,
        S     : float | None = None,
        enable_logging: bool = True
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)

        self.name = name
        self.core = core
        self.thick = thick
        self.angle = angle
        self.rho = rho
        self.gsm = MaterialHelper.get_gsm(self)

        # Ply file path derived from ply name, angle and thickness
        ply_filename = MaterialHelper.get_ply_filename(name, thick, angle)
        self.ply_db_filepath = Path(
            _DFLT_PLY_DIR / f"PlyData_{ply_filename}.json"
        )

        self.logger.info(f"Building Ply Database: '{self.ply_db_filepath}'.")

        if not self.core:
            self.Ex    = Ex
            self.Ey    = Ey
            self.Es    = Es
            self.nux   = nux
            self.nuy   = self._get_nuy()
            self.Xt    = Xt
            self.Xc    = Xc
            self.Yt    = Yt
            self.Yc    = Yc
            self.S     = S

            self._build_ply_data()
        else:
            # Core plies carry no stiffness data;
            # Data packed with only geometric and metadata info.
            self.ply_data = self._pack_core_data()

            io.write_json(
                obj=self.ply_data,
                filepath=self.ply_db_filepath,
            )


    # ----------------------------------------
    # Private methods - inner calculations
    # ----------------------------------------

    def _get_nuy(self) -> float:
        '''Returns nuy, derived from tensor symmetry.'''
        return self.nux * self.Ey / self.Ex


    def _calculate_ply_stiffness(self) -> None:
        '''
        On-axis stiffness tensor Qij (3x3) and column vector [Qxx Qyy Qxy Qss].
        '''
        den = 1.0 - self.nux * self.nuy

        Qxx = self.Ex          / den
        Qyy = self.Ey          / den
        Qxy = self.Ex * self.nuy / den
        Qyx = self.Ey * self.nux / den
        Qss = self.Es

        self.Qxys = np.array([
            [Qxx, Qxy, 0.0],
            [Qyx, Qyy, 0.0],
            [0.0, 0.0, Qss],
        ])

        self.Qxys_vec = np.array([[Qxx], [Qyy], [Qxy], [Qss]])


    def _calculate_ply_compliance(self) -> None:
        '''
        On-axis compliance tensor Sij (3x3) and column vector [Sxx Syy Sxy Sss].
        '''
        Sxx = 1.0  / self.Ex
        Syy = 1.0  / self.Ey
        Sxy = -self.nuy / self.Ey
        Syx = -self.nux / self.Ex
        Sss = 1.0  / self.Es

        self.Sxys = np.array([
            [Sxx, Sxy, 0.0],
            [Syx, Syy, 0.0],
            [0.0, 0.0, Sss],
        ])

        self.Sxys_vec = np.array([[Sxx], [Syy], [Sxy], [Sss]])


    def _calculate_stress_transformation_matrix(self) -> None:
        '''Stress transformation matrices Ts_p and Ts_n.'''
        m, n = self.m, self.n

        self.Ts_p = np.array([
            [ m**2,  n**2,  2*m*n],
            [ n**2,  m**2, -2*m*n],
            [-m*n,   m*n,  m**2 - n**2],
        ])
        self.Ts_n = np.linalg.inv(self.Ts_p)


    def _calculate_strain_transformation_matrix(self) -> None:
        '''Strain transformation matrices Te_p and Te_n.'''
        m, n = self.m, self.n

        self.Te_p = np.array([
            [ m**2,  n**2,  m*n],
            [ n**2,  m**2, -m*n],
            [-2*m*n, 2*m*n, m**2 - n**2],
        ])
        self.Te_n = np.linalg.inv(self.Te_p)


    def _calculate_stiffness_transformation_matrix(self) -> None:
        '''
        Stiffness transformation matrix Tstiff_p (6x4).

        Maps the on-axis vector [Qxx Qyy Qxy Qss] to the six off-axis
        components [Q11 Q22 Q12 Q66 Q16 Q26]. Because the matrix is not
        square, its pseudo-inverse (pinv) is used for back-transformation.
        '''
        m, n = self.m, self.n

        self.Tstiff_p = np.array([
            [m**4,     n**4,     2*m**2*n**2,       4*m**2*n**2],
            [n**4,     m**4,     2*m**2*n**2,       4*m**2*n**2],
            [m**2*n**2, m**2*n**2, m**4 + n**4,    -4*m**2*n**2],
            [m**2*n**2, m**2*n**2, -2*m**2*n**2,  (m**2-n**2)**2],
            [m**3*n,  -m*n**3,   m*n**3 - m**3*n, 2*(m*n**3 - m**3*n)],
            [m*n**3,  -m**3*n,   m**3*n - m*n**3, 2*(m**3*n - m*n**3)],
        ])

        # Moore-Penrose pseudo-inverse instead.
        self.Tstiff_n = np.linalg.pinv(self.Tstiff_p)


    def _calculate_compliance_transformation_matrix(self) -> None:
        '''
        Compliance transformation matrix Tcompl_p (6x4).

        Same shape rationale as Tstiff_p; pseudo-inverse used for inversion.
        '''
        m, n = self.m, self.n

        self.Tcompl_p = np.array([
            [m**4,      n**4,      2*m**2*n**2,       m**2*n**2],
            [n**4,      m**4,      2*m**2*n**2,       m**2*n**2],
            [m**2*n**2, m**2*n**2, m**4 + n**4,      -m**2*n**2],
            [4*m**2*n**2, 4*m**2*n**2, -8*m**2*n**2, (m**2-n**2)**2],
            [2*m**3*n, -2*m*n**3, 2*(m*n**3 - m**3*n), m*n**3 - m**3*n],
            [2*m*n**3, -2*m**3*n, 2*(m**3*n - m*n**3), m**3*n - m*n**3],
        ])
        
        self.Tcompl_n = np.linalg.pinv(self.Tcompl_p)


    def _transform_on_axis_to_off_axis(self) -> None:
        '''Rotate Q and S tensors from on-axis to off-axis frame.'''
        self.Q126 = self.Ts_n @ self.Qxys @ self.Te_p
        self.S126 = self.Te_n @ self.Sxys @ self.Ts_p

        self.Q126_vec = self.Tstiff_p @ self.Qxys_vec   # (6x1)
        self.S126_vec = self.Tcompl_p @ self.Sxys_vec   # (6x1)




    def _calculate_tsaiwu_factors(self) -> None:
        '''
        Tsai-Wu strength parameters.

        Biaxial interaction limit: Fxy = -0.5 * sqrt(Fxx * Fyy).
        '''
        self.Fxx = 1.0 / (self.Xt * self.Xc)
        self.Fyy = 1.0 / (self.Yt * self.Yc)
        self.Fss = 1.0 / (self.S ** 2)
        self.Fxy = -0.5 * np.sqrt(self.Fxx * self.Fyy)

        self.Fx = 1.0 / self.Xt - 1.0 / self.Xc
        self.Fy = 1.0 / self.Yt - 1.0 / self.Yc


    # ----------------------------------------
    # Private methods - data packing
    # ----------------------------------------

    def _pack_core_data(self) -> PlyData:
        '''Pack a core ply into PlyData (all stiffness fields remain None).'''
        return PlyData(
            name  = self.name,
            thick = float(self.thick),
            angle = float(self.angle),
            core  = self.core,
            rho   = float(self.rho),
            gsm   = float(self.gsm),
        )

    def _pack_ply_data(self) -> PlyData:
        '''Cast self attributes into PlyData with fixed decimal precision.'''
        return PlyData(
            name  = self.name,
            thick = float(self.thick),
            angle = float(self.angle),
            core  = self.core,
            rho   = float(self.rho),
            gsm   = float(self.gsm),

            Ex    = float(self.Ex),   
            Ey    = float(self.Ey),   
            Es    = float(self.Es),   
            nux   = float(self.nux),  
            nuy   = float(self.nuy),  

            Qxys     = self.Qxys.astype(float),    
            Sxys     = self.Sxys.astype(float),    
            Qxys_vec = self.Qxys_vec.astype(float),
            Sxys_vec = self.Sxys_vec.astype(float),

            Ts_p = self.Ts_p.astype(float),
            Ts_n = self.Ts_n.astype(float),
            Te_p = self.Te_p.astype(float),
            Te_n = self.Te_n.astype(float),

            Tstiff_p = self.Tstiff_p.astype(float),
            Tcompl_p = self.Tcompl_p.astype(float),
            Tstiff_n = self.Tstiff_n.astype(float),
            Tcompl_n = self.Tcompl_n.astype(float),

            Q126     = self.Q126.astype(float),    
            S126     = self.S126.astype(float),    
            Q126_vec = self.Q126_vec.astype(float),
            S126_vec = self.S126_vec.astype(float),

            Xt = float(self.Xt),
            Xc = float(self.Xc),
            Yt = float(self.Yt),
            Yc = float(self.Yc),
            S  = float(self.S), 

            Fxx = float(self.Fxx),
            Fyy = float(self.Fyy),
            Fss = float(self.Fss),
            Fxy = float(self.Fxy),
            Fx  = float(self.Fx), 
            Fy  = float(self.Fy), 
        )


    # ----------------------------------------
    # Private method - build pipeline
    # ----------------------------------------

    def _build_ply_data(self) -> None:
        '''Full ply computation pipeline for non-core plies.'''
        MaterialHelper.validate_lamina_inputs(self)

        self.m = np.cos(np.radians(self.angle))
        self.n = np.sin(np.radians(self.angle))

        self._calculate_ply_stiffness()
        self._calculate_ply_compliance()

        self._calculate_stress_transformation_matrix()
        self._calculate_strain_transformation_matrix()

        self._calculate_stiffness_transformation_matrix()
        self._calculate_compliance_transformation_matrix()

        self._transform_on_axis_to_off_axis()

        self._calculate_tsaiwu_factors()

        self.ply_data = self._pack_ply_data()

        io.write_json(
            obj=self.ply_data,
            filepath=self.ply_db_filepath,
        )



# ================================================================================
# Public API - Laminate engineering constants
# ================================================================================

class Laminate:

    def __init__(
        self,
        name: str,
        enable_logging: bool = True,
    ) -> None:
        '''
        Ply-by-ply laminate calculator.

        Correct usage pipeline:
          1. Instantiate with a name and a database path.
          2. Add plies FROM BOTTOM TO TOP with add_ply().
          3. Call define_laminate_data() to compute and persist all results.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        self.name  = name
        self.plies : list[PlyData] = []


    # ----------------------------------------
    # Private methods - inner calculations
    # ----------------------------------------

    def _write_stacking_sequence(self) -> None:
        '''
        Build the standard stacking sequence string for this laminate.

        Follows the ASTM/Tsai convention:
          - Angles listed from bottom to top inside brackets.
          - Even-count symmetric laminates use the "_s" suffix; only the
            first half is shown, since "_s" implies mirroring it.
          - Odd-count symmetric laminates are shown in full: an "_s" suffix
            implies an even mirror (e.g. [0,45,90]_s = [0,45,90,90,45,0]),
            which has a different midplane than [0,45,90,45,0].
          - Core plies are skipped (they carry no fiber orientation).

        Examples
        --------
          [0,45,-45,90,90,-45,45,0]   --> [0,45,-45,90]_s
          [0,45,90,45,0]              --> [0,45,90,45,0]
          [0,45,-45,0]                --> [0,45,-45,0]
        '''
        angles = [int(ply.angle) for ply in self.plies]

        is_sym = (angles == angles[::-1]) and (int(np.mod(len(angles),2)) == 0)

        if is_sym:
            n_half = (len(angles) + 1) // 2
            displayed = angles[:n_half]
        else:
            displayed = angles

        parts = []
        i = 0
        while i < len(displayed):
            a = displayed[i]
            count = 1
            while i + count < len(displayed) and displayed[i + count] == a:
                count += 1
            parts.append(f"{a}_{count}" if count > 1 else str(a))
            i += count

        seq    = ",".join(parts)
        suffix = "_s" if is_sym else ""

        self.stacking_seq = f"[{seq}]{suffix}"
    

    def _get_total_thickness(self) -> None:
        '''Compute total thickness of composite layup.'''
        self.thick = float(sum(ply.thick for ply in self.plies)) 


    def _calculate_laminate_density(self) -> None:
        '''Equivalent density via thickness-weighted average.'''
        self.rho = float(
            sum(ply.rho * ply.thick for ply in self.plies) / self.thick
        )
        self.gsm = MaterialHelper.get_gsm(self)


    def _calculate_z_coords(self) -> None:
        '''
        z-coordinates of each ply, referenced to the laminate mid-plane.

        z_start / z_end are the bottom and top faces; z_mid is the ply centroid.
        '''
        z_start, z_end, z_mid = [], [], []

        z = 0.0
        for ply in self.plies:
            z0 = z
            z1 = z0 + ply.thick
            z_start.append(z0)
            z_end.append(z1)
            z_mid.append(0.5 * (z0 + z1))
            z = z1

        half = self.thick / 2.0
        
        self.z_start = np.array(z_start) - half
        self.z_end   = np.array(z_end)   - half
        self.z_mid   = np.array(z_mid)   - half


    def _calculate_laminate_stiffness(self) -> None:
        '''ABD stiffness matrices via Classical Lamination Theory (CLT).'''
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for ply, z0, z1 in zip(self.plies, self.z_start, self.z_end):
            # Core plies have Q126 = None and contribute no stiffness.
            if ply.core or ply.Q126 is None:
                continue

            Q = np.asarray(ply.Q126, dtype=float)
            A += Q * (z1 - z0)
            B += 0.5  * Q * (z1**2 - z0**2)
            D += (1.0 / 3.0) * Q * (z1**3 - z0**3)

        self.stiff_A = A
        self.stiff_B = B
        self.stiff_D = D
        self.stiff_ABD = np.block([[A, B], [B, D]])


    def _calculate_laminate_compliance(self) -> None:
        '''abcd compliance matrices as the inverse of ABD.'''
        self.compl_abcd = np.linalg.inv(self.stiff_ABD)

        self.compl_a = self.compl_abcd[:3,  :3]
        self.compl_b = self.compl_abcd[:3, 3:6]
        self.compl_c = self.compl_abcd[3:6, :3]
        self.compl_d = self.compl_abcd[3:6, 3:6]


    def _calculate_equivalent_engineering_constants(self) -> None:
        '''
        Equivalent engineering constants from both the membrane (a) and
        bending (d) compliance matrices.

        Membrane constants (A-based, suffix-less):
            E1  = 1 / (h * a[0, 0])          Tsai & Hahn Eq. 4.39
            E2  = 1 / (h * a[1, 1])
            G12 = 1 / (h * a[2, 2])

        Bending constants (D-based, _bend suffix):
            E1_bend  = 12 / (h^3 * d[0, 0])  Eq. analogous to D = Eh^3/12
            E2_bend  = 12 / (h^3 * d[1, 1])
            G12_bend = 12 / (h^3 * d[2, 2])

        For homogeneous isotropic plates the two sets are identical.
        For sandwich laminates the bending constants are larger because
        the D matrix is amplified by the core lever arm (z^2 weighting).
        For B = 0 symmetric laminates compl_d = D^{-1}; for coupled
        laminates compl_d is the full Schur complement, which is the
        physically correct bending compliance.

        Also builds the 3x3 equivalent membrane compliance and stiffness
        tensors (eng_compl, eng_stiff).
        '''
        h = self.thick
        a = self.compl_a
        d = self.compl_d

        # -------- Membrane (A-based) constants --------
        E1   = 1.0 / (h * a[0, 0])
        E2   = 1.0 / (h * a[1, 1])
        G12  = 1.0 / (h * a[2, 2])
        nu12 = -a[1, 0] / a[0, 0]
        nu21 = -a[0, 1] / a[1, 1]
        eta61 = a[0, 2] / a[2, 2]
        eta16 = a[2, 0] / a[0, 0]
        eta26 = a[2, 1] / a[1, 1]
        eta62 = a[1, 2] / a[2, 2]

        self.eng_compl = np.array([
            [1.0/E1,     -nu21/E2,   eta61/G12],
            [-nu12/E1,   1.0/E2,     eta62/G12],
            [eta16/E1,   eta26/E2,   1.0/G12  ],
        ])

        self.eng_stiff = np.linalg.inv(self.eng_compl)

        self.E1    = E1
        self.E2    = E2
        self.G12   = G12
        self.nu12  = nu12
        self.nu21  = nu21
        self.eta61 = eta61 if abs(eta61) > TOL else 0.0
        self.eta16 = eta16 if abs(eta16) > TOL else 0.0
        self.eta26 = eta26 if abs(eta26) > TOL else 0.0
        self.eta62 = eta62 if abs(eta62) > TOL else 0.0

        # -------- Bending (D-based) constants --------
        h3 = h ** 3
        self.E1_bend   = 12.0 / (h3 * d[0, 0])
        self.E2_bend   = 12.0 / (h3 * d[1, 1])
        self.G12_bend  = 12.0 / (h3 * d[2, 2])
        self.nu12_bend = -d[0, 1] / d[0, 0]
        self.nu21_bend = -d[0, 1] / d[1, 1]


    # ----------------------------------------
    # Private method - LaminateData packing
    # ----------------------------------------

    def _pack_laminate_data(self) -> LaminateData:
        '''Cast Laminate instance into a LaminateData container.'''
        return LaminateData(
            name  = self.name,
            plies = [
                MaterialHelper.get_ply_filename(ply.name, ply.thick, ply.angle)
                for ply in self.plies
            ],
            n_plies = len(self.plies),

            thick = float(self.thick),
            rho   = float(self.rho),  
            gsm   = float(self.gsm),

            stiff_A   = self.stiff_A.astype(float),  
            stiff_B   = self.stiff_B.astype(float),  
            stiff_D   = self.stiff_D.astype(float),  
            stiff_ABD = self.stiff_ABD.astype(float),

            compl_a    = self.compl_a.astype(float),   
            compl_b    = self.compl_b.astype(float),   
            compl_c    = self.compl_c.astype(float),   
            compl_d    = self.compl_d.astype(float),   
            compl_abcd = self.compl_abcd.astype(float),

            E1    = float(self.E1),   
            E2    = float(self.E2),   
            G12   = float(self.G12),  
            nu12  = float(self.nu12), 
            nu21  = float(self.nu21), 
            eta16 = float(self.eta16),
            eta26 = float(self.eta26),
            eta61 = float(self.eta61),
            eta62 = float(self.eta62),

            E1_bend   = float(self.E1_bend),
            E2_bend   = float(self.E2_bend),
            G12_bend  = float(self.G12_bend),
            nu12_bend = float(self.nu12_bend),
            nu21_bend = float(self.nu21_bend),

            eng_compl = self.eng_compl.astype(float),
            eng_stiff = self.eng_stiff.astype(float),

            stacking_seq = self.stacking_seq,
        )


    # ----------------------------------------
    # Public methods - entry points
    # ----------------------------------------

    def add_ply(
        self,
        name  : str,
        thick : float,
        angle : float,
        core  : bool = False,
        rho   : float | None = None,
        Ex    : float | None = None,
        Ey    : float | None = None,
        Es    : float | None = None,
        nux   : float | None = None,
        Xt    : float | None = None,
        Xc    : float | None = None,
        Yt    : float | None = None,
        Yc    : float | None = None,
        S     : float | None = None,
        enable_logging: bool = True
    ) -> None:
        '''
        Add a ply on top of the current layup (bottom-to-top order).

        If a JSON file for already exists in the database, it is loaded
        directly. Otherwise a new Ply is computed and persisted. Core
        plies only require name, thick, and rho.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        ply_filename    = MaterialHelper.get_ply_filename(name, thick, angle)
        ply_db_filepath = _DFLT_PLY_DIR / f"PlyData_{ply_filename}.json"

        if ply_db_filepath.exists():
            self.logger.info(f"'{ply_filename}' found in Database.")
            
            ply_data: PlyData = io.read_json(
                filepath=ply_db_filepath,
                dcls=PlyData,
            )
        else:
            self.logger.info(f"Creating new Ply in Database: {ply_db_filepath}")

            ply_obj = Ply(
                name  = name,
                thick = thick,
                rho   = rho,
                core  = core,
                angle = angle,
                Ex=Ex, Ey=Ey, Es=Es, nux=nux,
                Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S=S,
            )
            ply_data = ply_obj.ply_data

        self.plies.append(ply_data)
        self.logger.info(f"Ply '{ply_filename}'"
                         f" added (total: {len(self.plies)}).")


    def define_laminate_data(self) -> None:
        '''
        Run all laminate calculations and persist the result to disk.

        Skips computation if the JSON file already exists in db_filepath.
        '''
        self._write_stacking_sequence()

        self.db_filepath = _DFLT_MAT_DIR / f"{self.name}_LaminateData.json"
        self.logger.info("-" * 80)

        self._get_total_thickness()
        self._calculate_laminate_density()
        self._calculate_z_coords()
        self._calculate_laminate_stiffness()
        self._calculate_laminate_compliance()
        self._calculate_equivalent_engineering_constants()

        self.mat_data = self._pack_laminate_data()

        io.write_json(
            obj=self.mat_data,
            filepath=self.db_filepath,
        )
        self.logger.info(f"Laminate data written to: {self.db_filepath}")



# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
