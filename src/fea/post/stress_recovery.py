'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Stress Recovery Module.

Recovers boom normal stresses and panel shear stresses from internal forces
obtained by the static analysis solution. Uses pre-computed bending moment and
shear fluxes influence coefficients stored in GeomData. Stresses are evaluated
at both the begin and end section of every element.

Pipeline
----------------
    1. Extract geometrical properties for each element
    2. Extract internal forces at both ends (LOCAL frame) from two sources:
        Q_c  (local at centroid)     -> N at [0], My at [4], Mz at [5]
        Q_sc (local at shear centre) -> Sy at [1], Sz at [2], T at [3]
        End-section uses offset +6 with a sign flip.
    3. Map to downstream names (y <- X, z <- Z):
        N  := Q_c [0],  MX := Q_c [4],  MZ := Q_c [5]
        SX := Q_sc[1],  SZ := Q_sc[2],  T  := Q_sc[3]
    4. Calculate normal stress acting at each boom:
        sigma =   N / A
                + IXstar * MX
                + IZstar * MZ
    5. Compute total shear flux at each panel (T2 topology):
        q = qsX_star * SX + qsZ_star * SZ + qT_star * T
    6. Compute shear stress acting at each panel (T2 topology):
        tau = q / t
    7. Store all in StressData

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================
_HERE = Path(__file__).resolve().parent           # src/fea/post/
_SRC  = _HERE.parent.parent                       # src/
_ROOT = _SRC.parent                               # project root

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ================ Module imports ================

# Constants
from Constants import N_BOOMS, N_PANELS

# Utilities
from utils import io_utils as io

# Geometry
from geometry.geom_properties import GeomData
from geometry.section_builder import SectionData

# FEA
from fea.solver.static_analysis import FeaResults


# ================================================================================
# Data persistence - Recovered stresses per element
# ================================================================================

@dataclass
class StressData:
    '''
    Container for the recovered stresses at the begin and end sections of
    every beam element.

    Property    Size            Description                             Units
    --------    ------------    ------------------------------------    --------
    n           (1,)            Number of nodes                         -
    m           (1,)            Number of elements                      -
    sigmaA      (m, nb, 2)      Normal stresses at booms (endA)         MPa
    sigmaB      (m, nb, 2)      Normal stresses at booms (endB)         MPa
    qA          (m, ns, 2)      Shear flow per panel (endA)             N/mm
    qB          (m, ns, 2)      Shear flow per panel (endB)             N/mm
    tauA        (m, ns, 2)      Shear stresses per panel (endA)         MPa
    tauB        (m, ns, 2)      Shear stresses per panel (endB)         MPa

    sigma       tuple           Both ends direct stresses               MPa
    q           tuple           Both ends shear flows                   N/mm
    tau         tuple           Both ends shear stresses                MPa

    *nb  -> number of booms
    **ns -> number of T2 segments
    '''
    n  : int = 0
    m  : int = 0
    
    sigmaA : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_BOOMS, 2))
        )
    sigmaB : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_BOOMS, 2))
        )
    qA     : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2))
        )
    qB     : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2))
        )
    tauA   : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2))
        )
    tauB   : np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PANELS, 2))
        )
    
    sigma : tuple[np.ndarray, np.ndarray] = None
    q     : tuple[np.ndarray, np.ndarray] = None
    tau   : tuple[np.ndarray, np.ndarray] = None



# ================================================================================
# PUBLIC API - Stress recovery
# ================================================================================

class StressRecovery:
    '''
    Recovers boom axial stresses and panel shear stresses at every begin
    and end section of every element, packing the result into StressData.
    '''

    def __init__(
        self,
        sections : SectionData,
        element_idx : np.ndarray,
        fea_results : FeaResults,
        use_local : bool = True,
        enable_logging : bool = True,
    ) -> None:
        '''
        Args:
            sections    : SectionData container with sec_data (list[GeomData])
                          per spanwise station.
            element_idx : (m, 2) array mapping element to (endA, endB) node
                          indices in sec_data. Equivalent to mesh.conn[:, :2].
            fea_results : FeaResults from LinearStaticSolver.
        '''
        self.logger = io.setup_logger(self, enable_logging)

        # 1.Retrieve inputs
        self.use_local = use_local
        self.sec = np.array(sections.sec_data, dtype=object)
        self.idx = element_idx
        self.fea = fea_results

        # Run
        self._recover()

    # ----------------------------------------
    # Private - Auxiliary functions
    # ----------------------------------------

    def _extract_internal_forces(
        self,
        Qc_e  : np.ndarray,
        Qsc_e : np.ndarray,
        Qc_gl_e  : np.ndarray,
        Qsc_gl_e : np.ndarray,
        at_end : bool,
    ) -> tuple[float, float, float, float, float, float]:
        '''
        Extract internal forces for a single element at either
        the begin (at_end=False) or end (at_end=True) section, from two
        sources consistent with the thin-walled-beam formulation:

            From Q_c  (local at centroid)     : N  = Q[0], My = Q[4], Mz = Q[5]
            From Q_sc (local at shear centre) : Sy = Q[1], Sz = Q[2], T  = Q[3]

        Mapping to downstream names follows y <- X, z <- Z:
            N  := Q_c [0],  MX := Q_c [4],  MZ := Q_c [5]
            SX := Q_sc[1],  SZ := Q_sc[2],  T  := Q_sc[3]

        At the end section, the MSA convention stores forces-on-element from
        the end node, which require a sign flip to obtain the outboard
        section force.

        Args:
            Q_c_e  : (12,) internal forces at centroid, local or global frame.
            Q_sc_e : (12,) internal forces at shear centre, local or global frame.
            at_end : False -> begin section; True -> end section.

        Returns:
            Tuple (N, SX, SZ, T, MX, MZ).
        '''
        off  = 6 if at_end else 0
        sign = -1.0 if at_end else 1.0

        N  = sign * float(Qc_e [0 + off])       # Always from local frame
        if self.use_local:
            M1 = sign * float(Qc_e [4 + off])
            M2 = sign * float(Qc_e [5 + off])

            S1 = sign * float(Qsc_e[1 + off])
            S2 = sign * float(Qsc_e[2 + off])
            T  = sign * float(Qsc_e[3 + off])
        else:
            M1 = sign * float(Qc_gl_e [3 + off])
            M2 = sign * float(Qc_gl_e [5 + off])
            
            S1 = sign * float(Qsc_gl_e[0 + off])
            S2 = sign * float(Qsc_gl_e[2 + off])
            T  = sign * float(Qsc_gl_e[4 + off])

        return (N, S1, S2, T, M1, M2)
    
    @staticmethod
    def _compute_boom_normal_stress(
        sec : GeomData,
        N  : float,
        MX : float,
        MZ : float,
    ) -> np.ndarray:
        '''
        Direct stress at every boom position 

        Args:
            sec: Section GeomData.
            N  : Axial force [N].
            MX : Bending moment about global X axis [N*mm].
            MZ : Bending moment about global Z axis [N*mm].

        Returns:
            sigma_booms : direct stresses in MPa.
        '''
        sigma  = (N / sec.A)
        sigma += sec.IXstar * MX
        sigma += sec.IZstar * MZ
        return sigma
    
    @staticmethod
    def _compute_total_shear_flow(
        sec : GeomData,
        SX : float,
        SZ : float,
        T  : float,
    ) -> np.ndarray:
        '''
        Shear flow per panel, linear superposition of unit-force flows:

            q_k = q*_SX_k * S_X + q*_SZ_k * S_Z + q*_T_k * T

        Args:
            sec : Section GeomData with qsX_star, qsZ_star, qT_star.
            SX  : Shear force along global X [N].
            SZ  : Shear force along global Z [N].
            T   : Torsional moment about global Y [N*mm].

        Returns:
            q (10,) shear flow [N/mm].
        '''
        qsX = np.asarray(sec.qsX_star, dtype=float)
        qsZ = np.asarray(sec.qsZ_star, dtype=float)
        qsT = np.asarray(sec.qT_star,  dtype=float)
        return (qsX * SX + qsZ * SZ + qsT * T).astype(float)

    @staticmethod
    def _compute_panel_shear_stress(
        q : np.ndarray,
        t : np.ndarray,
    ) -> np.ndarray:
        '''
        Shear stress per panel: tau_k = q_k / t_k.

        Args:
            q: (10,) shear flow [N/mm].
            t: (10,) panel thicknesses [mm].

        Returns:
            tau (10,) in MPa.
        '''
        return (
            np.asarray(q, dtype=float) / \
            np.asarray(t, dtype=float)
        )
    

    # ----------------------------------------
    # Private - Recovery pipeline
    # ----------------------------------------

    def _recover(self) -> None:
        '''
        Compute per element stresses at both ends:
            Begining (endA) : arr_[:, :, 0]
            End      (endB) : arr_[:, :, 1]

        2. Extract internal forces at both ends (LOCAL frame) from two
           sources, then sign-flip the end-section block:
            From Q_c  : N  = Q_c [0+off], MX = Q_c [4+off], MZ = Q_c [5+off]
            From Q_sc : SX = Q_sc[1+off], SZ = Q_sc[2+off], T = Q_sc[3+off]
            with off=0 for endA (sign=+1) and off=6 for endB (sign=-1).
        3. (Mapping handled inline by _extract_internal_forces.)
        4. Calculate normal stress acting at each boom:
            sigma = N / A + IXstar * MX + IZstar * MZ
        5. Compute total shear flux at each panel (T2 topology):
            q = qsX_star * SX + qsZ_star * SZ + qT_star * T
        6. Compute shear stress acting at each panel (T2 topology):
            tau = q / t
        7. Store all in StressData

        The correspondent section of each member is driven by 'conn' array.
        See :fem_setup: to learn more about.
        '''
        m = self.fea.m
        nc = self.fea.nc

        self.logger.info(
            f"Recovering stresses "
            f"[m={m} elements, nc={nc} load conditions]"
        )

        endA = self.idx[:, 0]
        endB = self.idx[:, 1]
        secA = self.sec[endA]
        secB = self.sec[endB]

        sigmaA = np.zeros((m, N_BOOMS, nc))
        qA     = np.zeros((m, N_PANELS, nc))
        tauA   = np.zeros((m, N_PANELS, nc))
        sigmaB = np.zeros((m, N_BOOMS, nc))
        qB     = np.zeros((m, N_PANELS, nc))
        tauB   = np.zeros((m, N_PANELS, nc))

        Qc  = self.fea.Q_c     # (12, m, nc) local at centroid
        Qsc = self.fea.Q_sc    # (12, m, nc) local at shear centre
        Qc_gl  = self.fea.Q_c_gl     # (12, m, nc) global at centroid
        Qsc_gl = self.fea.Q_sc_gl    # (12, m, nc) global at shear centre

        for i in range(m):

            secA_i = secA[i]
            secB_i = secB[i]
            Qc_i  = Qc [:, i, :]
            Qsc_i = Qsc[:, i, :]
            Qc_gl_i  = Qc_gl [:, i, :]
            Qsc_gl_i = Qsc_gl[:, i, :]
            for j in range(nc):

                # ---------------- #
                #       endA       #
                # ---------------- #

                # Step 2-3: Extract internal forces
                (N, SX, SZ, T, MX, MZ) = self._extract_internal_forces(
                    Qc_e  = Qc_i [:, j],
                    Qsc_e = Qsc_i[:, j],
                    Qc_gl_e  = Qc_gl_i [:, j],
                    Qsc_gl_e = Qsc_gl_i[:, j],
                    at_end = False,
                )

                # Step 4: Normal stress
                sigma = self._compute_boom_normal_stress(
                    sec=secA_i,
                    N=N,
                    MX=MX,
                    MZ=MZ,
                )

                # Step 5: Shear flow
                q = self._compute_total_shear_flow(
                    sec=secA_i,
                    SX=SX,
                    SZ=SZ,
                    T=T,
                )

                # Step 6: Shear stress
                tau = self._compute_panel_shear_stress(
                    q=q,
                    t=secA_i.t_k
                )

                # Step 7: Pack (first)
                sigmaA[i, :, j] = sigma
                qA[i, :, j]     = q
                tauA[i, :, j]   = tau

                # ---------------- #
                #       endB       #
                # ---------------- #

                # Step 2-3: Extract / explicit internal forces
                (N, SX, SZ, T, MX, MZ) = self._extract_internal_forces(
                    Qc_e  = Qc_i [:, j],
                    Qsc_e = Qsc_i[:, j],
                    Qc_gl_e  = Qc_gl_i [:, j],
                    Qsc_gl_e = Qsc_gl_i[:, j],
                    at_end = True,
                )

                # Step 4: Normal stress
                sigma = self._compute_boom_normal_stress(
                    sec=secB_i,
                    N=N,
                    MX=MX,
                    MZ=MZ,
                )

                # Step 5: Shear flow
                q = self._compute_total_shear_flow(
                    sec=secB_i,
                    SX=SX,
                    SZ=SZ,
                    T=T,
                )

                # Step 6: Shear stress
                tau = self._compute_panel_shear_stress(
                    q=q,
                    t=secB_i.t_k
                )

                # Step 7: Pack (initial)
                sigmaB[i, :, j] = sigma
                qB[i, :, j]     = q
                tauB[i, :, j]   = tau

        self.logger.debug(
            f"Stress summary — "
            f"sigma max: {max(np.max(np.abs(sigmaA)), np.max(np.abs(sigmaB))):.2f} MPa | "
            f"tau max: {max(np.max(np.abs(tauA)), np.max(np.abs(tauB))):.2f} MPa"
        )

        # Step 7: Pack (final)
        self.data = StressData(
            n      = self.fea.n,
            m      = m,
            sigmaA = sigmaA,
            sigmaB = sigmaB,
            qA     = qA    ,
            qB     = qB    ,
            tauA   = tauA  ,
            tauB   = tauB  ,
            sigma  = (sigmaA, sigmaB),
            q      = (qA, qB),
            tau    = (tauA, tauB),
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
