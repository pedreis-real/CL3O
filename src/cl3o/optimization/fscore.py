'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Structural Mass Module.

Computes the structural mass (kg) of a beam-element wing model by
summing, for every element, the material mass of its ten T2 sub-panels
and its four boom flanges:

    mu_panels_i  = Sum_k (rho_k  * t_k  * s_k )           [t/mm]
    mu_flanges_i = Sum_j (rho_fj * A_fj       )           [t/mm]
    mass_elem_i  = T_TO_KG * 0.5 * (mu_A + mu_B) * L_i    [kg]
    mass_total   = Sum_i mass_elem_i                      [kg]

The factor T_TO_KG converts the CL3O internal tonne unit (density is
stored in t/mm^3 and the product t/mm^3 * mm^3 returns tonnes) into
kilograms for the DE objective function.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from dataclasses import dataclass, field

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    T_TO_KG, N_PANELS, N_FLANGES, T2_TO_T1
)

# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.section_builder import SectionData

# Materials
from cl3o.materials.laminate import LaminateData, MaterialHelper


# ================================================================================
# Data persistence - Mass breakdown
# ================================================================================

@dataclass
class ScoreData:
    '''
    Container for the structural mass of every beam element and its
    global sum.

    Property    Size        Description                         Units
    --------    --------    --------------------------------    --------
    m           (1,)        Number of beam elements             -
    total       (1,)        Total structural mass               kg
    panels      (m,)        Panel mass per element              kg
    flanges     (m,)        Flange mass per element             kg
    per_elem    (m,)        Total mass per element              kg
    '''
    m        : int   = 0
    total    : float = 0.0
    panels   : np.ndarray = field(default_factory=lambda: np.zeros(0))
    flanges  : np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_elem : np.ndarray = field(default_factory=lambda: np.zeros(0))


# ================================================================================
# PUBLIC API - Structural mass
# ================================================================================

class StructuralMass:
    '''
    Evaluate the structural mass of the wing beam model.

    Use:
        score = StructuralMass(sections, element_idx, laminate_db)
        data  = score.data                                # ScoreData
        mass  = score.data.total                          # kg
    '''

    def __init__(
        self,
        sections       : SectionData,
        element_idx    : np.ndarray,
        laminate_db    : dict[str, LaminateData],
        enable_logging : bool = True,
        verbose        : bool = False,
    ) -> None:
        '''
        Args:
            sections      : SectionData with sec_data, lam_T1, lam_T4.
            element_idx   : (m, 2) per-element node-pair indices, with
                node index aligned to the station index used to build
                sections.sec_data.
            laminate_db   : Dict mapping 'MAT{k}' to LaminateData.
            enable_logging: Toggle logger.
            verbose       : When True, log at DEBUG level.
        '''
        self.logger = io.setup_logger(self, enable_logging, verbose)

        # Store inputs
        self.sec_data    = sections.sec_data
        self.lam_T1      = np.asarray(sections.lam_T1, dtype=int)
        self.lam_T4      = np.asarray(sections.lam_T4, dtype=int)
        self.element_idx = np.asarray(element_idx,     dtype=int)
        self.laminate_db = laminate_db

        # Calculate mass
        self._evaluate()

    # --------------------------------------------------------
    # Private methods - Mass math
    # --------------------------------------------------------

    def _get_laminate(self, lam_idx: int) -> LaminateData:
        '''Retrieve LaminateData from the database by integer key.'''
        return MaterialHelper.laminate_by_index(self.laminate_db, lam_idx)

    def _element_length(self, idx_a: int, idx_b: int) -> float:
        '''Euclidean length between two section centroids.'''
        Ca = np.asarray(self.sec_data[idx_a].C, dtype=float)
        Cb = np.asarray(self.sec_data[idx_b].C, dtype=float)
        return float(np.linalg.norm(Cb - Ca))

    def _panel_mass_per_length(self, sta_idx: int) -> float:
        '''
        Panel mass per unit span at station 'sta_idx':

            Sum_k rho_k * A_k            [t/mm]
        '''
        sec   = self.sec_data[sta_idx]
        A_k   = np.asarray(sec.A_k, dtype=float).ravel()
        total = 0.0
        for k in range(N_PANELS):
            seg_idx = int(T2_TO_T1[k])
            lam     = self._get_laminate(self.lam_T1[sta_idx, seg_idx])
            total  += float(lam.rho) * float(A_k[k])
        return total

    def _flange_mass_per_length(self, sta_idx: int) -> float:
        '''
        Flange mass per unit span at station 'sta_idx':

            Sum_j rho_fj * A_fj                [t/mm]
        '''
        sec   = self.sec_data[sta_idx]
        A_f   = np.asarray(sec.A_flange, dtype=float).ravel()
        total = 0.0
        for j in range(N_FLANGES):
            lam    = self._get_laminate(self.lam_T4[sta_idx, j])
            total += float(lam.rho) * float(A_f[j])
        return total

    # --------------------------------------------------------
    # Private method - Mass evaluation pipeline
    # --------------------------------------------------------

    def _evaluate(self) -> None:
        '''Element loop accumulating panel and flange mass in kg.'''
        m = int(self.element_idx.shape[0])

        mass_panels  = np.zeros(m, dtype=float)
        mass_flanges = np.zeros(m, dtype=float)

        for i in range(m):
            a = int(self.element_idx[i, 0])
            b = int(self.element_idx[i, 1])

            L_i  = self._element_length(a, b)

            mu_p = 0.5 * (self._panel_mass_per_length (a)
                          + self._panel_mass_per_length (b))
            mu_f = 0.5 * (self._flange_mass_per_length(a)
                          + self._flange_mass_per_length(b))

            mass_panels [i] = T_TO_KG * mu_p * L_i
            mass_flanges[i] = T_TO_KG * mu_f * L_i

        per_elem   = mass_panels + mass_flanges
        mass_total = float(np.sum(per_elem))

        self.logger.debug(
            f"Structural mass evaluated [elements={m}].\n"
            f"| panels   : {float(np.sum(mass_panels)) :.4f} kg\n"
            f"| flanges  : {float(np.sum(mass_flanges)):.4f} kg\n"
            f"| total    : {mass_total:.4f} kg"
        )

        # -------- Pack results --------
        self.data = ScoreData(
            m        = m,
            total    = mass_total,
            panels   = mass_panels,
            flanges  = mass_flanges,
            per_elem = per_elem,
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
