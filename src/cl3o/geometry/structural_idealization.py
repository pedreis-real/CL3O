'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Structural Idealization Module.

Computes boom areas (T4 idealization) and updates geometric properties
(centroid, inertia, principal axes) for a structurally idealized section.

Boom topology
-------------
The cross-section uses a 7-boom T2' set:

    B1 = upper rear-spar cap     (carries flange Af3)
    B2 = upper mid-skin point    (carries stringer As2, zero for now)
    B3 = upper front-spar cap    (carries flange Af1)
    B4 = leading edge
    B5 = lower front-spar cap    (carries flange Af2)
    B6 = lower mid-skin point    (carries stringer As6, zero for now)
    B7 = lower rear-spar cap     (carries flange Af4)

TE is a geometric endpoint but not a boom. Each of the 10 T2
sub-panels straddles exactly two booms, so Megson's pair-attribution
lemma applies to every sub-panel directly. The upper-rear and
lower-rear skin panels (r1, r8) both connect B1 and B7, with TE as
a geometric midpoint only.

Pipeline (executed by StructuralIdealization.run()):
  1. Extract the 8 boom positions from GeomData.boom_pos.
  2. For every T2 sub-panel apply the Megson pair contribution
        B_n += (t*L/6) * (2 + sigma_m / sigma_n)
        B_m += (t*L/6) * (2 + sigma_n / sigma_m)
     using the pre-idealization centroid and bending stress proxy.
  3. Add flange areas at B2, B4, B6, B8 and stringer areas at B3, B7.
  4. Recalculate centroid, inertia, and principal axes from the 8-boom
     T2' set.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ================ Pathing ================


# ================ Module imports ================

# Constants
from cl3o.Constants import (
    TOL,
    BOOM_LBLS,
    N_BOOMS,
    FLANGE_BOOM_IDX,
    STRINGER_BOOM_IDX,
)

# Utilities
from cl3o.utils import io_utils as io
from cl3o.utils import math_utils as mthu


# ================================================================================
# Data persistence - Boom idealization output
# ================================================================================

@dataclass
class BoomData:
    '''
    Output of the T4' structural idealization for a single cross-section.

    Property        Size      Description                                   Unit
    --------        ----      ----------------------------------------  ----
    Xc              (1,)      Centroid X in global frame                mm
    Zc              (1,)      Centroid Z in global frame                mm
    I_XX            (1,)      Area moment of inertia about X            mm^4
    I_ZZ            (1,)      Area moment of inertia about Z            mm^4
    I_XZ            (1,)      Product of inertia                        mm^4
    I_1             (1,)      Principal inertia 1                       mm^4
    I_2             (1,)      Principal inertia 2                       mm^4
    theta_P         (1,)      Principal axis angle                      rad
    A_cells         (3,)      Enclosed areas of cells I, II, III        mm^2
    J               (1,)      Torsional constant                        mm^4
    boom_labels     (7,)      Boom labels [B1..B7]                      -
    boom_u          (7,)      Boom u-coordinates (centroidal)           mm
    boom_w          (7,)      Boom w-coordinates (centroidal)           mm
    boom_A          (7,)      Boom areas                                mm^2
    boom_rat        dict      Per-boom list of (2+sigma_far/sigma_self)/6 -
    '''
    Xc          : float      = 0.0
    Zc          : float      = 0.0
    I_XX        : float      = 0.0
    I_ZZ        : float      = 0.0
    I_XZ        : float      = 0.0
    I_1         : float      = 0.0
    I_2         : float      = 0.0
    theta_P     : float      = 0.0
    A_cells     : np.ndarray = field(default_factory=lambda: np.zeros(3))
    J           : float      = 0.0
    boom_labels : tuple      = BOOM_LBLS
    boom_u      : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_w      : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_A      : np.ndarray = field(default_factory=lambda: np.zeros(N_BOOMS))
    boom_rat    : dict       = field(
        default_factory=lambda: {lbl: [] for lbl in BOOM_LBLS}
    )


@dataclass
class ShearFluxData:
    '''
    Shear flux per unit force for all T1 panels, and intermediate values.

    Property    Size    Description                                     Unit
    --------    ----    ----------------------------------------    --------
    qT_star     (n,)    Total q* for unit pure torque               1/mm
    qsX_star    (n,)    Total q* per panel, S_X load                1/mm
    qsZ_star    (n,)    Total q* per panel, S_Z load                1/mm
    us          (1,)    Shear centre u-coordinate (centroidal)      mm
    ws          (1,)    Shear centre w-coordinate (centroidal)      mm
    xi0         (1,)    Moment arm from P1 to SC (u direction)      mm
    eta0        (1,)    Moment arm from P1 to SC (w direction)      mm
    '''
    qT_star  : np.ndarray = field(default_factory=lambda: np.zeros(0))
    qsX_star : np.ndarray = field(default_factory=lambda: np.zeros(0))
    qsZ_star : np.ndarray = field(default_factory=lambda: np.zeros(0))
    us       : float      = 0.0
    ws       : float      = 0.0
    xi0      : float      = 0.0
    eta0     : float      = 0.0


# ================================================================================
# PUBLIC API - Structural idealization calculator
# ================================================================================

class StructuralIdealization:
    '''
    Computes T4' boom areas (10-boom topology) and updated geometric
    properties for a 3-cell closed cross-section, using Megson
    pair-attribution per T2 sub-panel.
    '''

    def __init__(
        self,
        geom_data     : object,
        enable_logging: bool = True,
    ) -> None:
        self.logger = io.setup_logger(self, enable_logging)
        self._gd = geom_data
        self._boom_pos = self._build_boom_index()

    # ----------------------------------------------------------------
    # Private - Boom indexing
    # ----------------------------------------------------------------

    def _build_boom_index(self) -> np.ndarray:
        '''
        Pick up the (7, 2) array of boom XZ positions directly from the
        owning GeomPropCalculator (canonical order B1..B7).
        '''
        boom_pos = np.asarray(self._gd.boom_pos, dtype=float).copy()
        if boom_pos.shape != (N_BOOMS, 2):
            raise ValueError(
                f"[CL3O] boom_pos must have shape ({N_BOOMS}, 2).\n"
                f"| received : {boom_pos.shape}\n"
                f"Check GeomPropCalculator._find_spar_intersections."
            )
        return boom_pos
    
    # ----------------------------------------------------------------
    # Private - Stress ratio and panel contribution
    # ----------------------------------------------------------------

    @staticmethod
    def _stress_ratio(
        um  : float,
        wm  : float,
        un  : float,
        wn  : float,
        I_ZZ: float,
        I_XZ: float,
    ) -> float:
        '''
        Bending stress ratio sigma_m / sigma_n for pure M_X bending.
        Preserves I_XX after Megson pair-attribution.
        '''
        num = -I_XZ * um + I_ZZ * wm
        den = -I_XZ * un + I_ZZ * wn
        return (num / den)

    @staticmethod
    def _panel_contrib(tL: float, ratio: float) -> float:
        '''Megson boom share for a panel: (t*L/6) * (2 + sigma_other/sigma_self).'''
        return (tL / 6.0) * (2.0 + ratio)

    # ----------------------------------------------------------------
    # Public - Boom areas (Megson pair attribution per T2 sub-panel)
    # ----------------------------------------------------------------

    def _compute_boom_areas(self) -> np.ndarray:
        '''
        Compute boom areas for the 7-boom T2' topology by applying Megson
        pair-attribution to every T2 sub-panel:

            B_n += (t*L/6) * (2 + sigma_m / sigma_n)
            B_m += (t*L/6) * (2 + sigma_n / sigma_m)

        Flanges (A_flange, size 4) lump at B3, B5, B1, B7 in canonical
        order [F1..F4] = [upper-front, lower-front, upper-rear,
        lower-rear]. Stringers lump at B2, B6 (zero placeholders).

        Special case — rear skin (r1, r8): TE is a geometric endpoint
        but not a boom, so r1 (B1→TE) and r8 (TE→B7) are combined into
        a single Megson step for the (B1, B7) pair using
        tL = t_r1*s_r1 + t_r8*s_r8.  This avoids two separate
        attributions with the same stress ratio and gives boom_rat a
        single clean entry per boom from the rear skin.

        Returns:
            boom_A: (7,) array of boom areas in canonical order
                    [B1..B7] in mm^2.
        '''
        gd   = self._gd
        Xc   = gd.Xc
        Zc   = gd.Zc
        I_ZZ = gd.I_ZZ
        I_XZ = gd.I_XZ

        # Centroidal boom coordinates
        u_boom = self._boom_pos[:, 0] - Xc
        w_boom = self._boom_pos[:, 1] - Zc

        # Initialize boom areas with flange and stringer lumps
        boom_A = np.zeros(N_BOOMS, dtype=float)
        for j, b in enumerate(FLANGE_BOOM_IDX):
            boom_A[b] += float(gd.A_flange[j])
        for j, b in enumerate(STRINGER_BOOM_IDX):
            boom_A[b] += float(gd.A_stringer[j])

        # Per-boom record of (2 + sigma_other/sigma_self)/6 coefficients
        boom_rat = {lbl: [] for lbl in BOOM_LBLS}

        def _apply(n_idx: int, m_idx: int, tL: float) -> None:
            r_mn = self._stress_ratio(
                u_boom[m_idx], w_boom[m_idx],
                u_boom[n_idx], w_boom[n_idx],
                I_ZZ, I_XZ,
            )
            r_nm = 1.0 / r_mn
            boom_A[n_idx] += self._panel_contrib(tL, r_mn)
            boom_A[m_idx] += self._panel_contrib(tL, r_nm)
            boom_rat[BOOM_LBLS[n_idx]].append(float((2.0 + r_mn) / 6.0))
            boom_rat[BOOM_LBLS[m_idx]].append(float((2.0 + r_nm) / 6.0))

        # -------- Regular panels (8 of 10, skip rear skin r1/r8) --------
        for r in gd.T2:
            if r['label'] in ('r1', 'r8'):
                continue
            _apply(int(r['boomA']), int(r['boomB']),
                   float(r['t']) * float(r['s']))

        # -------- Combined rear-skin panel (r1 + r8 → B1/B7 pair) --------
        # r1 and r8 share the TE free endpoint; combining them into a
        # single (B1, B7) attribution gives one clean boom_rat entry and
        # correctly represents the full rear skin arc.
        rear = {r['label']: r for r in gd.T2 if r['label'] in ('r1', 'r8')}
        tL_rear = (
            float(rear['r1']['t']) * float(rear['r1']['s'])
            + float(rear['r8']['t']) * float(rear['r8']['s'])
        )
        _apply(0, 6, tL_rear)    # B1 (idx 0) and B7 (idx 6)

        self.boom_A   = boom_A
        self.boom_rat = boom_rat

    # ----------------------------------------------------------------
    # Public - Recalculate geometry with T4'
    # ----------------------------------------------------------------

    def _recalculate_geometry(self) -> BoomData:
        '''
        Recalculate centroid, inertia, and principal axes from the
        7-boom T2' set.

        Returns:
            BoomData with updated centroid, inertia, and boom coordinates.
        '''
        bp = self._boom_pos
        (self.Xc, self.Zc,
         self.I_XX, self.I_ZZ, self.I_XZ) = mthu.discrete_section_properties(
            bp[:, 0], bp[:, 1], self.boom_A,
        )
        (self.I_1, self.I_2, self.theta_P) = mthu.mohr_circle(
            Ixx=self.I_XX,
            Iyy=self.I_ZZ,
            Ixy=self.I_XZ,
        )

    def _pack_boom_data(self):
        return BoomData(
            Xc          = self.Xc,
            Zc          = self.Zc,
            I_XX        = self.I_XX,
            I_ZZ        = self.I_ZZ,
            I_XZ        = self.I_XZ,
            I_1         = self.I_1,
            I_2         = self.I_2,
            theta_P     = self.theta_P,
            A_cells     = np.asarray(self._gd.A_cells),
            J           = float(self._gd.J),
            boom_labels = BOOM_LBLS,
            boom_u      = self._boom_pos[:, 0] - self.Xc,
            boom_w      = self._boom_pos[:, 1] - self.Zc,
            boom_A      = self.boom_A,
            boom_rat    = self.boom_rat,
        ) if self._gd.recalculate_props else BoomData(
            Xc          = self._gd.Xc,
            Zc          = self._gd.Zc,
            I_XX        = self._gd.I_XX,
            I_ZZ        = self._gd.I_ZZ,
            I_XZ        = self._gd.I_XZ,
            I_1         = self._gd.I_1,
            I_2         = self._gd.I_2,
            theta_P     = self._gd.theta_P,
            A_cells     = np.asarray(self._gd.A_cells),
            J           = float(self._gd.J),
            boom_labels = BOOM_LBLS,
            boom_u      = self._boom_pos[:, 0] - self._gd.Xc,
            boom_w      = self._boom_pos[:, 1] - self._gd.Zc,
            boom_A      = self.boom_A,
            boom_rat    = self.boom_rat,
        )
    

    # ----------------------------------------------------------------
    # Public - Run pipeline
    # ----------------------------------------------------------------

    def run(self) -> BoomData:
        '''
        Execute the structural idealization pipeline.

        Returns:
            BoomData with boom areas and updated geometric properties.
        '''
        self.logger.info("Running structural idealization (7-boom T4')")
        
        self._compute_boom_areas()
        if self._gd.recalculate_props:
            self._recalculate_geometry()
        return self._pack_boom_data()


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
