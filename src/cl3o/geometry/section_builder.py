'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Section Builder Module.

Build and store cross section points for all spanwise stations.

The cross section of each station is obtained by the following steps:
    - Interpolate opt_vars:
        -- xw1, xw2           ..... 1 per Y_sta
        -- bf1, bf2, bf3, bf4 ..... 1 per Y_sta
        -- ls1, ls2           ..... 1 per Y_cp (the values step out
                                    at each Y_sta)
        -- lf1, lf2, lf3, lf4 ..... 1 per Y_cp
        -- lw1, lw2           ..... 1 per Y_cp
    - Blend airfoil data from adjacent control points, by linear
      interpolation at each Y_sta;
    - Call GeomPropCalculator for each station, which:
        -- Builds the dimensional cross section (T1/T2/T3 topologies)
        -- Computes centroid, inertia, boom areas, shear center,
           and shear flux

T1 Segmentation topology: Base segments
    (1) : nose skin         .. B5 -> B4 -> B3   [cell-I outer]
    (2) : upper middle skin .. B3 -> B2 -> B1   [cell-II top]
    (3) : upper rear skin   .. B1 -> TE         [cell-III top]
    (4) : lower rear skin   .. TE -> B7         [cell-III bottom]
    (5) : lower middle skin .. B7 -> B6 -> B5   [cell-II bottom]
    (6) : front spar web    .. B5 -> B3         [spar 1 - I_II intersection]
    (7) : rear spar web     .. B7 -> B1         [spar 2 - II_III intersection]

T1 material mapping:
    seg1   (nose skin)      -> ls1
    seg2-5 (box skins)      -> ls2
    seg6   (front spar web) -> lw1
    seg7   (rear spar web)  -> lw2

T2 Segmentation topology: Cutting cross-section at boom positions

T3 Segmentation topology: Closed cells points,
    oriented couter-clockwise.

T4 Segmentation topology: Flanges
    F1: B3 (upper front) -> lf1
    F2: B5 (lower front) -> lf2
    F3: B1 (upper rear)  -> lf3
    F4: B7 (lower rear)  -> lf4

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import numpy as np

# ================ Paths bootstrap ================


# ================ Module imports ================

# Constants
from cl3o.Constants import N_SEG_T1, N_FLANGES

# Utilities
from cl3o.utils import io_utils as io

# Geometry
from cl3o.geometry.geom_properties import GeomPropCalculator, GeomData

# ================ Global variables ================
_TOL = 1e-16



# ========================================================================
# Data persistance - Container for wrapping all section properties
# ========================================================================

@dataclass
class SectionData:
    '''
    Container for storing the geometrical properties of the cross-section
    calculated by GeomPropCalculator, as well as laminate bundles to
    retrieve the laminate data in post processing features.

    Property    Size            Description
    --------    ------------    ------------------------------------------------
    sec_data    (n_sta,)        GeomData of every cross-section
    n_sta       (1,)            Number of spanwise stations (half wing)
    lam_T1      (n_sta, 7)      T1-panels laminate index to retrieve as 'MAT{k}'
    lam_T4      (n_sta, 4)      Flange laminate index to retrieve as 'MAT{k}'
    '''
    sec_data : list[GeomData]
    n_sta    : int = 0
    lam_T1   : np.ndarray = field(default_factory=lambda: np.zeros((0, N_SEG_T1)))
    lam_T4   : np.ndarray = field(default_factory=lambda: np.zeros((0, N_FLANGES)))



# ================================================================================
# PUBLIC API - calls Geometrical propeties calculator to build all sections
# ================================================================================

class SectionBuilder:

    def __init__(
        self,
        opt_vars : object,      # OptVars container
        static_data : object,   # StaticData container
        enable_logging: bool = True,
    ) -> None:
        '''
        Builds cross section properties for every spanwise station.

        Args:
            opt_vars:
                    xw1, xw2     : spar chord fractions
                    bf1..bf4     : flange widths in %c
                    ls1, ls2     : skin layup indices
                    lw1, lw2     : web layup indices
                    lf1..lf4     : flange layup indices
            static_data:
                    lerp_wing_db : interpolated wing geometry (n_sta,)
                    airfoil_db   : dict[str, AirfoilData]
                    material_db  : dict[str, LaminateData]
        '''
        self.log = enable_logging
        self.logger = io.setup_logger(self, enable_logging)

        self.opt = opt_vars
        self.st  = static_data

        self._build_cross_section()

    # ----------------------------------------------------------------
    # Private - Data preparation helpers
    # ----------------------------------------------------------------

    def _cp_index(self, Y_sta: float) -> int:
        '''Return i such that Y_cp[i] <= Y_sta < Y_cp[i+1], clamped.'''
        Y_cp = self.st.lerp_wing_db.Y_cp
        idx  = int(np.searchsorted(Y_cp, Y_sta, side='right')) - 1
        return int(np.clip(idx, 0, len(Y_cp) - 2))

    def _blend_airfoil(
        self,
        Y_sta: float,
    ) -> tuple[Any, ...]:
        '''
        Linearly blend adjacent control-point airfoils at Y_sta.

        Returns:
            Tuple (x_u, y_u, x_l, y_l) of blended adimensional
            airfoil coordinates.
        '''
        wng    = self.st.lerp_wing_db
        afl_db = self.st.airfoil_db
        Y_cp   = wng.Y_cp

        i     = self._cp_index(Y_sta)
        alpha = (Y_sta - Y_cp[i]) / (Y_cp[i + 1] - Y_cp[i] + _TOL)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        afl_A = afl_db[wng.afl_lst[i]]
        afl_B = afl_db[wng.afl_lst[i + 1]]

        _a = np.asarray
        x_u = (1.0 - alpha) * _a(afl_A.x_upper)  + alpha * _a(afl_B.x_upper)
        y_u = (1.0 - alpha) * _a(afl_A.y_upper)  + alpha * _a(afl_B.y_upper)
        x_l = (1.0 - alpha) * _a(afl_A.x_lower)  + alpha * _a(afl_B.x_lower)
        y_l = (1.0 - alpha) * _a(afl_A.y_lower)  + alpha * _a(afl_B.y_lower)

        return x_u, y_u, x_l, y_l

    def _interp_opt_vars(self, Y_sta: float) -> dict[str, float]:
        '''
        Linearly interpolate continuous design variables at Y_sta.
        Applied to: xw1, xw2, bf1, bf2, bf3, bf4.

        Returns:
            Dict with interpolated float values.
        '''
        opt  = self.opt
        Y_cp = self.st.lerp_wing_db.Y_cp
        interp = lambda arr: float(np.interp(Y_sta, Y_cp, arr))

        return {
            'xw1': interp(opt.xw1),
            'xw2': interp(opt.xw2),
            'bf1': interp(opt.bf1),
            'bf2': interp(opt.bf2),
            'bf3': interp(opt.bf3),
            'bf4': interp(opt.bf4),
        }

    def _step_opt_vars(self, Y_sta: float) -> dict[str, int]:
        '''
        Step-function lookup of discrete design variables at Y_sta.
        Applied to: ls1, ls2, lw1, lw2, lf1, lf2, lf3, lf4.

        The value at Y_cp[i] persists for all Y_sta in [Y_cp[i], Y_cp[i+1]).

        Returns:
            Dict with integer layup indices.
        '''
        opt = self.opt
        i   = self._cp_index(Y_sta)

        return {
            'ls1': int(opt.ls1[i]),
            'ls2': int(opt.ls2[i]),
            'lw1': int(opt.lw1[i]),
            'lw2': int(opt.lw2[i]),
            'lf1': int(opt.lf1[i]),
            'lf2': int(opt.lf2[i]),
            'lf3': int(opt.lf3[i]),
            'lf4': int(opt.lf4[i]),
        }

    def _extract_material(
        self,
        lam_idx: int,
    ) -> tuple[float, float, float, float, float, float]:
        '''
        Retrieve membrane and bending engineering constants for a laminate.

        Args:
            lam_idx: Integer key k such that material_db["mat_k"] exists.

        Returns:
            Tuple (thick, E1, E2, G12, E1_bend, E2_bend) from LaminateData.
        '''
        lam = self.st.laminate_db[f'MAT{int(lam_idx)}']
        return (
            float(lam.thick),
            float(lam.E1),
            float(lam.E2),
            float(lam.G12),
            float(lam.E1_bend),
            float(lam.E2_bend),
        )

    def _build_segment_arrays(
        self,
        discrete_opt_vars : dict[str, int],
        continuous_opt_vars : dict[str, float],
        chord_i : float,
    ) -> tuple[tuple, tuple]:
        '''
        Build (7,) material arrays for T1 segments and (4,) flange arrays.

        T1 mapping:
            seg1 -> ls1, seg2..5 -> ls2, seg6 -> lw1, seg7 -> lw2

        Returns:
            T1_mat_props : (t_seg, E1_seg, E2_seg, G_seg,
                           E1_bend_seg, E2_bend_seg)
            T4_mat_props : (t_flange, E1_flange, E2_flange, G_flange, bf,
                           E1_bend_flange, E2_bend_flange)
        '''
        # T1 segment -> layup key mapping (seg1=ls1, seg2..5=ls2,
        # seg6=lw1, seg7=lw2). Aligned with the index passed to _seg
        # in geom_properties._segment_T1.
        seg_layup_keys = ['ls1', 'ls2', 'ls2', 'ls2', 'ls2', 'lw1', 'lw2']
        fln_layup_keys = ['lf1', 'lf2', 'lf3', 'lf4']
        fln_width_keys = ['bf1', 'bf2', 'bf3', 'bf4']

        _a = np.asarray
        seg_lam = _a([discrete_opt_vars[k] for k in seg_layup_keys])
        fln_lam = _a([discrete_opt_vars[k] for k in fln_layup_keys])
        bf      = _a([continuous_opt_vars[k] * chord_i for k in fln_width_keys])

        # Build T1
        t_seg       = np.zeros(N_SEG_T1)
        G_seg       = np.zeros(N_SEG_T1)
        E1_seg      = np.zeros(N_SEG_T1)
        E2_seg      = np.zeros(N_SEG_T1)
        E1_bend_seg = np.zeros(N_SEG_T1)
        E2_bend_seg = np.zeros(N_SEG_T1)
        for k, lam_idx in enumerate(seg_lam):
            t, E1, E2, G12, E1b, E2b = self._extract_material(lam_idx)
            t_seg[k]       = t
            E1_seg[k]      = E1
            E2_seg[k]      = E2
            G_seg[k]       = G12
            E1_bend_seg[k] = E1b
            E2_bend_seg[k] = E2b

        # Build T4
        t_flange       = np.zeros(N_FLANGES)
        E1_flange      = np.zeros(N_FLANGES)
        E2_flange      = np.zeros(N_FLANGES)
        G_flange       = np.zeros(N_FLANGES)
        E1_bend_flange = np.zeros(N_FLANGES)
        E2_bend_flange = np.zeros(N_FLANGES)
        for j, lf_idx in enumerate(fln_lam):
            t, E1, E2, G12, E1b, E2b = self._extract_material(lf_idx)
            t_flange[j]       = t
            E1_flange[j]      = E1
            E2_flange[j]      = E2
            G_flange[j]       = G12
            E1_bend_flange[j] = E1b
            E2_bend_flange[j] = E2b

        T1_mat_props = (t_seg, E1_seg, E2_seg, G_seg,
                        E1_bend_seg, E2_bend_seg)
        T4_mat_props = (t_flange, E1_flange, E2_flange, G_flange, bf,
                        E1_bend_flange, E2_bend_flange)

        self.lam_T1.append(seg_lam)
        self.lam_T4.append(fln_lam)

        return T1_mat_props, T4_mat_props

    # ----------------------------------------------------------------
    # Private - Build all stations
    # ----------------------------------------------------------------

    def _build_cross_section(self) -> list[object]:
        '''
        Execute the full pipeline for every spanwise station defined
        in lerp_wing_db.Y_sta.

        Stores:
            self.sec_data : list[GeomData], one entry per station,
                            ordered root -> tip.
        '''
        wng = self.st.lerp_wing_db

        self.lam_T1 = []
        self.lam_T4 = []

        self.logger.info("Building cross-sections.")
        sec: list[GeomData] = []
        for k, station in enumerate(wng.Y_sta):

            # Blend airfoil
            afl_pts = self._blend_airfoil(station)

            # Interpolate opt-vars
            continuous_opt_vars = self._interp_opt_vars(station)
            discrete_opt_vars   = self._step_opt_vars(station)

            # Build material arrays
            (T1_props, T4_props) = self._build_segment_arrays(
                discrete_opt_vars, continuous_opt_vars, wng.chord[k]
            )

            # Calculate geometrical properties
            calc = GeomPropCalculator(
                afl_pts = afl_pts,
                chord   = float(wng.chord[k]),
                twist   = float(np.degrees(wng.twist[k])),
                Y_sta   = float(station),
                xw1     = continuous_opt_vars['xw1'],
                xw2     = continuous_opt_vars['xw2'],
                T1_props  = T1_props,
                T4_props  = T4_props,
                LE_xz   = wng.LE[k, [0, 2]],
                enable_logging = self.log,
            )
            sec.append(calc.run())

        self.data = SectionData(
            n_sta    = wng.n_sta,
            sec_data = sec,
            lam_T1   = np.array(self.lam_T1),
            lam_T4   = np.array(self.lam_T4),
        )


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
